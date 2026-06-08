"""
Hunyuan3D service — loads and manages all 3D generation models.

Replaces the standalone Hunyuan3D-2GP server by integrating hy3dgen
directly into the Backend.
"""
from __future__ import annotations

import base64
import os
import gc
import logging
import threading
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Any

import torch
import torch.quantization
import numpy as np
from PIL import Image


from app.services.vector_store import VectorStore

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import (
    DegenerateFaceRemover,
    FaceReducer,
    FloaterRemover,
    Hunyuan3DDiTFlowMatchingPipeline,
)
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.system_utils import empty_cache

from app.core.hunyuan3d_config import Hunyuan3DSettings, T2I_STEPS_DEFAULT
from hy3dgen.device_utils import get_device_manager, DeviceManager

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"glb", "obj", "ply", "stl"}


class Hunyuan3DService:
    """Manages all Hunyuan3D model pipelines and provides generation methods."""

    def __init__(self, settings: Hunyuan3DSettings, vector_store: Optional[VectorStore] = None) -> None:
        self.settings = settings
        self.device = settings.device
        self._dm = get_device_manager(settings.device)
        self._dm.setup_globals()
        self._ready = False

        i23d_dtype = self._dm.dtype

        # --- Primary shape generation ---
        logger.info("Loading shape-gen pipeline: %s / %s", settings.model_path, settings.subfolder)
        self.i23d_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            settings.model_path,
            subfolder=settings.subfolder,
            use_safetensors=True,
            device=settings.device,
            dtype=i23d_dtype,
        )
        if settings.enable_flashvdm:
            mc_algo = self._dm.mc_algo if settings.mc_algo == "mc" else settings.mc_algo
            self.i23d_pipeline.enable_flashvdm(mc_algo=mc_algo)
        # VAE slicing: process in smaller chunks → lower peak memory, faster on MPS
        if hasattr(self.i23d_pipeline, 'vae') and hasattr(self.i23d_pipeline.vae, 'use_slicing'):
            self.i23d_pipeline.vae.use_slicing = True
            logger.info("Enabled VAE slicing for shape-gen")

        # --- Multi-view pipeline (deferred — loaded on first use) ---
        self.mv_pipeline: Optional[Any] = None
        self._mv_loaded = False
        self._mv_lock = threading.Lock()

        # --- Post-processors ---
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        self.rembg = BackgroundRemover()

        # --- Texture generation (deferred — loaded on first use) ---
        self.texgen_pipeline: Optional[Any] = None
        self._texgen_loaded = False

        # --- Text-to-image bridge (deferred — loaded on first use) ---
        self.t2i_pipeline: Optional[Any] = None
        self._t2i_loaded = False

        # --- Quantization engine (required on macOS/ARM) ---
        torch.backends.quantized.engine = self._dm.quantization_engine

        # --- Memory optimization (CUDA only) ---
        try:
            from mmgp import offload as mmgp_offload
            if torch.cuda.is_available():
                pipe_dict = mmgp_offload.extract_models("i23d", self.i23d_pipeline)
                mmgp_offload.profile(pipe_dict, profile_no=settings.profile, verboseLevel=settings.verbose)
        except ImportError:
            logger.info("mmgp not available — skipping GPU memory offloading.")

        empty_cache()

        # torch.compile: CUDA and MPS (tracing overhead on first call, faster on subsequent)
        if settings.device == "cuda":
            try:
                self.i23d_pipeline.model = torch.compile(self.i23d_pipeline.model, mode="reduce-overhead")
                logger.info("Applied torch.compile to DiT model (%s)", settings.device)
            except Exception as _exc:
                logger.warning("torch.compile failed for DiT: %s", _exc)
            try:
                self.i23d_pipeline.vae = torch.compile(self.i23d_pipeline.vae, mode="reduce-overhead")
                logger.info("Applied torch.compile to shape-gen VAE")
            except Exception as _exc:
                logger.warning("torch.compile failed for shape-gen VAE: %s", _exc)
            try:
                self.i23d_pipeline.conditioner = torch.compile(self.i23d_pipeline.conditioner, mode="reduce-overhead")
                logger.info("Applied torch.compile to shape-gen conditioner")
            except Exception as _exc:
                logger.warning("torch.compile failed for shape-gen conditioner: %s", _exc)

        # Quantize DiT to int8 on CPU — 2-3x speedup over float32 (quantized models must stay on CPU)
        if settings.enable_quantization and settings.device == "cpu":
            try:
                self.i23d_pipeline.model = self._quantize_model(
                    self.i23d_pipeline.model, "shape-gen DiT", target_device="cpu"
                )
            except Exception as exc:
                logger.warning("Could not quantize shape-gen DiT: %s", exc)

        # --- Vector cache ---
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            try:
                cache_threshold = float(os.environ.get("HY3D_CACHE_THRESHOLD", "0.98"))
                self.vector_store = VectorStore(
                    persist_dir=str(Path(settings.cache_path).parent / "vector_store"),
                    similarity_threshold=cache_threshold,
                )
            except Exception as exc:
                logger.warning("Vector cache unavailable: %s", exc)
                self.vector_store = None


        self._ready = True
        logger.info("Hunyuan3DService ready.")



    def _ensure_mv_loaded(self):
        """Lazy-load multi-view pipeline on first use."""
        if self._mv_loaded or not self.settings.enable_mv:
            return
        logger.info("Lazy-loading multi-view pipeline...")
        self.mv_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.settings.mv_model_path,
            subfolder=self.settings.mv_subfolder,
            use_safetensors=True,
            device="cpu",
            dtype=self._dm.dtype,
        )
        if self.settings.enable_flashvdm:
            mc_algo = self._dm.mc_algo
            self.mv_pipeline.enable_flashvdm(mc_algo=mc_algo)
        # VAE slicing — same as i23d_pipeline
        if hasattr(self.mv_pipeline, 'vae') and hasattr(self.mv_pipeline.vae, 'use_slicing'):
            self.mv_pipeline.vae.use_slicing = True
            logger.info("Enabled VAE slicing for mv pipeline")
        # torch.compile on GPU
        if self.settings.device == "cuda":
            try:
                self.mv_pipeline.model = torch.compile(self.mv_pipeline.model, mode="reduce-overhead")
                logger.info("Applied torch.compile to mv DiT model (%s)", self.settings.device)
            except Exception as exc:
                logger.warning("torch.compile failed for mv DiT: %s", exc)
            try:
                self.mv_pipeline.vae = torch.compile(self.mv_pipeline.vae, mode="reduce-overhead")
                logger.info("Applied torch.compile to mv VAE")
            except Exception as exc:
                logger.warning("torch.compile failed for mv VAE: %s", exc)
            try:
                self.mv_pipeline.conditioner = torch.compile(self.mv_pipeline.conditioner, mode="reduce-overhead")
                logger.info("Applied torch.compile to mv conditioner")
            except Exception as exc:
                logger.warning("torch.compile failed for mv conditioner: %s", exc)
        # int8 quantization on CPU — same as i23d_pipeline
        if self.settings.enable_quantization and self.device == "cpu":
            try:
                self.mv_pipeline.model = self._quantize_model(
                    self.mv_pipeline.model, "mv-DiT", target_device="cpu"
                )
            except Exception as exc:
                logger.warning("Could not quantize mv-DiT: %s", exc)
        self._mv_loaded = True
        logger.info("Multi-view pipeline ready")

    def _ensure_texgen_loaded(self):
        """Lazy-load texture pipeline on first use."""
        if self._texgen_loaded or not self.settings.enable_tex:
            return
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            logger.info("Lazy-loading texture pipeline...")
            self.texgen_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                self.settings.tex_model_path, device=self.device
            )
            self.texgen_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True
            # Attention slicing + VAE slicing for both sub-pipelines (CPU efficiency)
            try:
                pass  # attention_slicing removed — breaks MPS with 52GB OOM
            except Exception as exc:
                logger.warning("Could not enable attention slicing for multiview: %s", exc)
            try:
                _delight = self.texgen_pipeline.models.get("delight_model")
                if _delight and hasattr(_delight, "pipeline"):
                    pass  # attention_slicing removed — breaks MPS
                    if hasattr(_delight.pipeline, "vae"):
                        _delight.pipeline.vae.enable_slicing()
            except Exception as exc:
                logger.warning("Could not enable attention slicing for delight: %s", exc)
            # xFormers memory-efficient attention (GPU only, optional dependency)
            try:
                self.texgen_pipeline.models["multiview_model"].pipeline.enable_xformers_memory_efficient_attention()
                logger.info("Enabled xFormers attention for multiview model")
            except Exception:
                pass  # xFormers not installed — SDPA fallback is fine
            try:
                _dl = self.texgen_pipeline.models.get("delight_model")
                if _dl and hasattr(_dl, "pipeline"):
                    _dl.pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("Enabled xFormers attention for delight model")
            except Exception:
                pass
            # channels_last on VAE components too (conv-heavy, benefits from NHWC)
            try:
                mv_vae = self.texgen_pipeline.models["multiview_model"].pipeline.vae
                mv_vae.to(memory_format=torch.channels_last)
            except Exception:
                pass
            try:
                _dl2 = self.texgen_pipeline.models.get("delight_model")
                if _dl2 and hasattr(_dl2, "pipeline") and hasattr(_dl2.pipeline, "vae"):
                    _dl2.pipeline.vae.to(memory_format=torch.channels_last)
            except Exception:
                pass
            # Texgen models (multiview UNet, delight UNet) must NOT be int8 quantized —
            # diffusion models require full precision for correct color output

            # Opt-in bfloat16 cast — currently DISABLED.
            #
            # The naive cast (unet → bf16, vae stays fp32) crashes on first
            # forward: the VAE-encoded latent enters the UNet as fp32 but
            # encounters bf16 weights at the first matmul (mat1=Float vs
            # mat2=BFloat16). Fixing it cleanly requires casting the entire
            # sub-pipeline including the VAE, which then risks the texture-
            # colour regression that commit 4caf310 already had to fix.
            #
            # Until we have a safer path (e.g. runtime input up-cast at the
            # UNet boundary, or a verified all-bf16 mode that preserves
            # colour), this flag is acknowledged but no cast is performed.
            # Wall-clock improvement targets the step-count flags instead:
            # HY3D_MULTIVIEW_STEPS / HY3D_DELIGHT_STEPS.
            if self.settings.bf16_texgen:
                logger.info(
                    "[bf16-texgen] flag set but cast is currently disabled "
                    "(pipeline dtype-mismatch crash on partial cast, colour "
                    "regression risk on full cast). Set "
                    "HY3D_MULTIVIEW_STEPS / HY3D_DELIGHT_STEPS for the main "
                    "speedup lever."
                )

            self._texgen_loaded = True
            logger.info("Texture pipeline ready")
        except Exception as exc:
            logger.warning("Failed to load texture pipeline: %s", exc)

    def _ensure_t2i_loaded(self, model_override: Optional[str] = None):
        """Lazy-load text-to-image pipeline. If already loaded and model_override
        differs from the currently-loaded model, swap by unloading and reloading.
        """
        if not self.settings.enable_t23d:
            return
        target_model = (model_override or self.settings.t2i_model).lower()
        current_model = getattr(self, "_t2i_current_model", None)
        if self._t2i_loaded and current_model == target_model:
            return
        if self._t2i_loaded and current_model != target_model:
            logger.info("Swapping t2i model: %s → %s", current_model, target_model)
            try:
                if self.t2i_pipeline is not None and hasattr(self.t2i_pipeline, "pipe"):
                    self.t2i_pipeline.pipe.to("cpu")
            except Exception:
                pass
            self.t2i_pipeline = None
            gc.collect()
            empty_cache()
            self._t2i_loaded = False
        try:
            logger.info("Lazy-loading text-to-image pipeline (model=%s)...", target_model)
            # Use the GPU device when not quantizing (qint8 can only run on CPU)
            t2i_device = "cpu" if self.settings.enable_quantization else self.device
            t2i_dtype = torch.float32 if t2i_device == "cpu" else self._dm.dtype
            if target_model == "hunyuan":
                # Fallback: HunyuanDiT v1.2-Distilled (bilingual, slower)
                from hy3dgen.text2image import HunyuanDiTPipeline
                self.t2i_pipeline = HunyuanDiTPipeline(
                    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
                    device=t2i_device,
                    dtype=t2i_dtype,
                )
            else:
                # Default: Hyper-SDXL 4-step (English, faster, higher quality)
                from hy3dgen.text2image import SDXLHyperPipeline
                self.t2i_pipeline = SDXLHyperPipeline(
                    device=t2i_device,
                    dtype=t2i_dtype,
                )
            # Quantize transformer (selective — skip incompatible pooler)
            if self.settings.enable_quantization:
                try:
                    pipe = self.t2i_pipeline.pipe
                    if hasattr(pipe, "transformer"):
                        transformer = pipe.transformer
                        saved_modules = {}
                        for name, mod in transformer.named_modules():
                            if "pooler" in name or "AttentionPool" in type(mod).__name__:
                                saved_modules[name] = mod
                        pipe.transformer = self._quantize_model(transformer, "t2i transformer", target_device="cpu")
                        for name, mod in saved_modules.items():
                            parts = name.split(".")
                            parent = pipe.transformer
                            for p in parts[:-1]:
                                parent = getattr(parent, p)
                            setattr(parent, parts[-1], mod)
                except Exception as exc:
                    logger.warning("Could not quantize t2i: %s", exc)
            # Attention slicing + VAE slicing for t2i
            try:
                t2i_pipe = self.t2i_pipeline.pipe if hasattr(self.t2i_pipeline, 'pipe') else None
                if t2i_pipe is not None:
                    if hasattr(t2i_pipe, 'enable_attention_slicing'):
                        t2i_pipe.enable_attention_slicing(1)
                        logger.info("Enabled attention slicing for t2i pipeline")
                    if hasattr(t2i_pipe, 'vae') and t2i_pipe.vae is not None:
                        if hasattr(t2i_pipe.vae, 'enable_slicing'):
                            t2i_pipe.vae.enable_slicing()
                        if hasattr(t2i_pipe.vae, 'enable_tiling'):
                            t2i_pipe.vae.enable_tiling()
            except Exception as exc:
                logger.warning("Could not enable slicing for t2i: %s", exc)
            self._t2i_loaded = True
            self._t2i_current_model = target_model
            logger.info("Text-to-image pipeline ready (model=%s)", target_model)
        except Exception as exc:
            logger.warning("Failed to load text-to-image pipeline: %s", exc)

    def _resolve_t2i_steps(self) -> int:
        """Pick the step count for the currently-loaded t2i model.

        Order of precedence:
          1. HY3D_T2I_STEPS env var (if set), via settings.t2i_steps
          2. Per-model default from T2I_STEPS_DEFAULT
          3. Hard fallback of 4 (the SDXL value)
        """
        if self.settings.t2i_steps is not None:
            return max(1, int(self.settings.t2i_steps))
        model_name = getattr(self, "_t2i_current_model", None) or self.settings.t2i_model
        return T2I_STEPS_DEFAULT.get(model_name, 4)

    def _unload_pipeline(self, name: str):
        """Unload a pipeline to free RAM."""
        pipe = getattr(self, name, None)
        if pipe is not None:
            del pipe
            setattr(self, name, None)
            gc.collect()
            empty_cache()
            logger.info("Unloaded %s to free RAM", name)

    def _remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background only when needed — skip if image already has transparent pixels."""
        if image.mode == "RGBA":
            alpha = np.array(image.getchannel("A"))
            # Require >5% of pixels to be nearly fully transparent before trusting existing alpha
            transparent_fraction = (alpha < 10).mean()
            if transparent_fraction > 0.05:
                logger.info("Skipping rembg — image already has transparent background (%.0f%% transparent)", transparent_fraction * 100)
                return image
        return self.rembg(image.convert("RGB"))



    def _quantize_model(self, model, name: str, target_device=None):
        """Apply dynamic int8 quantization to Linear layers. Moves to CPU for quantization then back."""
        try:
            original_device = next(model.parameters()).device
            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            # Must quantize on CPU
            model = model.to("cpu")
            quantized = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            dest = target_device or original_device
            # int8 quantized models only run on CPU
            if str(dest) != "cpu":
                logger.info("Quantized %s: %.1fM params -> int8 (stays on CPU, MPS fallback)", name, param_count)
            else:
                logger.info("Quantized %s: %.1fM params -> int8", name, param_count)
            return quantized
        except Exception as exc:
            logger.warning("Failed to quantize %s: %s", name, exc)
            return model


    def _extract_embedding(self, clean_image: Image.Image) -> list:
        """Extract CLIP/DINO embedding from a clean image. ~50ms."""
        try:
            cond_inputs = self.i23d_pipeline.prepare_image(clean_image)
            image_tensor = cond_inputs.pop("image")
            device = next(self.i23d_pipeline.conditioner.parameters()).device
            image_tensor = image_tensor.to(device)
            for k, v in cond_inputs.items():
                if isinstance(v, torch.Tensor):
                    cond_inputs[k] = v.to(device)
            with torch.no_grad():
                cond = self.i23d_pipeline.conditioner(image=image_tensor, **cond_inputs)
            emb = cond["main"].mean(dim=1).squeeze(0)
            emb = emb / emb.norm()
            result = emb.cpu().float().tolist()
            del cond, image_tensor, emb
            return result
        except Exception as exc:
            logger.warning("Embedding extraction failed: %s", exc)
            return None

    def _make_params_hash(self, **kwargs) -> str:
        return VectorStore.compute_params_hash(**kwargs)

    # --- Properties ---
    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def has_texgen(self) -> bool:
        return self.settings.enable_tex

    @property
    def has_t2i(self) -> bool:
        return self.settings.enable_t23d

    @property
    def has_mv(self) -> bool:
        return self.settings.enable_mv

    # --- Mesh post-processing helpers ---
    def _keep_largest_component(self, mesh):
        """Split the mesh into connected components and keep the largest.
        Silently returns the input on any issue (non-fatal)."""
        try:
            import trimesh
            if not isinstance(mesh, trimesh.Trimesh):
                return mesh
            parts = mesh.split(only_watertight=False)
            if len(parts) <= 1:
                return mesh
            largest = max(parts, key=lambda m: len(m.faces))
            removed = sum(len(p.faces) for p in parts) - len(largest.faces)
            logger.info("Dropped %d disconnected components (%d faces); kept %d-face body",
                        len(parts) - 1, removed, len(largest.faces))
            return largest
        except Exception as exc:
            logger.warning("_keep_largest_component failed: %s", exc)
            return mesh

    # --- Memory management (MPS) ---
    def _offload_to_cpu(self, *pipelines: str) -> None:
        if not self._dm.is_gpu:
            return
        import time as _t
        for name in pipelines:
            pipe = getattr(self, name, None)
            if pipe is not None:
                _ts = _t.time()
                if hasattr(pipe, "to"):
                    pipe.to("cpu")
                elif hasattr(pipe, "pipe"):
                    pipe.pipe.to("cpu")
                logger.info("_offload_to_cpu(%s) took %.1f s", name, _t.time() - _ts)
        gc.collect()
        empty_cache()

    def _move_to_device(self, *pipelines: str) -> None:
        if not self._dm.is_gpu:
            return
        for name in pipelines:
            pipe = getattr(self, name, None)
            if pipe is not None:
                if hasattr(pipe, "to"):
                    pipe.to(self.device)
                elif hasattr(pipe, "pipe"):
                    pipe.pipe.to(self.device)
                    pipe.device = self.device



    def _move_texgen_to_device(self):
        """Move texture gen sub-models to GPU for inference."""
        if self.texgen_pipeline is None or not self._dm.is_gpu:
            return
        dev = str(self._dm.device)
        for name, model in self.texgen_pipeline.models.items():
            if hasattr(model, 'pipeline'):
                model.pipeline.to(dev)
            elif hasattr(model, 'to'):
                model.to(dev)
        import gc; gc.collect()
        empty_cache()
        logger.info("Moved texgen to %s", dev)

    def _offload_texgen_to_cpu(self):
        """Move texture gen sub-models back to CPU."""
        if self.texgen_pipeline is None or not self._dm.is_gpu:
            return
        for name, model in self.texgen_pipeline.models.items():
            if hasattr(model, 'pipeline'):
                model.pipeline.to('cpu')
            elif hasattr(model, 'to'):
                model.to('cpu')
        import gc; gc.collect()
        empty_cache()
        logger.info("Moved texgen to CPU")

    # --- Helpers ---
    def _decode_b64_image(self, b64: str) -> Image.Image:
        return Image.open(BytesIO(base64.b64decode(b64)))

    def _get_save_dir(self) -> Path:
        p = Path(self.settings.cache_path)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _export_mesh(self, mesh, uid: str, out_type: str, include_normals: bool = False) -> Dict:
        if mesh is None:
            raise ValueError(
                "Shape generation produced no geometry (mesh is None). "
                "The diffusion output may be degenerate — try a different seed or mc_level."
            )
        save_dir = self._get_save_dir()
        preview_path = save_dir / f"{uid}.glb"

        # Export in background thread for faster response
        import threading
        def _do_export():
            mesh.export(str(preview_path), include_normals=include_normals)
            if out_type != "glb":
                download_path = save_dir / f"{uid}.{out_type}"
                mesh.export(str(download_path), include_normals=include_normals)

        t = threading.Thread(target=_do_export, daemon=True)
        t.start()
        t.join(timeout=30)  # Wait max 10s, usually done in <2s

        return {
            "uid": str(uid),
            "preview_url": f"/api/v1/outputs/{uid}.glb",
            "download_url": f"/api/v1/outputs/{uid}.{out_type}",
            "format": out_type,
        }

    # --- Generation methods ---
    def image_to_3d(
        self,
        image_b64: str,
        seed: int = 1234,
        steps: int = 5,
        guidance_scale: float = 5.0,
        octree_resolution: int = 128,
        num_chunks: int = 8000,
        texture: bool = False,
        face_count: int = 20000,
        output_type: str = "glb",
    ) -> Dict:
        # Cap octree_resolution to 256 — 3-level FlashVDM (>256) has indexing bug
        octree_resolution = min(octree_resolution, 192)

        uid = uuid.uuid4()
        out_type = output_type if output_type in SUPPORTED_FORMATS else "glb"

        image = self._decode_b64_image(image_b64)
        clean_image = self._remove_background(image)
        # DEBUG: save cleaned image to inspect background removal quality
        try:
            import os as _os
            _dbg_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
            _os.makedirs(_dbg_dir, exist_ok=True)
            clean_image.save(_os.path.join(_dbg_dir, 'clean_image.png'))
        except Exception:
            pass
        _t_start = time.time()

        # Each generation is independent — no cross-request parameter modification.
        # (The old "incremental cache" varied seeds/steps based on CLIP similarity,
        # but different images scored as "similar" and got wrong parameters.)
        _embedding = None
        _params_hash = None
        _attempt = 1

        generator = torch.Generator(self.device).manual_seed(seed)
        t0 = time.time()
        with torch.inference_mode():
            outputs = self.i23d_pipeline(
                image=clean_image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                output_type="mesh",
                enable_pbar=False,
            )
        _meshes = export_to_trimesh(outputs)
        mesh = _meshes[0] if _meshes else None
        del outputs
        if mesh is None:
            raise RuntimeError(
                "Shape generation produced no mesh. "
                "The input image may be blank/transparent or too ambiguous. "
                "Try a different seed, increase steps, or use a clearer input image."
            )
        logger.info("Shape gen took %.1f s", time.time() - t0)

        # Clean mesh: remove tiny disconnected floaters + degenerate faces,
        # then keep only the largest connected component (removes stubborn debris
        # that the small-component filter misses — e.g. ball/toy artefacts).
        try:
            _pp_t0 = time.time()
            mesh = self.floater_remover(mesh)
            mesh = self.degenerate_face_remover(mesh)
            mesh = self._keep_largest_component(mesh)
            logger.info("Mesh cleanup took %.2f s", time.time() - _pp_t0)
        except Exception as _e:
            logger.warning("Mesh cleanup failed (non-fatal): %s", _e)

        include_normals = False
        if texture and self.has_texgen:
            self._ensure_texgen_loaded()
            mesh = self.face_reducer(mesh, max_facenum=face_count)
            if self._dm.is_gpu:
                self._offload_to_cpu("i23d_pipeline")
                self._offload_to_cpu("t2i_pipeline")  # free SDXL MPS for multiview
            t0 = time.time()
            with torch.inference_mode():
                mesh = self.texgen_pipeline(mesh, clean_image)
            logger.info("Texture gen took %.1f s", time.time() - t0)
            if self._dm.is_gpu:
                self._move_to_device("i23d_pipeline")
            include_normals = True
        del clean_image, image

        result = self._export_mesh(mesh, uid, out_type, include_normals)

        # --- Vector cache store (with attempt tracking) ---
        result["attempt"] = _attempt
        result["generation_time"] = round(time.time() - _t_start, 1)
        if self.vector_store is not None and _embedding is not None and _params_hash is not None:
            # Delete previous entry for same params so only latest attempt is cached
            if _attempt > 1:
                prev = self.vector_store.search(_embedding, _params_hash)
                if prev:
                    self.vector_store.delete(prev["id"])
            self.vector_store.store(_embedding, _params_hash, result, metadata={"source": "image-to-3d"})

        empty_cache()
        return result

    def text_to_3d(
        self,
        text: str,
        seed: int = 1234,
        steps: int = 5,
        guidance_scale: float = 5.0,
        octree_resolution: int = 128,
        num_chunks: int = 8000,
        texture: bool = False,
        face_count: int = 20000,
        output_type: str = "glb",
        t2i_model: Optional[str] = None,
    ) -> Dict:
        # Cap octree_resolution to 256 — 3-level FlashVDM (>256) has indexing bug
        octree_resolution = min(octree_resolution, 192)
        self._ensure_t2i_loaded(model_override=t2i_model)
        if not self.has_t2i or self.t2i_pipeline is None:
            raise RuntimeError("Text-to-3D is disabled. Enable with HY3D_ENABLE_T23D=true.")

        # FIRST ATTEMPT — full t2i + shape gen
        _t_start = time.time()
        # Move t2i back to device (may have been offloaded to CPU after previous run)
        self._move_to_device("t2i_pipeline")
        t0 = time.time()
        # Step count is model-aware: SDXL=4 (distilled), HunyuanDiT=10 (native).
        # Override globally with HY3D_T2I_STEPS.
        t2i_steps = self._resolve_t2i_steps()
        image = self.t2i_pipeline(text, seed=seed, num_inference_steps=t2i_steps)
        logger.info("Text-to-image took %.1f s (%s, %d steps, attempt #1)",
                    time.time() - t0, self._t2i_current_model, t2i_steps)
        # DEBUG: save t2i output so we can inspect if mesh gen fails
        try:
            import os as _os
            _dbg_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
            _os.makedirs(_dbg_dir, exist_ok=True)
            image.save(_os.path.join(_dbg_dir, 't2i_output.png'))
            logger.info("DEBUG: saved t2i output to %s", _dbg_dir + '/t2i_output.png')
        except Exception as _e:
            logger.warning("Could not save t2i debug image: %s", _e)
        # Free t2i memory before shape gen
        self._offload_to_cpu("t2i_pipeline")

        buf = BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()
        buf.close()

        # Shape gen may fail on ambiguous t2i images — retry with different seeds
        max_retries = 3
        last_error = None
        result = None
        for _try in range(max_retries):
            _seed_try = seed + _try * 1000
            try:
                result = self.image_to_3d(
                    image_b64=image_b64,
                    seed=_seed_try,
                    steps=steps + _try,  # bump steps on retry
                    guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    texture=texture,
                    face_count=face_count,
                    output_type=output_type,
                )
                if _try > 0:
                    logger.info("Text-to-3D succeeded on retry #%d with seed=%d", _try, _seed_try)
                break
            except Exception as _e:
                last_error = _e
                logger.warning("Text-to-3D shape gen failed (attempt %d/%d, seed=%d): %s",
                                _try + 1, max_retries, _seed_try, _e)
        if result is None:
            raise RuntimeError(
                f"Text-to-3D failed after {max_retries} attempts. "
                f"The generated image may lack a clear subject. "
                f"Last error: {last_error}"
            )
        result["attempt"] = 1
        result["generation_time"] = round(time.time() - _t_start, 1)
        result["prompt"] = text
        return result

    def multiview_to_3d(
        self,
        views: Dict[str, str],  # {view_name: base64_string}
        seed: int = 1234,
        steps: int = 5,
        guidance_scale: float = 5.0,
        octree_resolution: int = 128,
        num_chunks: int = 8000,
        texture: bool = False,
        face_count: int = 20000,
        output_type: str = "glb",
    ) -> Dict:
        # Cap octree_resolution to 256 — 3-level FlashVDM (>256) has indexing bug
        octree_resolution = min(octree_resolution, 192)
        self._ensure_mv_loaded()
        if not self.has_mv or self.mv_pipeline is None:
            raise RuntimeError("Multi-view mode is disabled. Enable with HY3D_ENABLE_MV=true.")

        uid = uuid.uuid4()
        out_type = output_type if output_type in SUPPORTED_FORMATS else "glb"

        _t_start = time.time()
        image_dict: Dict[str, Image.Image] = {}
        for view_name, b64 in views.items():
            if b64:
                pil = self._decode_b64_image(b64)
                pil = self._remove_background(pil)
                image_dict[view_name] = pil
                try:
                    import os as _os
                    from hy3dgen.shapegen.preprocessors import ImageProcessorV2 as _IVP
                    _dbg_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'generated', '_debug')
                    _os.makedirs(_dbg_dir, exist_ok=True)
                    pil.save(_os.path.join(_dbg_dir, f'mv_rembg_{view_name}.png'))
                    _proc = _IVP(size=512)
                    _img_np, _ = _proc.load_image(pil, border_ratio=0.15, to_tensor=False)
                    Image.fromarray(_img_np).save(_os.path.join(_dbg_dir, f'mv_model_input_{view_name}.png'))
                except Exception as _e:
                    logger.debug("MV debug save failed: %s", _e)

        # Each generation is independent — no cross-request parameter modification.
        _embedding = None
        _params_hash = None
        _attempt = 1

        # Offload other pipelines for MV
        with self._mv_lock:
            self._offload_to_cpu("i23d_pipeline", "t2i_pipeline")
            self._move_to_device("mv_pipeline")
    
            generator = torch.Generator(self.device).manual_seed(seed)
            t0 = time.time()
            with torch.inference_mode():
                outputs = self.mv_pipeline(
                    image=image_dict,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    octree_resolution=octree_resolution,
                    num_chunks=num_chunks,
                    output_type="mesh",
                    enable_pbar=False,
                )
            mesh = export_to_trimesh(outputs)[0]
            logger.info("MV shape gen took %.1f s", time.time() - t0)
    
            # Clean mesh: remove floaters + degenerate faces + keep largest component
            try:
                _pp_t0 = time.time()
                mesh = self.floater_remover(mesh)
                mesh = self.degenerate_face_remover(mesh)
                mesh = self._keep_largest_component(mesh)
                logger.info("MV mesh cleanup took %.2f s", time.time() - _pp_t0)
            except Exception as _e:
                logger.warning("MV mesh cleanup failed (non-fatal): %s", _e)
    
            include_normals = False
            front_image = image_dict.get("front")
            if texture and self.has_texgen and front_image:
                self._ensure_texgen_loaded()
                mesh = self.face_reducer(mesh, max_facenum=face_count)
                self._offload_to_cpu("mv_pipeline")
                t0 = time.time()
                # Pass all user-provided views so texgen can substitute real back/left/right
                # images instead of hallucinating them from the front image alone.
                mesh = self.texgen_pipeline(mesh, front_image, user_views=image_dict)
                logger.info("Texture gen took %.1f s", time.time() - t0)
                include_normals = True
    
            # Restore primary pipeline
            self._offload_to_cpu("mv_pipeline")
            self._move_to_device("i23d_pipeline")
    
        result = self._export_mesh(mesh, uid, out_type, include_normals)

        # --- Vector cache store (with attempt tracking) ---
        result["attempt"] = _attempt
        result["generation_time"] = round(time.time() - _t_start, 1)
        if self.vector_store is not None and _embedding is not None and _params_hash is not None:
            if _attempt > 1:
                prev = self.vector_store.search(_embedding, _params_hash)
                if prev:
                    self.vector_store.delete(prev["id"])
            self.vector_store.store(_embedding, _params_hash, result, metadata={"source": "multiview-to-3d"})

        empty_cache()
        return result


    def retexture(self, uid: str, prompt: str = "", seed: int = 0, out_type: str = "glb") -> dict:
        """Re-apply texture to an existing mesh from a new text-guided reference image."""
        import trimesh as _trimesh

        if not self.has_texgen:
            raise RuntimeError("Texture pipeline not available — enable with HY3D_ENABLE_TEX=true")
        if not self.has_t2i or self.t2i_pipeline is None:
            raise RuntimeError("Text-to-image pipeline required — enable with HY3D_ENABLE_T23D=true")

        glb_path = Path("generated/3d_outputs") / f"{uid}.glb"
        if not glb_path.exists():
            raise FileNotFoundError(f"Mesh not found: {uid}")

        _t_start = time.time()
        new_uid = uuid.uuid4()

        # Load geometry, strip existing textures so texgen starts clean
        loaded = _trimesh.load(str(glb_path), force="mesh")
        if isinstance(loaded, _trimesh.Scene):
            meshes = [g for g in loaded.geometry.values() if isinstance(g, _trimesh.Trimesh)]
            mesh = _trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        else:
            mesh = loaded
        mesh.visual = _trimesh.visual.ColorVisuals()

        # Generate reference image from new prompt
        self._move_to_device("t2i_pipeline")
        t2i_steps = self._resolve_t2i_steps()
        image = self.t2i_pipeline(prompt, seed=seed, num_inference_steps=t2i_steps)
        logger.info("Retexture: t2i took %.1f s", time.time() - _t_start)
        self._offload_to_cpu("t2i_pipeline")

        clean_image = self._remove_background(image)

        # Apply texture pipeline to existing mesh
        self._ensure_texgen_loaded()
        if self._dm.is_gpu:
            self._offload_to_cpu("i23d_pipeline")
            self._move_texgen_to_device()
        t0 = time.time()
        with torch.inference_mode():
            mesh = self.texgen_pipeline(mesh, clean_image)
        logger.info("Retexture: texgen took %.1f s", time.time() - t0)
        if self._dm.is_gpu:
            self._offload_texgen_to_cpu()

        result = self._export_mesh(mesh, new_uid, out_type, include_normals=True)
        result["generation_time"] = round(time.time() - _t_start, 1)
        result["retextured_from"] = uid

        empty_cache()
        return result


# --- Singleton ---
_service: Optional[Hunyuan3DService] = None


def init_hunyuan3d(settings: Optional[Hunyuan3DSettings] = None, vector_store: Optional[VectorStore] = None) -> Hunyuan3DService:
    global _service
    if settings is None:
        settings = Hunyuan3DSettings()
    _service = Hunyuan3DService(settings, vector_store=vector_store)
    return _service


def get_hunyuan3d() -> Hunyuan3DService:
    assert _service is not None, "Hunyuan3DService not initialized. Call init_hunyuan3d() first."
    return _service

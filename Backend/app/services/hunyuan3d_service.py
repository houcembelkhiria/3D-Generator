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
from typing import Dict, Optional, Any, Callable, Type

import torch
import torch.quantization
import numpy as np
from PIL import Image

# --- Pipeline Loading Utilities ---

def _load_pipeline(pipeline_class: Type[Any], **kwargs) -> Any:
    """Load a pipeline dynamically.
    
    Args:
        pipeline_class: Class defining the pipeline (e.g., Hunyuan3DDiTFlowMatchingPipeline)
        **kwargs: Arguments for pipeline initialization
        
    Returns:
        Loaded pipeline instance
    """
    try:
        pipeline = pipeline_class.from_pretrained(**kwargs)
        return pipeline
    except Exception as exc:
        logger.error(f"Failed to load pipeline: {exc}")
        raise


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


def _wrap_unet_bf16(unet, name: str) -> None:
    """Cast UNet weights to bf16 and patch forward with fp32-boundary casts.

    Standard Diffusers mixed-precision pattern: VAE stays fp32 (colour
    fidelity preserved), UNet runs bf16 (activations and KV cache halve).
    The wrapper transparently up-casts fp32 inputs to bf16 at the UNet
    boundary and down-casts bf16 outputs to fp32 so the next pipeline
    stage (the fp32 VAE decoder) sees the expected dtype.

    Called by `_ensure_texgen_loaded` only when `HY3D_BF16_TEXGEN=1`.
    Default-off; the user opts in once they have verified colour parity.
    """
    unet.to(torch.bfloat16)
    _original_forward = unet.forward

    def _bf16_boundary_forward(*args, **kwargs):
        # Up-cast fp32 tensor inputs at the boundary
        def _cast_in(x):
            if isinstance(x, torch.Tensor) and x.dtype == torch.float32:
                return x.to(torch.bfloat16)
            return x
        new_args = tuple(_cast_in(a) for a in args)
        new_kwargs = {k: _cast_in(v) for k, v in kwargs.items()}

        out = _original_forward(*new_args, **new_kwargs)

        # Down-cast bf16 tensor outputs so the fp32 VAE / next stage works
        def _cast_out(x):
            if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16:
                return x.to(torch.float32)
            return x
        if isinstance(out, torch.Tensor):
            return _cast_out(out)
        # Diffusers UNet outputs often wrap the tensor in a dataclass with
        # a `.sample` field — handle that case without copying the object.
        if hasattr(out, "sample") and isinstance(out.sample, torch.Tensor):
            out.sample = _cast_out(out.sample)
        return out

    unet.forward = _bf16_boundary_forward
    logger.info("bf16-wrapped %s UNet (VAE stays fp32 — no colour regression)", name)


class PipelineManager:
    """Manages dynamic pipeline loading/unloading.
    
    Uses a decorator to simplify pipeline switching.
    """
    def __init__(self, settings: Hunyuan3DSettings, vector_store: Optional[VectorStore] = None):
        self.settings = settings
        self.device = settings.device
        self._dm = get_device_manager(settings.device)
        self.vector_store = vector_store
        self._ready = False
        
    def _load_pipeline(self, pipeline_class: Type[Any], **kwargs) -> Any:
        """Load a pipeline dynamically.
        
        Args:
            pipeline_class: Pipeline class (e.g., Hunyuan3DDiTFlowMatchingPipeline)
            **kwargs: Pipeline initialization arguments
            
        Returns:
            Loaded pipeline
        """
        try:
            pipeline = pipeline_class.from_pretrained(**kwargs)
            return pipeline
        except Exception as exc:
            logger.error(f"Failed to load pipeline: {exc}")
            raise

    def _setup_pipeline(self, pipeline: Any, name: str) -> None:
        """Configure pipeline settings (e.g., VAE slicing, quantization).
        
        Args:
            pipeline: Pipeline instance
            name: Pipeline identifier (e.g., "i23d_pipeline")
        """
        if hasattr(pipeline, 'vae') and hasattr(pipeline.vae, 'use_slicing'):
            pipeline.vae.use_slicing = True
            logger.info(f"Enabled VAE slicing for {name}")
        
        if self.settings.enable_flashvdm:
            mc_algo = self._dm.mc_algo if self.settings.mc_algo == "mc" else self.settings.mc_algo
            pipeline.enable_flashvdm(mc_algo=mc_algo)
            logger.info(f"Enabled FlashVDM for {name}")

    def _quantize_pipeline(self, pipeline: Any, name: str) -> None:
        """Quantize pipeline if CPU and enabled.
        
        Args:
            pipeline: Pipeline instance
            name: Pipeline identifier
        """
        if self.settings.enable_quantization and self.device == "cpu":
            try:
                pipeline.model = self._quantize_model(pipeline.model, name, target_device="cpu")
                logger.info(f"Quantized {name} model (CPU)")
            except Exception as exc:
                logger.warning(f"Failed to quantize {name}: {exc}")

    def _compile_pipeline(self, pipeline: Any, name: str) -> None:
        """Apply torch.compile for faster inference.
        
        Args:
            pipeline: Pipeline instance
            name: Pipeline identifier
        """
        if self.settings.device == "cuda":
            try:
                for component in [pipeline.model, pipeline.vae, pipeline.conditioner]:
                    if hasattr(component, "__class__") and hasattr(component, "__class__.__name__"):
                        torch.compile(component, mode="reduce-overhead")
                        logger.info(f"Applied torch.compile to {name} components")
            except Exception as exc:
                logger.warning(f"Failed to compile {name}: {exc}")


class Hunyuan3DService:
    """Manages all Hunyuan3D model pipelines and provides generation methods.
    
    Uses PipelineManager for dynamic pipeline loading.
    """
    
    def __init__(self, settings: Hunyuan3DSettings, vector_store: Optional[VectorStore] = None) -> None:
        self.settings = settings
        self.device = settings.device
        self._dm = get_device_manager(settings.device)
        self.vector_store = vector_store
        self._ready = False
        self.pipeline_manager = PipelineManager(settings, vector_store)

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

        # --- Primary shape generation pipeline (i23d) ---
        logger.info("Loading i23d pipeline from %s (subfolder=%s)...", settings.model_path, settings.subfolder)
        self.i23d_pipeline = self.pipeline_manager._load_pipeline(
            Hunyuan3DDiTFlowMatchingPipeline,
            model_path=settings.model_path,
            subfolder=settings.subfolder,
            use_safetensors=True,
            device="cpu",
            dtype=self._dm.dtype,
        )
        self.pipeline_manager._setup_pipeline(self.i23d_pipeline, "i23d_pipeline")
        logger.info("i23d pipeline loaded.")

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
        if self._dm.is_gpu:
            self._offload_to_cpu("i23d_pipeline")
        logger.info("Hunyuan3DService ready.")



    def _ensure_mv_loaded(self):
        """Lazy-load multi-view pipeline on first use."""
        if self._mv_loaded or not self.settings.enable_mv:
            return
        logger.info("Lazy-loading multi-view pipeline...")
        self.mv_pipeline = self.pipeline_manager._load_pipeline(
            Hunyuan3DDiTFlowMatchingPipeline,
            model_path=self.settings.mv_model_path,
            subfolder=self.settings.mv_subfolder,
            use_safetensors=True,
            device="cpu",
            dtype=self._dm.dtype,
        )
        self.pipeline_manager._setup_pipeline(self.mv_pipeline, "mv_pipeline")
        self.pipeline_manager._quantize_pipeline(self.mv_pipeline, "mv_pipeline")
        self.pipeline_manager._compile_pipeline(self.mv_pipeline, "mv_pipeline")
        self._mv_loaded = True
        logger.info("Multi-view pipeline ready")

    def _ensure_texgen_loaded(self):
        """Lazy-load texture pipeline on first use."""
        if self._texgen_loaded or not self.settings.enable_tex:
            return
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
            logger.info("Lazy-loading texture pipeline...")
            self.texgen_pipeline = self.pipeline_manager._load_pipeline(
                Hunyuan3DPaintPipeline,
                model_path=self.settings.tex_model_path,
                device=self.device
            )
            self.pipeline_manager._setup_pipeline(self.texgen_pipeline, "texgen_pipeline")
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

            # B4: opt-in bfloat16 UNet cast with fp32 VAE-boundary wrappers.
            #
            # The naive partial cast crashed because the fp32 VAE latent hit
            # bf16 UNet weights at the first matmul. The fix: cast only the
            # UNet weights, then wrap UNet.forward so any fp32 input tensor
            # is up-cast to bf16 at the boundary and any bf16 output is
            # cast back to fp32 before re-entering the VAE. VAE encode/decode
            # stay fp32, so colour fidelity is preserved (the original
            # commit 4caf310 concern). UNet activations + KV cache halve.
            #
            # Targets the multiview 6-view cross-attention which is the
            # largest single MPS buffer in the pipeline (~3 GB → ~1.5 GB).
            # Default OFF (HY3D_BF16_TEXGEN unset) so existing runs are
            # bit-identical; flip on once a side-by-side render confirms
            # texture colour matches on the user's representative assets.
            if self.settings.bf16_texgen:
                try:
                    _wrap_unet_bf16(
                        self.texgen_pipeline.models["multiview_model"].pipeline.unet,
                        "multiview",
                    )
                except Exception as exc:
                    logger.warning("Could not bf16-wrap multiview UNet: %s", exc)
                try:
                    _delight_w = self.texgen_pipeline.models.get("delight_model")
                    if _delight_w and hasattr(_delight_w, "pipeline") and hasattr(_delight_w.pipeline, "unet"):
                        _wrap_unet_bf16(_delight_w.pipeline.unet, "delight")
                except Exception as exc:
                    logger.warning("Could not bf16-wrap delight UNet: %s", exc)

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
                    model_path=self.settings.t2i_hunyuan_dit_model,
                    device=t2i_device,
                    dtype=t2i_dtype,
                )
            else:
                # Default: Hyper-SDXL 4-step (English, faster, higher quality)
                from hy3dgen.text2image import SDXLHyperPipeline
                self.t2i_pipeline = SDXLHyperPipeline(
                    model_path=self.settings.t2i_sdxl_model,
                    vae_path=self.settings.t2i_sdxl_vae,
                    lora_repo=self.settings.t2i_sdxl_lora_repo,
                    lora_weight=self.settings.t2i_sdxl_lora_weight,
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

    def _straighten_straps(self, mesh):
        """Reduce Z-axis banana curvature of strap geometry from single-image depth ambiguity."""
        try:
            import trimesh as _trimesh
            if not isinstance(mesh, _trimesh.Trimesh):
                return mesh
            v = mesh.vertices.copy()
            y_center = (v[:, 1].max() + v[:, 1].min()) / 2.0
            y_range  = v[:, 1].max() - v[:, 1].min()
            # Vertices beyond 20 % of half-height from the Y-center are strap
            case_half = y_range * 0.20
            dist = np.maximum(np.abs(v[:, 1] - y_center) - case_half, 0.0)
            max_dist = dist.max()
            if max_dist < 1e-6:
                return mesh
            case_mask = dist < 1e-6
            z_target = float(v[case_mask, 2].mean()) if case_mask.any() else float(v[:, 2].mean())
            # Moderate ramp: reduce banana curve while preserving real strap depth
            t = (dist / max_dist) ** 0.60
            z_weight = np.clip(t * 0.55, 0.0, 0.55)
            v[:, 2] = v[:, 2] * (1.0 - z_weight) + z_target * z_weight
            # X-width clamp: strap shouldn't be wider than the case (watch lug constraint)
            x_center = (v[:, 0].max() + v[:, 0].min()) / 2.0
            case_x_hw = float(np.abs(v[case_mask, 0] - x_center).max()) if case_mask.any() else 0.0
            if case_x_hw > 1e-6:
                strap_mask = ~case_mask
                strap_x_limit = case_x_hw * 0.88
                strap_indices = np.where(strap_mask)[0]
                excess = np.abs(v[strap_indices, 0] - x_center) - strap_x_limit
                over = excess > 0
                if over.any():
                    oi = strap_indices[over]
                    signs = np.sign(v[oi, 0] - x_center)
                    v[oi, 0] = x_center + signs * strap_x_limit
            mesh = mesh.copy()
            mesh.vertices = v
            logger.info("Strap fix applied: Z-std %.4f → %.4f",
                        mesh.vertices[:, 2].std(), v[:, 2].std())
            return mesh
        except Exception as exc:
            logger.warning("_straighten_straps failed (non-fatal): %s", exc)
            return mesh

    # --- Memory management (MPS) ---
    def _offload_to_cpu(self, *pipelines: str) -> None:
        """Move pipelines to CPU to free GPU/MPS memory.

        On Apple Silicon, CPU and GPU share the same physical memory, so
        "offloading to CPU" just moves tensors from MPS-backed to CPU-backed
        storage. The tensors are *not* freed – they still consume RAM.  For
        Apple Silicon the real benefit is that MPS internal caches are freed
        after empty_cache().
        """
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

    def _release_pipeline(self, attr_name: str) -> None:
        """Fully delete a pipeline from memory and force GC+cache flush.

        Unlike offloading (which keeps the model in CPU RAM), this
        fully releases the memory. The pipeline will be reloaded on next use.
        """
        obj = getattr(self, attr_name, None)
        if obj is not None:
            delattr(self, attr_name)
            del obj
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
            # Respect models that explicitly target CPU (e.g. multiview on MPS —
            # 6-view cross-attention blows past MPS buffer limits at 22.5 GiB)
            target = getattr(model, 'device', dev)
            if target == 'cpu':
                continue
            if hasattr(model, 'pipeline'):
                model.pipeline.to(dev)
            elif hasattr(model, 'to'):
                model.to(dev)
        import gc; gc.collect()
        empty_cache()
        logger.info("Moved texgen to %s (CPU-flagged models kept on CPU)", dev)

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

        if self._dm.is_gpu:
            self._move_to_device("i23d_pipeline")

        gen_device = "cpu" if str(self.device) == "mps" else self.device
        generator = torch.Generator(gen_device).manual_seed(seed)
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
            # B5: aggressively offload every non-texgen pipeline before the
            # 6-view multiview attention kernel allocates its large buffer.
            # On MPS the multiview pass peaks at ~6-8 GB on its own; leaving
            # i23d/t2i/mv resident pushes total occupancy past the
            # per-process watermark and the OS terminates the worker.
            if self._dm.is_gpu:
                self._offload_to_cpu("i23d_pipeline")
                self._offload_to_cpu("t2i_pipeline")
                self._offload_to_cpu("mv_pipeline")  # may be no-op if never loaded
                gc.collect()
                empty_cache()
            t0 = time.time()
            with torch.inference_mode():
                mesh = self.texgen_pipeline(mesh, clean_image)
            logger.info("Texture gen took %.1f s", time.time() - t0)
            # B1: release texgen MPS allocations immediately after the bake.
            # The result dict only carries paths/URLs (B2) — there is no
            # reason for the multiview UNet to stay resident through Celery
            # serialisation and the next idle period.
            if self._dm.is_gpu:
                self._offload_texgen_to_cpu()
                gc.collect()
                empty_cache()
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

        # B2: explicit cleanup of the trimesh + tensors before returning.
        # `result` is a small dict of paths/URLs/uid — bytes already on disk.
        # Letting these objects survive Python GC until Celery serialises the
        # result has historically allowed MPS buffers to live ~30s longer
        # than necessary, enough to cause OOM on the next task.
        del mesh
        if self._dm.is_gpu:
            self._offload_texgen_to_cpu()
            self._release_pipeline("texgen_pipeline")
            gc.collect()
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
        guidance_scale: float = 7.5,
        octree_resolution: int = 256,
        num_chunks: int = 20000,
        texture: bool = False,
        face_count: int = 40000,
        output_type: str = "glb",
    ) -> Dict:
        # FIX #1: only front view provided → delegate to image_to_3d (better checkpoint + retry)
        non_empty = {k for k, v in views.items() if v}
        if non_empty == {"front"}:
            logger.info("MV: only front view provided — delegating to image_to_3d")
            return self.image_to_3d(
                image_b64=views["front"],
                seed=seed, steps=steps, guidance_scale=guidance_scale,
                octree_resolution=octree_resolution, num_chunks=num_chunks,
                texture=texture, face_count=face_count, output_type=output_type,
            )

        uid = uuid.uuid4()
        out_type = output_type if output_type in SUPPORTED_FORMATS else "glb"
        _t_start = time.time()

        # Decode all views; cap size before rembg to avoid OOM on large uploads
        _MAX_VIEW_DIM = 1024
        image_dict: Dict[str, Image.Image] = {}
        for view_name, b64 in views.items():
            if not b64:
                continue
            pil = self._decode_b64_image(b64)
            if max(pil.size) > _MAX_VIEW_DIM:
                pil.thumbnail((_MAX_VIEW_DIM, _MAX_VIEW_DIM), Image.LANCZOS)
            pil = self._remove_background(pil)
            # FIX #3: skip views where rembg clearly failed (all-transparent or all-opaque)
            if pil.mode == "RGBA":
                alpha = np.array(pil.getchannel("A"))
                t_frac = (alpha < 10).mean()
                o_frac = (alpha > 245).mean()
                if t_frac > 0.95 or o_frac > 0.95:
                    logger.warning(
                        "MV view '%s' failed rembg (%.0f%% transparent / %.0f%% opaque) — skipping",
                        view_name, t_frac * 100, o_frac * 100,
                    )
                    continue
            image_dict[view_name] = pil

        front_image = image_dict.get("front")
        if front_image is None:
            raise RuntimeError(
                "Front view missing or failed background removal. "
                "Provide a clearer image with a distinct subject."
            )

        # Shape generation: mv_pipeline primary (1.1B multi-view model, correct for this mode).
        # Official recommended params: octree_resolution=380, num_chunks=20000.
        # i23d (mini 0.6B) is fallback only — mv_pipeline produces superior geometry.
        mesh = None
        last_error = None
        _attempt = 1

        if self.has_mv:
            self._ensure_mv_loaded()
            for _try in range(2):
                _seed_try = seed + _try * 1000
                try:
                    with self._mv_lock:
                        self._offload_to_cpu("i23d_pipeline", "t2i_pipeline")
                        self._move_to_device("mv_pipeline")
                        gen_device = "cpu" if str(self.device) == "mps" else self.device
                        generator = torch.Generator(gen_device).manual_seed(_seed_try)
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
                        _meshes = export_to_trimesh(outputs)
                        _mv_mesh = _meshes[0] if _meshes else None
                        del outputs
                        self._offload_to_cpu("mv_pipeline")
                    if _mv_mesh is not None and len(_mv_mesh.faces) > 500:
                        mesh = _mv_mesh
                        _attempt = _try + 1
                        logger.info("MV shape gen (mv_pipeline) %.1f s, %d faces (attempt %d)",
                                    time.time() - t0, len(mesh.faces), _attempt)
                        break
                    logger.warning("mv_pipeline mesh degenerate (%d faces) — retrying",
                                   len(_mv_mesh.faces) if _mv_mesh else 0)
                except Exception as _e:
                    last_error = _e
                    logger.warning("mv_pipeline attempt %d failed: %s", _try + 1, _e)
                    try:
                        self._offload_to_cpu("mv_pipeline")
                    except Exception:
                        pass

        # Fallback: i23d with front image only
        if mesh is None:
            logger.info("MV falling back to i23d with front image")
            if self._dm.is_gpu:
                self._move_to_device("i23d_pipeline")
            for _try in range(2):
                _seed_try = seed + _try * 1000
                try:
                    gen_device = "cpu" if str(self.device) == "mps" else self.device
                    generator = torch.Generator(gen_device).manual_seed(_seed_try)
                    t0 = time.time()
                    with torch.inference_mode():
                        outputs = self.i23d_pipeline(
                            image=front_image,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            octree_resolution=min(octree_resolution, 192),
                            num_chunks=num_chunks,
                            output_type="mesh",
                            enable_pbar=False,
                        )
                    _meshes = export_to_trimesh(outputs)
                    mesh = _meshes[0] if _meshes else None
                    del outputs
                    if mesh is None:
                        raise RuntimeError("Shape generation produced no mesh.")
                    _attempt = _try + 1
                    logger.info("MV i23d fallback %.1f s (attempt %d)", time.time() - t0, _attempt)
                    break
                except Exception as _e:
                    last_error = _e
                    logger.warning("MV i23d fallback attempt %d failed: %s", _try + 1, _e)

        if mesh is None:
            raise RuntimeError(f"Multiview shape gen failed. Last error: {last_error}")

        # Mesh cleanup
        try:
            _pp_t0 = time.time()
            mesh = self.floater_remover(mesh)
            mesh = self.degenerate_face_remover(mesh)
            mesh = self._keep_largest_component(mesh)
            import trimesh as _tr
            _tr.repair.fill_holes(mesh)
            logger.info("MV mesh cleanup took %.2f s", time.time() - _pp_t0)
        except Exception as _e:
            logger.warning("MV mesh cleanup failed (non-fatal): %s", _e)

        # _straighten_straps disabled: Z-flattening causes loss of overall mesh depth

        include_normals = False
        if texture and self.has_texgen and front_image:
            self._ensure_texgen_loaded()
            mesh = self.face_reducer(mesh, max_facenum=face_count)
            # B5: offload every non-texgen pipeline before the multiview pass.
            # mv_pipeline is the heavy one here — leaving it resident plus
            # texgen multiview UNet doubles peak MPS occupancy.
            if self._dm.is_gpu:
                self._offload_to_cpu("i23d_pipeline")
                self._offload_to_cpu("t2i_pipeline")
                self._offload_to_cpu("mv_pipeline")
                gc.collect()
                empty_cache()
                # Release t2i and mv pipelines fully — they are expensive to
                # keep in RAM and they can be reloaded on demand.
                self._release_pipeline("t2i_pipeline")
            t0 = time.time()
            with torch.inference_mode():
                # Pass all validated views so texgen uses real back/left/right images
                # instead of hallucinating them from the front image alone.
                mesh = self.texgen_pipeline(mesh, front_image, user_views=image_dict)
            logger.info("MV texture gen took %.1f s", time.time() - t0)
            # B1: free texgen MPS immediately after bake.
            if self._dm.is_gpu:
                self._offload_texgen_to_cpu()
                gc.collect()
                empty_cache()
            include_normals = True

        result = self._export_mesh(mesh, uid, out_type, include_normals)
        result["attempt"] = _attempt
        result["generation_time"] = round(time.time() - _t_start, 1)

        # B2: drop mesh + intermediates so Celery doesn't serialise around
        # live MPS allocations.
        del mesh
        if "image_dict" in dir():
            try:
                image_dict.clear()
            except Exception:
                pass
            if self._dm.is_gpu:
                self._offload_texgen_to_cpu()
                self._release_pipeline("texgen_pipeline")
                gc.collect()
                empty_cache()
        return result


    def retexture(self, uid: str, prompt: str = "", seed: int = 0, out_type: str = "glb") -> dict:
        """Re-apply texture to an existing mesh from a new text-guided reference image."""
        import trimesh as _trimesh

        if not self.has_texgen:
            raise RuntimeError("Texture pipeline not available — enable with HY3D_ENABLE_TEX=true")
        if not self.has_t2i:
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

        # Lazy-load t2i pipeline then generate reference image
        self._ensure_t2i_loaded()
        self._move_to_device("t2i_pipeline")
        t2i_steps = self._resolve_t2i_steps()
        image = self.t2i_pipeline(prompt, seed=seed, num_inference_steps=t2i_steps)
        logger.info("Retexture: t2i took %.1f s", time.time() - _t_start)
        self._offload_to_cpu("t2i_pipeline")
        empty_cache()  # flush MPS cache so texgen can allocate large attention buffers

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
    global _service
    if _service is None:
        _service = init_hunyuan3d()
    return _service

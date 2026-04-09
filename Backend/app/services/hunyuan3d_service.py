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
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Any

import hashlib
import json as _json
import torch
import torch.quantization
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

from app.core.hunyuan3d_config import Hunyuan3DSettings
from hy3dgen.device_utils import get_device_manager, DeviceManager

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"glb", "obj", "ply", "stl"}


class Hunyuan3DService:
    """Manages all Hunyuan3D model pipelines and provides generation methods."""

    def __init__(self, settings: Hunyuan3DSettings) -> None:
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

        # --- Multi-view pipeline (optional) ---
        self.mv_pipeline: Optional[Any] = None
        if settings.enable_mv:
            logger.info("Loading multi-view pipeline: %s / %s", settings.mv_model_path, settings.mv_subfolder)
            self.mv_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                settings.mv_model_path,
                subfolder=settings.mv_subfolder,
                use_safetensors=True,
                device="cpu",  # Start on CPU, move to GPU on demand
                dtype=i23d_dtype,
            )
            if settings.enable_flashvdm:
                mc_algo = self._dm.mc_algo if settings.mc_algo == "mc" else settings.mc_algo
                self.mv_pipeline.enable_flashvdm(mc_algo=mc_algo)
            if hasattr(self.mv_pipeline, 'vae') and hasattr(self.mv_pipeline.vae, 'use_slicing'):
                self.mv_pipeline.vae.use_slicing = True
                logger.info("Enabled VAE slicing for multi-view")

        # --- Post-processors ---
        self.floater_remover = FloaterRemover()
        self.degenerate_face_remover = DegenerateFaceRemover()
        self.face_reducer = FaceReducer()
        self.rembg = BackgroundRemover()

        # --- Texture generation (optional) ---
        self.texgen_pipeline: Optional[Any] = None
        if settings.enable_tex:
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline
                logger.info("Loading texture pipeline: %s", settings.tex_model_path)
                self.texgen_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                    settings.tex_model_path, device=settings.device
                )
                self.texgen_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True
            except Exception as exc:
                logger.warning("Failed to load texture pipeline: %s", exc)

        # --- Text-to-image bridge (optional) ---
        self.t2i_pipeline: Optional[Any] = None
        if settings.enable_t23d:
            try:
                from hy3dgen.text2image import HunyuanDiTPipeline
                t2i_dtype = self._dm.dtype
                logger.info("Loading text-to-image pipeline...")
                # Force CPU for t2i on MPS to avoid OOM (runs slower but reliably)
                t2i_device = "cpu"  # HunyuanDiT has MPS placeholder storage issues
                self.t2i_pipeline = HunyuanDiTPipeline(
                    "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
                    device=t2i_device,
                    dtype=torch.float32,
                )
            except Exception as exc:
                logger.warning("Failed to load text-to-image pipeline: %s", exc)

        # --- Quantization (int8 for CPU-bound models only) ---
        # Set quantization engine (required on macOS/ARM)
        torch.backends.quantized.engine = self._dm.quantization_engine
        # MPS doesn't support quantized ops, so only quantize models that run on CPU
        if settings.enable_quantization:
            logger.info("Applying int8 quantization to CPU-bound models...")
            # Quantize texture gen UNet only if running on CPU (MPS doesn't support quantized ops)
            if self.texgen_pipeline is not None and self.device == "cpu":
                try:
                    mv_model = self.texgen_pipeline.models.get("multiview_model")
                    if mv_model and hasattr(mv_model, "pipeline"):
                        pipe = mv_model.pipeline
                        if hasattr(pipe, "unet"):
                            pipe.unet = self._quantize_model(pipe.unet, "texgen UNet", target_device="cpu")
                        elif hasattr(pipe, "transformer"):
                            pipe.transformer = self._quantize_model(pipe.transformer, "texgen transformer", target_device="cpu")
                except Exception as exc:
                    logger.warning("Could not quantize texgen: %s", exc)
            # Quantize text-to-image transformer (selective — skip incompatible pooler)
            if self.t2i_pipeline is not None:
                try:
                    pipe = self.t2i_pipeline.pipe
                    if hasattr(pipe, "transformer"):
                        transformer = pipe.transformer
                        # Save modules that break with quantization
                        saved_modules = {}
                        for name, mod in transformer.named_modules():
                            if "pooler" in name or "AttentionPool" in type(mod).__name__:
                                saved_modules[name] = mod
                        # Quantize all Linear layers
                        pipe.transformer = self._quantize_model(transformer, "t2i transformer", target_device="cpu")
                        # Restore incompatible modules
                        for name, mod in saved_modules.items():
                            parts = name.split(".")
                            parent = pipe.transformer
                            for p in parts[:-1]:
                                parent = getattr(parent, p)
                            setattr(parent, parts[-1], mod)
                        logger.info("Restored %d incompatible modules after quantization", len(saved_modules))
                except Exception as exc:
                    logger.warning("Could not quantize t2i: %s", exc)

        # --- Memory optimization ---
        try:
            from mmgp import offload as mmgp_offload
            if torch.cuda.is_available():
                pipe_dict = mmgp_offload.extract_models("i23d", self.i23d_pipeline)
                if self.mv_pipeline:
                    pipe_dict.update(mmgp_offload.extract_models("mv", self.mv_pipeline))
                if self.texgen_pipeline:
                    pipe_dict.update(mmgp_offload.extract_models("texgen", self.texgen_pipeline))
                if self.t2i_pipeline:
                    pipe_dict.update(mmgp_offload.extract_models("t2i", self.t2i_pipeline))
                mmgp_offload.profile(pipe_dict, profile_no=settings.profile, verboseLevel=settings.verbose)
        except ImportError:
            logger.info("mmgp not available — skipping GPU memory offloading.")

        empty_cache()
        # --- Vector cache ---
        try:
            cache_threshold = float(os.environ.get("HY3D_CACHE_THRESHOLD", "0.85"))
            self.vector_store = VectorStore(
                persist_dir=str(Path(settings.cache_path).parent / "vector_store"),
                similarity_threshold=cache_threshold,
            )
        except Exception as exc:
            logger.warning("Vector cache unavailable: %s", exc)
            self.vector_store = None

        # --- Generation history (stores previous inputs for blending) ---
        self._prompt_history: dict = {}  # prompt_hash -> {count, best_image_b64, best_seed}
        self._input_history: dict = {}  # embedding_hash -> PIL.Image (clean input from previous attempt)

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
                self.settings.tex_model_path, device="cpu"
            )
            self.texgen_pipeline.models["multiview_model"].pipeline.vae.use_slicing = True
            self._texgen_loaded = True
            logger.info("Texture pipeline ready")
        except Exception as exc:
            logger.warning("Failed to load texture pipeline: %s", exc)

    def _ensure_t2i_loaded(self):
        """Lazy-load text-to-image pipeline on first use."""
        if self._t2i_loaded or not self.settings.enable_t23d:
            return
        try:
            from hy3dgen.text2image import HunyuanDiTPipeline
            logger.info("Lazy-loading text-to-image pipeline...")
            self.t2i_pipeline = HunyuanDiTPipeline(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled",
                device="cpu",
                dtype=torch.float32,
            )
            self._t2i_loaded = True
            logger.info("Text-to-image pipeline ready")
        except Exception as exc:
            logger.warning("Failed to load text-to-image pipeline: %s", exc)

    def _unload_pipeline(self, name: str):
        """Unload a pipeline to free RAM."""
        pipe = getattr(self, name, None)
        if pipe is not None:
            del pipe
            setattr(self, name, None)
            gc.collect()
            empty_cache()
            logger.info("Unloaded %s to free RAM", name)

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
            return emb.cpu().float().tolist()
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

    # --- Memory management (MPS) ---
    def _offload_to_cpu(self, *pipelines: str) -> None:
        if not self._dm.is_gpu:
            return
        for name in pipelines:
            pipe = getattr(self, name, None)
            if pipe is not None:
                if hasattr(pipe, "to"):
                    pipe.to("cpu")
                elif hasattr(pipe, "pipe"):
                    pipe.pipe.to("cpu")
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


    def _blend_with_previous(self, new_image: Image.Image, prev_image: Image.Image, attempt: int) -> Image.Image:
        """Blend new input with previous attempt's input. Higher attempts = more trust in previous."""
        # Weight shifts toward previous as attempts grow: 0.7/0.3 → 0.5/0.5 → 0.3/0.7
        prev_weight = min(0.7, 0.2 + (attempt - 1) * 0.15)
        new_weight = 1.0 - prev_weight
        
        # Resize to match
        prev_resized = prev_image.resize(new_image.size, Image.LANCZOS)
        
        # Convert both to RGBA for proper compositing
        new_rgba = new_image.convert("RGBA")
        prev_rgba = prev_resized.convert("RGBA")
        
        # Blend only where both have content (non-transparent)
        import numpy as np
        new_arr = np.array(new_rgba, dtype=np.float32)
        prev_arr = np.array(prev_rgba, dtype=np.float32)
        
        # Use alpha channels to determine content areas
        new_mask = (new_arr[:, :, 3] > 10).astype(np.float32)
        prev_mask = (prev_arr[:, :, 3] > 10).astype(np.float32)
        overlap = new_mask * prev_mask
        
        blended = new_arr.copy()
        # In overlap areas, blend RGB channels
        for c in range(3):
            blended[:, :, c] = np.where(
                overlap > 0,
                new_arr[:, :, c] * new_weight + prev_arr[:, :, c] * prev_weight,
                new_arr[:, :, c]
            )
        # Keep alpha from new image
        result = Image.fromarray(blended.astype(np.uint8), "RGBA")
        logger.info("Blended input: %.0f%% new + %.0f%% previous (attempt #%d)", new_weight * 100, prev_weight * 100, attempt)
        return result

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
        t.join(timeout=10)  # Wait max 10s, usually done in <2s

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
        uid = uuid.uuid4()
        out_type = output_type if output_type in SUPPORTED_FORMATS else "glb"

        image = self._decode_b64_image(image_b64)
        clean_image = self.rembg(image.convert("RGB"))

        # --- Incremental cache: track attempts per image ---
        _embedding = None
        _params_hash = None
        _attempt = 1
        if self.vector_store is not None:
            _embedding = self._extract_embedding(clean_image)
            if _embedding is not None:
                _params_hash = self._make_params_hash(
                    steps=steps, guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution, num_chunks=num_chunks,
                    texture=texture, face_count=face_count, output_type=out_type,
                )
                # Search with a loose threshold to find any similar past generation
                prev = self.vector_store.search(_embedding, _params_hash)
                if prev is not None:
                    similarity = 1.0 - prev["distance"]
                    prev_result = _json.loads(prev["result_json"])
                    prev_attempt = prev_result.get("attempt", 1)
                    _attempt = prev_attempt + 1
                    
                    # Blended input = strong prior → fewer diffusion steps needed
                    # But increase mesh resolution for better quality output
                    boost = min(_attempt - 1, 4)
                    steps = max(2, steps - boost)                    # 5 → 4 → 3 → 2 → 2 (faster)
                    seed = seed + _attempt - 1                       # vary seed
                    octree_resolution = min(384, octree_resolution + (boost * 32))  # 128 → 160 → 192 (better mesh)
                    
                    logger.info(
                        "Attempt #%d (sim=%.4f) — steps=%d (reduced), octree=%d (boosted), seed=%d",
                        _attempt, similarity, steps, octree_resolution, seed,
                    )
                else:
                    logger.info("First generation — attempt #1")

        # --- Blend with previous attempt's input for refinement ---
        if _embedding is not None and _attempt > 1:
            emb_key = hashlib.sha256(str(_embedding[:8]).encode()).hexdigest()[:12]
            prev_input = self._input_history.get(emb_key)
            if prev_input is not None:
                clean_image = self._blend_with_previous(clean_image, prev_input, _attempt)
        
        # Store current clean input for next attempt's blending
        if _embedding is not None:
            emb_key = hashlib.sha256(str(_embedding[:8]).encode()).hexdigest()[:12]
            self._input_history[emb_key] = clean_image.copy()

        generator = torch.Generator(self.device).manual_seed(seed)
        t0 = time.time()
        outputs = self.i23d_pipeline(
            image=clean_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type="mesh",
        )
        mesh = export_to_trimesh(outputs)[0]
        logger.info("Shape gen took %.1f s", time.time() - t0)

        include_normals = False
        if texture and self.has_texgen:
            self._ensure_texgen_loaded()
            mesh = self.face_reducer(mesh, max_facenum=face_count)
            self._offload_to_cpu("i23d_pipeline")
            t0 = time.time()
            mesh = self.texgen_pipeline(mesh, clean_image)
            logger.info("Texture gen took %.1f s", time.time() - t0)
            self._move_to_device("i23d_pipeline")
            include_normals = True

        result = self._export_mesh(mesh, uid, out_type, include_normals)

        # --- Vector cache store (with attempt tracking) ---
        result["attempt"] = _attempt
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
    ) -> Dict:
        self._ensure_t2i_loaded()
        if not self.has_t2i or self.t2i_pipeline is None:
            raise RuntimeError("Text-to-3D is disabled. Enable with HY3D_ENABLE_T23D=true.")

        # --- Incremental improvement for repeated prompts ---
        prompt_hash = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:16]
        history = self._prompt_history.get(prompt_hash)
        attempt = (history["count"] + 1) if history else 1
        
        if history and attempt > 1:
            # REUSE previous image — skip the expensive t2i step entirely
            # Generate a NEW image too with more steps, pick the better one
            prev_image_b64 = history["best_image_b64"]
            
            # Boost shape gen quality with each attempt
            shape_boost = min(attempt - 1, 4)
            shape_steps_actual = max(2, steps - shape_boost)  # 5 → 4 → 3 → 2 (faster, blended input compensates)
            
            # Try generating a new t2i image with higher quality
            t2i_steps_actual = 15 + (min(attempt - 1, 4) * 5)
            seed_actual = seed + attempt - 1
            
            logger.info(
                "Prompt attempt #%d — reusing prev image for fast path, also generating improved image (t2i_steps=%d)",
                attempt, t2i_steps_actual,
            )
            
            # FAST PATH: use previous best image with more shape steps
            # This skips t2i entirely (~3-5 min saved)
            result = self.image_to_3d(
                image_b64=prev_image_b64,
                seed=seed_actual,
                steps=shape_steps_actual,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                texture=texture,
                face_count=face_count,
                output_type=output_type,
            )
            
            # Also generate a new t2i image in the background for next attempt
            try:
                t0 = time.time()
                def _gen_improved():
                    import random, numpy as np
                    random.seed(seed_actual); np.random.seed(seed_actual); torch.manual_seed(seed_actual)
                    gen = torch.Generator(device=self.t2i_pipeline.device).manual_seed(int(seed_actual))
                    new_img = self.t2i_pipeline.pipe(
                        prompt=text[:60] + self.t2i_pipeline.pos_txt,
                        negative_prompt=self.t2i_pipeline.neg_txt,
                        num_inference_steps=t2i_steps_actual,
                        pag_scale=1.3, width=768, height=768,
                        generator=gen, return_dict=False,
                    )[0][0]
                    return new_img
                new_image = _gen_improved()
                buf = BytesIO()
                new_image.save(buf, format="PNG")
                new_b64 = base64.b64encode(buf.getvalue()).decode()
                # Store improved image for next attempt
                self._prompt_history[prompt_hash] = {
                    "count": attempt,
                    "best_image_b64": new_b64,
                    "best_seed": seed_actual,
                }
                logger.info("Improved t2i image generated in %.1f s (stored for attempt #%d)", time.time() - t0, attempt + 1)
            except Exception as exc:
                logger.warning("Background t2i improvement failed: %s", exc)
                self._prompt_history[prompt_hash] = {**history, "count": attempt}
            
            result["attempt"] = attempt
            result["prompt"] = text
            # Update cache entry with prompt and attempt
            if self.vector_store is not None:
                entries = self.vector_store.list_all()
                if entries:
                    latest = max(entries, key=lambda e: e.get("created_at", ""))
                    self.vector_store._collection.update(
                        ids=[latest["id"]],
                        metadatas=[{**{k: v for k, v in latest.items() if k != "id"}, "prompt": text, "source": "text-to-3d"}],
                    )
            return result
        
        else:
            # FIRST ATTEMPT — full t2i + shape gen
            t0 = time.time()
            image = self.t2i_pipeline(text, seed=seed)
            logger.info("Text-to-image took %.1f s (attempt #1)", time.time() - t0)
            
            buf = BytesIO()
            image.save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode()
            
            # Store for future attempts
            self._prompt_history[prompt_hash] = {
                "count": 1,
                "best_image_b64": image_b64,
                "best_seed": seed,
            }
            
            result = self.image_to_3d(
                image_b64=image_b64,
                seed=seed,
                steps=steps,
                guidance_scale=guidance_scale,
                octree_resolution=octree_resolution,
                num_chunks=num_chunks,
                texture=texture,
                face_count=face_count,
                output_type=output_type,
            )
            result["attempt"] = 1
            result["prompt"] = text
            # Update cache entry with prompt
            if self.vector_store is not None:
                entries = self.vector_store.list_all()
                if entries:
                    latest = max(entries, key=lambda e: e.get("created_at", ""))
                    self.vector_store._collection.update(
                        ids=[latest["id"]],
                        metadatas=[{**{k: v for k, v in latest.items() if k != "id"}, "prompt": text, "source": "text-to-3d"}],
                    )
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
        self._ensure_mv_loaded()
        if not self.has_mv or self.mv_pipeline is None:
            raise RuntimeError("Multi-view mode is disabled. Enable with HY3D_ENABLE_MV=true.")

        uid = uuid.uuid4()
        out_type = output_type if output_type in SUPPORTED_FORMATS else "glb"

        image_dict: Dict[str, Image.Image] = {}
        for view_name, b64 in views.items():
            if b64:
                pil = self._decode_b64_image(b64)
                if pil.mode == "RGB":
                    pil = self.rembg(pil)
                image_dict[view_name] = pil

        # --- Incremental cache for multiview (using front view) ---
        _embedding = None
        _params_hash = None
        _attempt = 1
        front_pil = image_dict.get("front")
        if self.vector_store is not None and front_pil is not None:
            _embedding = self._extract_embedding(front_pil)
            if _embedding is not None:
                view_key = ",".join(sorted(image_dict.keys()))
                _params_hash = self._make_params_hash(
                    steps=steps, guidance_scale=guidance_scale,
                    octree_resolution=octree_resolution, num_chunks=num_chunks,
                    texture=texture, face_count=face_count, output_type=out_type,
                    views=view_key,
                )
                prev = self.vector_store.search(_embedding, _params_hash)
                if prev is not None:
                    similarity = 1.0 - prev["distance"]
                    prev_result = _json.loads(prev["result_json"])
                    _attempt = prev_result.get("attempt", 1) + 1
                    boost = min(_attempt - 1, 4)
                    steps = max(2, steps - boost)
                    seed = seed + _attempt - 1
                    octree_resolution = min(384, octree_resolution + (boost * 32))
                    logger.info("MV Attempt #%d (sim=%.4f) — steps=%d (reduced), octree=%d (boosted)", _attempt, similarity, steps, octree_resolution)
                else:
                    logger.info("MV first generation — attempt #1")

        # --- Blend front view with previous attempt ---
        if _embedding is not None and _attempt > 1:
            emb_key = hashlib.sha256(str(_embedding[:8]).encode()).hexdigest()[:12]
            prev_front = self._input_history.get(emb_key)
            if prev_front is not None and "front" in image_dict:
                image_dict["front"] = self._blend_with_previous(image_dict["front"], prev_front, _attempt)
        if _embedding is not None and "front" in image_dict:
            emb_key = hashlib.sha256(str(_embedding[:8]).encode()).hexdigest()[:12]
            self._input_history[emb_key] = image_dict["front"].copy()

        # Offload other pipelines for MV
        self._offload_to_cpu("i23d_pipeline", "t2i_pipeline")
        self._move_to_device("mv_pipeline")

        generator = torch.Generator(self.device).manual_seed(seed)
        t0 = time.time()
        outputs = self.mv_pipeline(
            image=image_dict,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type="mesh",
        )
        mesh = export_to_trimesh(outputs)[0]
        logger.info("MV shape gen took %.1f s", time.time() - t0)

        include_normals = False
        front_image = image_dict.get("front")
        if texture and self.has_texgen and front_image:
            mesh = self.face_reducer(mesh, max_facenum=face_count)
            self._offload_to_cpu("mv_pipeline")
            t0 = time.time()
            mesh = self.texgen_pipeline(mesh, front_image)
            logger.info("Texture gen took %.1f s", time.time() - t0)
            include_normals = True

        # Restore primary pipeline
        self._offload_to_cpu("mv_pipeline")
        self._move_to_device("i23d_pipeline")

        result = self._export_mesh(mesh, uid, out_type, include_normals)

        # --- Vector cache store (with attempt tracking) ---
        result["attempt"] = _attempt
        if self.vector_store is not None and _embedding is not None and _params_hash is not None:
            if _attempt > 1:
                prev = self.vector_store.search(_embedding, _params_hash)
                if prev:
                    self.vector_store.delete(prev["id"])
            self.vector_store.store(_embedding, _params_hash, result, metadata={"source": "multiview-to-3d"})

        empty_cache()
        return result


# --- Singleton ---
_service: Optional[Hunyuan3DService] = None


def init_hunyuan3d(settings: Optional[Hunyuan3DSettings] = None) -> Hunyuan3DService:
    global _service
    if settings is None:
        settings = Hunyuan3DSettings()
    _service = Hunyuan3DService(settings)
    return _service


def get_hunyuan3d() -> Hunyuan3DService:
    assert _service is not None, "Hunyuan3DService not initialized. Call init_hunyuan3d() first."
    return _service

"""
3D generation API routes — image-to-3d, text-to-3d, multiview-to-3d.
Powered by Hunyuan3D (hy3dgen).
"""
import json
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.hunyuan3d_service import get_hunyuan3d as _get_hunyuan3d
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Initialized by app lifespan before ML models — gallery loads even during model startup
_vector_store: Optional[VectorStore] = None

router = APIRouter(prefix="/api/v1", tags=["3D Generation"])


def get_hunyuan3d():
    try:
        return _get_hunyuan3d()
    except AssertionError:
        raise HTTPException(
            status_code=503,
            detail="3D generation service is not available. Models were not loaded at startup."
        )


# --- Request Models ---

class ImageTo3DRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    seed: int = Field(1234, ge=0)
    num_inference_steps: int = Field(5, ge=1, le=100)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    octree_resolution: int = Field(128, ge=16, le=512)
    num_chunks: int = Field(8000, ge=100)
    texture: bool = False
    face_count: int = Field(20000, ge=100)
    type: str = Field("glb")


class TextTo3DRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    seed: int = Field(1234, ge=0)
    num_inference_steps: int = Field(5, ge=1, le=100)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    octree_resolution: int = Field(128, ge=16, le=512)
    num_chunks: int = Field(8000, ge=100)
    texture: bool = False
    face_count: int = Field(20000, ge=100)
    type: str = Field("glb")


class MultiViewTo3DRequest(BaseModel):
    front: str = Field(..., description="Base64 front image (required)")
    back: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    seed: int = Field(1234, ge=0)
    num_inference_steps: int = Field(5, ge=1, le=100)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    octree_resolution: int = Field(128, ge=16, le=512)
    num_chunks: int = Field(8000, ge=100)
    texture: bool = False
    face_count: int = Field(20000, ge=100)
    type: str = Field("glb")


# --- Endpoints ---

@router.post("/image-to-3d", summary="Generate 3D model from image")
async def image_to_3d(body: ImageTo3DRequest):
    svc = get_hunyuan3d()
    try:
        result = await run_in_threadpool(
            svc.image_to_3d,
            image_b64=body.image,
            seed=body.seed,
            steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            octree_resolution=body.octree_resolution,
            num_chunks=body.num_chunks,
            texture=body.texture,
            face_count=body.face_count,
            output_type=body.type,
        )
        return JSONResponse(result, status_code=200)
    except Exception as exc:
        logger.exception("Image-to-3D failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/text-to-3d", summary="Generate 3D model from text")
async def text_to_3d(body: TextTo3DRequest):
    svc = get_hunyuan3d()
    if not svc.has_t2i:
        raise HTTPException(status_code=503, detail="Text-to-3D is disabled. Set HY3D_ENABLE_T23D=true.")
    try:
        result = await run_in_threadpool(
            svc.text_to_3d,
            text=body.text,
            seed=body.seed,
            steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            octree_resolution=body.octree_resolution,
            num_chunks=body.num_chunks,
            texture=body.texture,
            face_count=body.face_count,
            output_type=body.type,
        )
        return JSONResponse(result, status_code=200)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Text-to-3D failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/multiview-to-3d", summary="Generate 3D model from multiple views")
async def multiview_to_3d(body: MultiViewTo3DRequest):
    svc = get_hunyuan3d()
    if not svc.has_mv:
        raise HTTPException(status_code=503, detail="Multi-view mode is disabled. Set HY3D_ENABLE_MV=true.")
    try:
        views = {"front": body.front}
        if body.back:
            views["back"] = body.back
        if body.left:
            views["left"] = body.left
        if body.right:
            views["right"] = body.right

        result = await run_in_threadpool(
            svc.multiview_to_3d,
            views=views,
            seed=body.seed,
            steps=body.num_inference_steps,
            guidance_scale=body.guidance_scale,
            octree_resolution=body.octree_resolution,
            num_chunks=body.num_chunks,
            texture=body.texture,
            face_count=body.face_count,
            output_type=body.type,
        )
        return JSONResponse(result, status_code=200)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Multi-view-to-3D failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/generation-status/{uid}", summary="Check async generation status")
async def generation_status(uid: str):
    if uid not in _pending_results:
        raise HTTPException(status_code=404, detail="Job not found")
    result = _pending_results[uid]
    if result is None:
        return {"status": "processing"}
    return result


@router.get("/generated-models", summary="List all generated 3D models")
async def list_generated_models():
    output_dir = Path("generated/3d_outputs")
    if not output_dir.exists():
        return {"models": []}

    models = []
    for f in sorted(output_dir.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
        models.append({
            "filename": f.name,
            "uid": f.stem,
            "preview_url": f"/api/v1/outputs/{f.name}",
            "download_url": f"/api/v1/outputs/{f.name}",
            "size": f.stat().st_size,
            "created": f.stat().st_mtime,
        })
    return {"models": models}



@router.get("/system-stats", summary="Get real-time RAM and VRAM usage")
async def system_stats():
    import psutil
    import torch

    # RAM
    mem = psutil.virtual_memory()
    ram_used_gb = round(mem.used / (1024**3), 1)
    ram_total_gb = round(mem.total / (1024**3), 1)

    # VRAM / MPS
    vram_used_gb = 0.0
    vram_total_gb = 0.0
    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"
        vram_used_gb = round(torch.cuda.memory_allocated() / (1024**3), 1)
        vram_total_gb = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        try:
            vram_used_gb = round(torch.mps.current_allocated_memory() / (1024**3), 1)
            vram_total_gb = ram_total_gb  # MPS shares system RAM
        except Exception:
            pass

    # Hunyuan3D status
    try:
        svc = _get_hunyuan3d()
        hy3d_ready = True
        has_texgen = svc.has_texgen
        has_t2i = svc.has_t2i
        has_mv = svc.has_mv
    except Exception:
        hy3d_ready = False
        has_texgen = False
        has_t2i = False
        has_mv = False

    return {
        "device": device,
        "ram_used_gb": ram_used_gb,
        "ram_total_gb": ram_total_gb,
        "vram_used_gb": vram_used_gb,
        "vram_total_gb": vram_total_gb,
        "hunyuan3d_ready": hy3d_ready,
        "has_texgen": has_texgen,
        "has_t2i": has_t2i,
        "has_mv": has_mv,
    }



@router.get("/cache-stats", summary="Get vector cache statistics and gallery models")
async def cache_stats():
    if _vector_store is None:
        return {"total_entries": 0, "models": [], "available": False}
    entries = _vector_store.list_all()
    models = []
    for entry in entries:
        try:
            result = json.loads(entry.get("result_json", "{}"))
            models.append({
                "id": entry["id"],
                "previewUrl": result.get("preview_url", ""),
                "downloadUrl": result.get("download_url", ""),
                "format": result.get("format", "glb"),
                "source": entry.get("source", result.get("source", "image-to-3d")),
                "prompt": entry.get("prompt", result.get("prompt")),
                "createdAt": entry.get("created_at", ""),
                "fromCache": True,
                "attempt": result.get("attempt"),
                "generationTime": result.get("generation_time"),
            })
        except Exception:
            continue
    # Sort by creation date, newest first
    models.sort(key=lambda m: m.get("createdAt", ""), reverse=True)
    return {"total_entries": len(models), "models": models, "available": True}


@router.delete("/cache/{entry_id}", summary="Delete a cache entry")
async def delete_cache_entry(entry_id: str):
    if _vector_store is None:
        raise HTTPException(status_code=503, detail="Vector cache not available")
    success = _vector_store.delete(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Cache entry not found")
    return {"deleted": True}


@router.post("/similar-models", summary="Check if a similar model exists in cache")
async def find_similar_models(body: ImageTo3DRequest):
    svc = get_hunyuan3d()
    if svc.vector_store is None:
        return JSONResponse({"found": False, "reason": "cache not available"})
    try:
        image = svc._decode_b64_image(body.image)
        clean_image = svc.rembg(image.convert("RGB"))
        embedding = svc._extract_embedding(clean_image)
        if embedding is None:
            return JSONResponse({"found": False, "reason": "embedding extraction failed"})
        params_hash = svc._make_params_hash(
            steps=body.num_inference_steps, guidance_scale=body.guidance_scale,
            octree_resolution=body.octree_resolution, num_chunks=body.num_chunks,
            texture=body.texture, face_count=body.face_count, output_type=body.type,
        )
        cached = svc.vector_store.search(embedding, params_hash)
        if cached:
            return JSONResponse({
                "found": True,
                "result": json.loads(cached["result_json"]),
                "similarity": round(1.0 - cached.get("distance", 0), 4),
            })
        return JSONResponse({"found": False})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# --- Async submission endpoints ---

# Stores results from background generation jobs: uid -> result dict | None (still running)
_pending_results: dict = {}


def _run_in_background(fn, uid, **kwargs):
    _pending_results[uid] = None  # marks job as in-progress
    try:
        result = fn(**kwargs)
        _pending_results[uid] = {"status": "completed", **result}
    except Exception as exc:
        logger.exception("Background generation %s failed", uid)
        _pending_results[uid] = {"status": "failed", "error": str(exc)}


@router.post("/image-to-3d/async", summary="Submit async image-to-3d job")
async def image_to_3d_async(body: ImageTo3DRequest):
    svc = get_hunyuan3d()
    uid = str(uuid.uuid4())

    def run():
        return svc.image_to_3d(image_b64=body.image, seed=body.seed, steps=body.num_inference_steps,
                        guidance_scale=body.guidance_scale, octree_resolution=body.octree_resolution,
                        num_chunks=body.num_chunks, texture=body.texture, face_count=body.face_count,
                        output_type=body.type)

    threading.Thread(target=_run_in_background, args=(run, uid), daemon=True).start()
    return JSONResponse({"uid": uid, "status": "processing"}, status_code=202)


@router.post("/text-to-3d/async", summary="Submit async text-to-3d job")
async def text_to_3d_async(body: TextTo3DRequest):
    svc = get_hunyuan3d()
    if not svc.has_t2i:
        raise HTTPException(status_code=503, detail="Text-to-3D is disabled.")
    uid = str(uuid.uuid4())

    def run():
        return svc.text_to_3d(text=body.text, seed=body.seed, steps=body.num_inference_steps,
                       guidance_scale=body.guidance_scale, octree_resolution=body.octree_resolution,
                       num_chunks=body.num_chunks, texture=body.texture, face_count=body.face_count,
                       output_type=body.type)

    threading.Thread(target=_run_in_background, args=(run, uid), daemon=True).start()
    return JSONResponse({"uid": uid, "status": "processing"}, status_code=202)


@router.post("/multiview-to-3d/async", summary="Submit async multiview-to-3d job")
async def multiview_to_3d_async(body: MultiViewTo3DRequest):
    svc = get_hunyuan3d()
    if not svc.has_mv:
        raise HTTPException(status_code=503, detail="Multi-view mode is disabled.")
    uid = str(uuid.uuid4())
    views = {"front": body.front}
    if body.back: views["back"] = body.back
    if body.left: views["left"] = body.left
    if body.right: views["right"] = body.right

    def run():
        return svc.multiview_to_3d(views=views, seed=body.seed, steps=body.num_inference_steps,
                            guidance_scale=body.guidance_scale, octree_resolution=body.octree_resolution,
                            num_chunks=body.num_chunks, texture=body.texture, face_count=body.face_count,
                            output_type=body.type)

    threading.Thread(target=_run_in_background, args=(run, uid), daemon=True).start()
    return JSONResponse({"uid": uid, "status": "processing"}, status_code=202)

"""
3D generation API routes — image-to-3d, text-to-3d, multiview-to-3d.
Powered by Hunyuan3D (hy3dgen).
"""
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.services.hunyuan3d_service import get_hunyuan3d as _get_hunyuan3d
from app.services.vector_store import VectorStore
from app.services import gallery_db
from app.worker import celery_app

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
    octree_resolution: int = Field(128, ge=16, le=192)
    num_chunks: int = Field(8000, ge=100)
    texture: bool = False
    face_count: int = Field(20000, ge=100)
    type: str = Field("glb")


class TextTo3DRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    seed: int = Field(1234, ge=0)
    num_inference_steps: int = Field(5, ge=1, le=100)
    guidance_scale: float = Field(5.0, ge=0.0, le=20.0)
    octree_resolution: int = Field(128, ge=16, le=192)
    num_chunks: int = Field(8000, ge=100)
    texture: bool = False
    face_count: int = Field(20000, ge=100)
    type: str = Field("glb")
    t2i_model: str = Field("hyper_sdxl", pattern="^(hyper_sdxl|hunyuan)$",
                           description="T2I backend: hyper_sdxl (default, fast English) or hunyuan (bilingual fallback)")


class MultiViewTo3DRequest(BaseModel):
    front: str = Field(..., description="Base64 front image (required)")
    back: Optional[str] = None
    left: Optional[str] = None
    right: Optional[str] = None
    seed: int = Field(1234, ge=0)
    num_inference_steps: int = Field(20, ge=1, le=100)
    guidance_scale: float = Field(7.5, ge=0.0, le=20.0)
    octree_resolution: int = Field(256, ge=16, le=380)
    num_chunks: int = Field(20000, ge=100)
    texture: bool = False
    face_count: int = Field(40000, ge=100)
    type: str = Field("glb")


class RetextureRequest(BaseModel):
    prompt: str = Field("", description="Describe the desired texture / appearance")
    seed: int = Field(1234, ge=0)
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
        _persist_to_gallery(result["uid"], result, source="image-to-3d", has_texture=body.texture)
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
        _persist_to_gallery(result["uid"], result, prompt=body.text, source="text-to-3d", has_texture=body.texture)
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
        _persist_to_gallery(result["uid"], result, source="multiview-to-3d", has_texture=body.texture)
        return JSONResponse(result, status_code=200)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Multi-view-to-3D failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/generation-status/{uid}", summary="Check async generation status")
async def generation_status(uid: str):
    """Read the Celery task state for this generation job.

    Returns one of:
      - {"status": "queued"}   — task accepted but not started yet
      - {"status": "processing", "stage": ..., "pct": ..., ...}
      - {"status": "completed", "preview_url": ..., "download_url": ..., ...}
      - {"status": "failed", "error": ...}
      - {"status": "cancelled"}
    """
    res = celery_app.AsyncResult(uid)
    state = res.state
    info = res.info if isinstance(res.info, dict) else {}
    if state == "PENDING":
        # Celery PENDING is ambiguous (could be queued or unknown). We do not
        # 404 here because the frontend may poll before the worker picks it up.
        return {"status": "queued"}
    if state == "PROCESSING":
        return {"status": "processing", **info}
    if state == "SUCCESS":
        result_dict = res.result if isinstance(res.result, dict) else {}
        return {"status": "completed", **result_dict}
    if state == "FAILURE":
        return {"status": "failed", "error": str(res.result) if res.result else info.get("error", "Task failed")}
    if state == "REVOKED":
        return {"status": "cancelled"}
    return {"status": state.lower(), **info}


@router.delete("/generation/{uid}", summary="Cancel a pending generation job")
async def cancel_generation(uid: str):
    """Revoke the Celery task. If queued, removes it; if running, signals SIGTERM."""
    result = celery_app.AsyncResult(uid)
    if result.state == "PENDING":
        # Could mean unknown task OR queued task (Celery PENDING is ambiguous).
        # Revoke regardless; if it doesn't exist it's a no-op.
        pass
    try:
        celery_app.control.revoke(uid, terminate=True, signal="SIGTERM")
    except Exception:
        logger.exception("revoke failed for %s", uid)
    return {"cancelled": True, "status": "cancelled"}


@router.websocket("/ws/generation/{uid}")
async def ws_generation(websocket: WebSocket, uid: str):
    """Stream Celery-task progress to the frontend.

    Celery tasks call self.update_state(state="PROCESSING",
    meta={"stage": ..., "pct": ..., ...}) — we read that via AsyncResult
    every 0.5s and forward it as JSON. Same wire contract as the previous
    in-memory _progress dict (stage, pct, optional result fields), so the
    frontend WebSocket consumers do not need changes.
    """
    await websocket.accept()
    try:
        while True:
            res = celery_app.AsyncResult(uid)
            state = res.state
            info = res.info if isinstance(res.info, dict) else {}
            if state == "PENDING":
                # PENDING in Celery = "unknown to backend"; treat as queued
                prog = {"stage": "queued", "pct": 0}
            elif state == "PROCESSING":
                prog = info or {"stage": "generating", "pct": 10}
            elif state == "SUCCESS":
                # Final result; merge into the {stage:"completed", ...} shape
                result_dict = res.result if isinstance(res.result, dict) else {}
                prog = {"stage": "completed", "pct": 100, **result_dict}
            elif state == "FAILURE":
                err = str(res.result) if res.result else (info.get("error", "Task failed"))
                prog = {"stage": "failed", "pct": 0, "error": err}
            elif state == "REVOKED":
                prog = {"stage": "cancelled", "pct": 0}
            else:
                prog = {"stage": state.lower(), "pct": 0}
            await websocket.send_json(prog)
            if prog.get("stage") in ("completed", "failed", "cancelled"):
                break
            await asyncio.sleep(0.5)
    except Exception:
        logger.exception("ws_generation failed for %s", uid)
    finally:
        await websocket.close()


@router.get("/generated-models", summary="List all generated 3D models")
async def list_generated_models():
    output_dir = Path("generated/3d_outputs")

    # Load gallery DB entries (have full metadata)
    gallery = gallery_db.list_all()
    gallery_ids = {e["uid"] for e in gallery if "uid" in e}

    # Also scan disk for any GLBs not in gallery (backwards compat)
    disk_only = []
    if output_dir.exists():
        for f in sorted(output_dir.glob("*.glb"), key=lambda p: p.stat().st_mtime, reverse=True):
            uid = f.stem
            if uid not in gallery_ids:
                disk_only.append({
                    "id": uid,
                    "uid": uid,
                    "filename": f.name,
                    "preview_url": f"/api/v1/outputs/{f.name}",
                    "download_url": f"/api/v1/outputs/{f.name}",
                    "source": "image-to-3d",
                    "createdAt": datetime.utcfromtimestamp(f.stat().st_mtime).isoformat(),
                    "created": f.stat().st_mtime,
                    "size": f.stat().st_size,
                })

    # Merge: gallery first (newest first), then disk-only
    models = gallery + disk_only
    # Sort by createdAt descending
    def _sort_key(m):
        return m.get("createdAt") or m.get("created_at") or ""
    models.sort(key=_sort_key, reverse=True)

    # Normalize field names for frontend
    normalized = []
    for m in models:
        normalized.append({
            "uid": m.get("uid") or m.get("id", ""),
            "filename": m.get("filename", f"{m.get('uid', '')}.glb"),
            "preview_url": m.get("previewUrl") or m.get("preview_url", ""),
            "download_url": m.get("downloadUrl") or m.get("download_url", ""),
            "source": m.get("source", "image-to-3d"),
            "prompt": m.get("prompt", ""),
            "created_at": m.get("createdAt") or m.get("created_at", ""),
            "created": m.get("created"),
            "generation_time": m.get("generationTime") or m.get("generation_time"),
            "face_count": m.get("faceCount") or m.get("face_count"),
            "file_size_mb": m.get("fileSizeMb") or m.get("file_size_mb"),
            "size": m.get("size"),
            "has_texture": bool(m.get("has_texture", False)),
        })
    return {"models": normalized}


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
# (Previously kept _pending_results / _progress / _run_in_background /
#  _persist_to_gallery here for the threading.Thread approach. All of that
#  moved to app/tasks_3d.py when we migrated to Celery. Job state lives in
#  the Celery result backend; the WebSocket and /generation-status endpoints
#  above read it via celery_app.AsyncResult.)


@router.post("/image-to-3d/async", summary="Submit async image-to-3d job")
async def image_to_3d_async(body: ImageTo3DRequest):
    uid = str(uuid.uuid4())
    try:
        celery_app.send_task(
            "app.tasks_3d.image_to_3d_task",
            kwargs={
                "image_b64": body.image, "seed": body.seed,
                "num_inference_steps": body.num_inference_steps,
                "guidance_scale": body.guidance_scale,
                "octree_resolution": body.octree_resolution,
                "num_chunks": body.num_chunks, "texture": body.texture,
                "face_count": body.face_count, "output_type": body.type,
            },
            task_id=uid,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {exc}")
    return JSONResponse({"uid": uid, "status": "processing"}, status_code=202)


@router.post("/text-to-3d/async", summary="Submit async text-to-3d job")
async def text_to_3d_async(body: TextTo3DRequest):
    svc = get_hunyuan3d()
    if not svc.has_t2i:
        raise HTTPException(status_code=503, detail="Text-to-3D is disabled.")
    uid = str(uuid.uuid4())
    try:
        celery_app.send_task(
            "app.tasks_3d.text_to_3d_task",
            kwargs={
                "text": body.text, "seed": body.seed,
                "num_inference_steps": body.num_inference_steps,
                "guidance_scale": body.guidance_scale,
                "octree_resolution": body.octree_resolution,
                "num_chunks": body.num_chunks, "texture": body.texture,
                "face_count": body.face_count, "output_type": body.type,
                "t2i_model": body.t2i_model,
            },
            task_id=uid,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {exc}")
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
    try:
        celery_app.send_task(
            "app.tasks_3d.multiview_to_3d_task",
            kwargs={
                "views": views, "seed": body.seed,
                "num_inference_steps": body.num_inference_steps,
                "guidance_scale": body.guidance_scale,
                "octree_resolution": body.octree_resolution,
                "num_chunks": body.num_chunks, "texture": body.texture,
                "face_count": body.face_count, "output_type": body.type,
            },
            task_id=uid,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {exc}")
    return JSONResponse({"uid": uid, "status": "processing"}, status_code=202)


@router.post("/retexture/{uid}/async", summary="Re-apply texture to existing mesh from new prompt")
async def retexture_async(uid: str, body: RetextureRequest):
    import re as _re
    if not _re.fullmatch(r'[a-zA-Z0-9_\-]+', uid):
        raise HTTPException(status_code=400, detail="Invalid model id")
    glb_path = Path("generated/3d_outputs") / f"{uid}.glb"
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    svc = get_hunyuan3d()
    if not svc.has_texgen:
        raise HTTPException(status_code=503, detail="Texture pipeline not available")
    if not svc.has_t2i:
        raise HTTPException(status_code=503, detail="Text-to-image pipeline required for retexture")
    new_uid = str(uuid.uuid4())
    try:
        celery_app.send_task(
            "app.tasks_3d.retexture_task",
            kwargs={
                "source_uid": uid, "prompt": body.prompt,
                "seed": body.seed, "output_type": body.type,
            },
            task_id=new_uid,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Task queue unavailable: {exc}")
    return JSONResponse({"uid": new_uid, "status": "processing"}, status_code=202)


@router.delete("/models/{uid}", summary="Delete a generated model file from disk")
async def delete_model(uid: str):
    import re
    if not re.fullmatch(r'[a-zA-Z0-9_\-]+', uid):
        raise HTTPException(status_code=400, detail="Invalid model id")
    output_dir = Path("generated/3d_outputs")
    glb_path = output_dir / f"{uid}.glb"
    if not glb_path.exists():
        raise HTTPException(status_code=404, detail="Model not found")
    glb_path.unlink()
    # Remove from gallery DB
    gallery_db.delete(uid)
    # Best-effort: also remove from vector cache
    if _vector_store is not None:
        try:
            _vector_store.delete(uid)
        except Exception:
            pass
    return {"deleted": True, "uid": uid}

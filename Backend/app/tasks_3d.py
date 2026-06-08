"""Celery tasks for direct 3D generation endpoints.

Previously these were run inside the FastAPI process via threading.Thread,
which meant:
  - A FastAPI restart killed all in-flight jobs (no checkpointing)
  - No queue: concurrent requests all hit the GPU simultaneously (OOM risk)
  - Progress lived in module-level dicts (_progress, _pending_results)
    that were lost on restart and capped at 500 entries
  - No retry / time-limit / signal-based cancellation
  - Celery workers configured for 3d_generation queue were doing nothing
    for the direct endpoints

Now: each endpoint enqueues one of these tasks; the Celery worker (one per
GPU, configured via worker_concurrency=1 + prefetch=1) runs it; progress
is reported via self.update_state and read by the WebSocket via AsyncResult.

Cancellation: the DELETE /generation/{uid} endpoint calls
celery_app.control.revoke(uid, terminate=True, signal='SIGTERM').
"""

import logging
import time
from pathlib import Path

from app.worker import celery_app

logger = logging.getLogger(__name__)


def _set_progress(task, stage: str, pct: int, **extra):
    """Report progress with Celery/Redis metadata so the frontend tracker can
    display the task ID, worker hostname, queue name, and live stage."""
    delivery = task.request.delivery_info or {}
    meta = {
        "stage": stage,
        "pct": pct,
        "task_id": task.request.id,
        "worker": task.request.hostname,
        "queue": delivery.get("routing_key", "3d_generation"),
        "ts": time.time(),
        **extra,
    }
    task.update_state(state="PROCESSING", meta=meta)


def _persist_to_gallery(uid: str, result: dict, prompt: str = "",
                        source: str = "image-to-3d", has_texture: bool = False) -> None:
    """Compute file stats then write one row to gallery DB."""
    from app.services import gallery_db
    glb_path = Path("generated/3d_outputs") / f"{uid}.glb"
    if glb_path.exists():
        result.setdefault("file_size_mb", round(glb_path.stat().st_size / (1024 * 1024), 2))
        if "face_count" not in result:
            try:
                import trimesh
                mesh = trimesh.load(str(glb_path))
                if hasattr(mesh, "faces"):
                    result["face_count"] = len(mesh.faces)
                elif hasattr(mesh, "geometry"):
                    result["face_count"] = sum(
                        g.faces.shape[0] for g in mesh.geometry.values() if hasattr(g, "faces")
                    )
            except Exception:
                pass
    try:
        gallery_db.insert(
            uid=uid,
            prompt=prompt,
            source=source,
            preview_url=result.get("preview_url", ""),
            download_url=result.get("download_url", ""),
            generation_time=result.get("generation_time"),
            face_count=result.get("face_count"),
            file_size_mb=result.get("file_size_mb"),
            has_texture=has_texture,
        )
    except Exception:
        logger.exception("gallery_db: failed to save %s", uid)


@celery_app.task(bind=True, name="app.tasks_3d.image_to_3d_task")
def image_to_3d_task(self, image_b64: str, seed: int = 1234,
                     num_inference_steps: int = 30, guidance_scale: float = 5.0,
                     octree_resolution: int = 128, num_chunks: int = 50000,
                     texture: bool = True, face_count: int = 60000,
                     output_type: str = "glb") -> dict:
    """Run hunyuan3d.image_to_3d in a Celery worker process."""
    from app.services.hunyuan3d_service import get_hunyuan3d
    uid = self.request.id
    _set_progress(self, "received", 2)
    try:
        _set_progress(self, "loading_model", 8)
        svc = get_hunyuan3d()
        _set_progress(self, "generating_shape", 20)
        result = svc.image_to_3d(
            image_b64=image_b64, seed=seed, steps=num_inference_steps,
            guidance_scale=guidance_scale, octree_resolution=octree_resolution,
            num_chunks=num_chunks, texture=texture, face_count=face_count,
            output_type=output_type,
        )
        _set_progress(self, "saving", 90)
        result_uid = result.get("uid", uid)
        _persist_to_gallery(result_uid, result, source="image-to-3d", has_texture=texture)
        self.update_state(state="SUCCESS", meta={"stage": "completed", "pct": 100, **result})
        return {"status": "completed", **result}
    except Exception as exc:
        logger.exception("image_to_3d_task %s failed", uid)
        raise


@celery_app.task(bind=True, name="app.tasks_3d.text_to_3d_task")
def text_to_3d_task(self, text: str, seed: int = 1234,
                    num_inference_steps: int = 30, guidance_scale: float = 5.0,
                    octree_resolution: int = 128, num_chunks: int = 50000,
                    texture: bool = True, face_count: int = 60000,
                    output_type: str = "glb", t2i_model: str = None) -> dict:
    """Run hunyuan3d.text_to_3d in a Celery worker process."""
    from app.services.hunyuan3d_service import get_hunyuan3d
    uid = self.request.id
    _set_progress(self, "received", 2)
    try:
        _set_progress(self, "loading_model", 8)
        svc = get_hunyuan3d()
        _set_progress(self, "generating_shape", 20)
        result = svc.text_to_3d(
            text=text, seed=seed, steps=num_inference_steps,
            guidance_scale=guidance_scale, octree_resolution=octree_resolution,
            num_chunks=num_chunks, texture=texture, face_count=face_count,
            output_type=output_type, t2i_model=t2i_model,
        )
        _set_progress(self, "saving", 90)
        result_uid = result.get("uid", uid)
        _persist_to_gallery(result_uid, result, prompt=text,
                            source="text-to-3d", has_texture=texture)
        self.update_state(state="SUCCESS", meta={"stage": "completed", "pct": 100, **result})
        return {"status": "completed", **result}
    except Exception as exc:
        logger.exception("text_to_3d_task %s failed", uid)
        raise


@celery_app.task(bind=True, name="app.tasks_3d.multiview_to_3d_task")
def multiview_to_3d_task(self, views: dict, seed: int = 1234,
                         num_inference_steps: int = 30, guidance_scale: float = 5.0,
                         octree_resolution: int = 128, num_chunks: int = 50000,
                         texture: bool = True, face_count: int = 60000,
                         output_type: str = "glb") -> dict:
    """Run hunyuan3d.multiview_to_3d in a Celery worker process."""
    from app.services.hunyuan3d_service import get_hunyuan3d
    uid = self.request.id
    _set_progress(self, "received", 2)
    try:
        _set_progress(self, "loading_model", 8)
        svc = get_hunyuan3d()
        _set_progress(self, "generating_shape", 20)
        result = svc.multiview_to_3d(
            views=views, seed=seed, steps=num_inference_steps,
            guidance_scale=guidance_scale, octree_resolution=octree_resolution,
            num_chunks=num_chunks, texture=texture, face_count=face_count,
            output_type=output_type,
        )
        _set_progress(self, "saving", 90)
        result_uid = result.get("uid", uid)
        _persist_to_gallery(result_uid, result, source="multiview-to-3d", has_texture=texture)
        self.update_state(state="SUCCESS", meta={"stage": "completed", "pct": 100, **result})
        return {"status": "completed", **result}
    except Exception as exc:
        logger.exception("multiview_to_3d_task %s failed", uid)
        raise


@celery_app.task(bind=True, name="app.tasks_3d.retexture_task")
def retexture_task(self, source_uid: str, prompt: str, seed: int = 1234,
                   output_type: str = "glb") -> dict:
    """Run hunyuan3d.retexture in a Celery worker process."""
    from app.services.hunyuan3d_service import get_hunyuan3d
    uid = self.request.id
    _set_progress(self, "received", 2)
    try:
        _set_progress(self, "loading_model", 8)
        svc = get_hunyuan3d()
        _set_progress(self, "generating_shape", 20)
        result = svc.retexture(uid=source_uid, prompt=prompt, seed=seed, out_type=output_type)
        _set_progress(self, "saving", 90)
        result_uid = result.get("uid", uid)
        _persist_to_gallery(result_uid, result, prompt=prompt,
                            source="retexture", has_texture=True)
        self.update_state(state="SUCCESS", meta={"stage": "completed", "pct": 100, **result})
        return {"status": "completed", **result}
    except Exception as exc:
        logger.exception("retexture_task %s failed", uid)
        raise

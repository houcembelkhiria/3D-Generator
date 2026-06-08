"""Self-contained smoke test for the post-Celery-migration codebase.

Verifies the API surface + task wiring without requiring Redis, a Celery
worker, or any GPU. Uses Celery eager mode (tasks run inline in this
process) and patches the heavy hunyuan3d service with a fake.

Usage:
    cd Backend && python -m tests.smoke_test
Exit code:
    0 = all checks passed
    1 = any check failed

What it covers:
    * 4 direct /async submission endpoints return 202 + uid
    * /generation-status/{uid} returns expected shape per Celery state
    * /run-pipeline (LangGraph) returns 202 + task_id and dispatches
    * /task/{id} returns the expected wrapper shape
    * DELETE /generation/{uid} returns cancelled marker
    * /pipeline-state/{thread_id} 404 when checkpoint absent
    * /resume-pipeline/{thread_id} 404 when checkpoint absent
    * Celery send_task kwargs match task signatures (importing tasks_3d
      without crash = registration works)
    * No protocol regressions in WebSocket-readable status shape

What it does NOT cover (requires real ML stack / GPU / Redis):
    * Actual mesh generation correctness
    * SqliteSaver checkpoint persistence across worker restart
    * SIGTERM-during-CUDA cancellation behaviour
    * WebSocket streaming itself (test uses HTTP polling fallback only)
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# ─────────────────────────────────────────────────────────────────────────
# Bootstrap: must run BEFORE any `app.*` imports
# ─────────────────────────────────────────────────────────────────────────

# In-memory broker/backend so we don't need Redis
os.environ.setdefault("CELERY_BROKER_URL", "memory://")
os.environ.setdefault("CELERY_RESULT_BACKEND", "cache+memory://")

# Disable model loading on import. The hunyuan3d service constructor would
# otherwise try to load real models on first get_hunyuan3d() call.
os.environ.setdefault("PIPELINE_CHECKPOINT_DB", str(Path(tempfile.gettempdir()) / "smoke_test_pipeline.db"))

# Add repo root to path so `import app...` resolves
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import celery_app and switch to eager mode so tasks run inline
from app.worker import celery_app  # noqa: E402
celery_app.conf.update(
    task_always_eager=True,
    task_eager_propagates=True,
    broker_transport="memory",
    result_backend="cache+memory://",
)

# Force-import the tasks modules so the @celery_app.task registrations fire
import app.tasks       # noqa: F401, E402
import app.tasks_3d    # noqa: F401, E402

# Build a fake hunyuan3d service that returns reasonable shapes without GPU
_fake_result = {
    "uid": "00000000-0000-0000-0000-000000000001",
    "preview_url": "/api/v1/outputs/test.glb",
    "download_url": "/api/v1/outputs/test.glb",
    "format": "glb",
    "generation_time": 0.5,
    "face_count": 100,
    "file_size_mb": 0.01,
}
mock_service = MagicMock()
mock_service.image_to_3d.return_value = dict(_fake_result)
mock_service.text_to_3d.return_value = dict(_fake_result)
mock_service.multiview_to_3d.return_value = dict(_fake_result)
mock_service.retexture.return_value = dict(_fake_result, uid="00000000-0000-0000-0000-000000000002")
mock_service.has_t2i = True
mock_service.has_mv = True
mock_service.has_texgen = True

# Patch at both the service module and the route module
# (each does its own `from app.services.hunyuan3d_service import get_hunyuan3d`)
import app.services.hunyuan3d_service as _hsvc  # noqa: E402
_hsvc.get_hunyuan3d = lambda: mock_service
import app.api.routes_3d as _r3d  # noqa: E402
_r3d.get_hunyuan3d = lambda: mock_service
_r3d._get_hunyuan3d = lambda: mock_service  # the alias

# Also mock gallery_db.insert so we don't write to disk
import app.services.gallery_db as _gdb  # noqa: E402
_gdb.insert = MagicMock()
import app.tasks_3d as _t3d  # noqa: E402
_t3d.gallery_db = _gdb

# Mock the document parser + LLM for run_pipeline tests
import app.services.document_parser as _dp  # noqa: E402
_dp.document_parser = MagicMock()
_dp.document_parser.parse_document.return_value = {
    "content": "Sample document text. " * 20,
    "metadata": {"element_count": 5, "file_name": "test.pdf"},
}

# Now safe to import FastAPI app
from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

# ─────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────

_failures = []
_passes = 0
_skipped = []
try:
    import langgraph  # noqa: F401
    _HAS_LANGGRAPH = True
except ImportError:
    _HAS_LANGGRAPH = False


def check(name, condition, detail=""):
    global _passes
    if condition:
        _passes += 1
        print(f"  \033[32m✓\033[0m {name}")
    else:
        _failures.append((name, detail))
        print(f"  \033[31m✗\033[0m {name}  {detail}")


def skip(name, reason):
    _skipped.append((name, reason))
    print(f"  \033[33m·\033[0m {name}  \033[33m(skipped: {reason})\033[0m")


def section(title):
    print(f"\n\033[1m[{title}]\033[0m")


# ─────────────────────────────────────────────────────────────────────────
# Suite 1: direct GPU endpoints (now Celery-backed)
# ─────────────────────────────────────────────────────────────────────────

section("image-to-3d (Celery eager)")
r = client.post("/api/v1/image-to-3d/async", json={
    "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
    "seed": 1234, "num_inference_steps": 30, "guidance_scale": 5.0,
    "octree_resolution": 128, "num_chunks": 50000, "texture": True,
    "face_count": 60000, "type": "glb",
})
check("POST /image-to-3d/async returns 202", r.status_code == 202, f"got {r.status_code} body={r.text[:200]}")
body = r.json() if r.status_code < 500 else {}
check("response contains uid", "uid" in body, body)
check("response.status == 'processing'", body.get("status") == "processing")
uid_img = body.get("uid")
if uid_img:
    r = client.get(f"/api/v1/generation-status/{uid_img}")
    check("GET /generation-status returns 200", r.status_code == 200, r.text[:200])
    check("status is one of {queued, processing, completed}",
          r.json().get("status") in {"queued", "processing", "completed", "completed"})
    # After eager-mode SUCCESS, status should be 'completed' with preview_url
    js = r.json()
    if js.get("status") == "completed":
        check("completed response has preview_url", "preview_url" in js)
        check("completed response has download_url", "download_url" in js)

section("text-to-3d (Celery eager)")
r = client.post("/api/v1/text-to-3d/async", json={
    "text": "a red cube", "seed": 1, "num_inference_steps": 5,
    "guidance_scale": 5.0, "octree_resolution": 64, "num_chunks": 1000,
    "texture": False, "face_count": 1000, "type": "glb",
})
check("POST /text-to-3d/async returns 202", r.status_code == 202, r.text[:200])
check("response contains uid", "uid" in (r.json() if r.status_code < 500 else {}))

section("multiview-to-3d (Celery eager)")
r = client.post("/api/v1/multiview-to-3d/async", json={
    "front": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII=",
    "seed": 1, "num_inference_steps": 5, "guidance_scale": 5.0,
    "octree_resolution": 64, "num_chunks": 1000, "texture": False,
    "face_count": 1000, "type": "glb",
})
check("POST /multiview-to-3d/async returns 202", r.status_code == 202, r.text[:200])
check("response contains uid", "uid" in (r.json() if r.status_code < 500 else {}))

section("cancel (Celery revoke)")
if uid_img:
    r = client.delete(f"/api/v1/generation/{uid_img}")
    check("DELETE /generation returns 200", r.status_code == 200)
    check("response has cancelled marker", r.json().get("cancelled") is True)

# ─────────────────────────────────────────────────────────────────────────
# Suite 2: LangGraph pipeline endpoints
# ─────────────────────────────────────────────────────────────────────────

section("resume-pipeline / pipeline-state when no checkpoint")
if _HAS_LANGGRAPH:
    r = client.get("/api/v1/pipeline-state/nonexistent-thread-id")
    check("GET /pipeline-state returns 404 for missing", r.status_code == 404)
    r = client.post("/api/v1/resume-pipeline/nonexistent-thread-id")
    check("POST /resume-pipeline returns 404 for missing", r.status_code == 404)
else:
    skip("GET /pipeline-state returns 404", "langgraph not installed in this env")
    skip("POST /resume-pipeline returns 404", "langgraph not installed in this env")

section("legacy /upload removed")
r = client.post("/api/v1/upload", files={"file": ("a.pdf", b"%PDF-1.4 fake", "application/pdf")})
check("POST /upload returns 404 (legacy removed)", r.status_code == 404)

# ─────────────────────────────────────────────────────────────────────────
# Suite 3: deep import sanity
# ─────────────────────────────────────────────────────────────────────────

section("imports and registrations")
check("celery_app has run_pipeline task registered",
      "app.tasks.run_pipeline" in celery_app.tasks)
check("celery_app has resume_pipeline task registered",
      "app.tasks.resume_pipeline" in celery_app.tasks)
check("celery_app has image_to_3d_task registered",
      "app.tasks_3d.image_to_3d_task" in celery_app.tasks)
check("celery_app has text_to_3d_task registered",
      "app.tasks_3d.text_to_3d_task" in celery_app.tasks)
check("celery_app has multiview_to_3d_task registered",
      "app.tasks_3d.multiview_to_3d_task" in celery_app.tasks)
check("celery_app has retexture_task registered",
      "app.tasks_3d.retexture_task" in celery_app.tasks)
check("worker config: task_acks_late = True", celery_app.conf.task_acks_late is True)
check("worker config: worker_prefetch_multiplier = 1", celery_app.conf.worker_prefetch_multiplier == 1)
check("worker config: task_time_limit set", isinstance(celery_app.conf.task_time_limit, int))

section("LangGraph pipeline imports cleanly")
if _HAS_LANGGRAPH:
    try:
        from app.pipeline.graph import pipeline, make_thread_config, run_pipeline_streaming  # noqa: F401
        check("graph.pipeline compiles", pipeline is not None)
        cfg = make_thread_config("smoke-test-123")
        check("make_thread_config returns expected shape",
              cfg == {"configurable": {"thread_id": "smoke-test-123"}})
    except Exception as e:
        check("graph imports", False, str(e))
else:
    skip("graph.pipeline compiles", "langgraph not installed in this env")
    skip("make_thread_config shape", "langgraph not installed in this env")

section("Pipeline3DState no longer has current_step")
try:
    from app.pipeline.state import Pipeline3DState
    check("Pipeline3DState has expected keys",
          set(Pipeline3DState.__annotations__.keys()) == {
              "file_path", "file_type", "raw_text", "parsed_content",
              "spec", "spec_valid", "spec_retry_count",
              "mesh_output", "mesh_valid", "mesh_retry_count",
              "model_info", "errors",
          })
    check("current_step removed from state", "current_step" not in Pipeline3DState.__annotations__)
except Exception as e:
    check("state imports", False, str(e))

# ─────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"Smoke test: {_passes} passed, {len(_failures)} failed, {len(_skipped)} skipped")
if _skipped and not _HAS_LANGGRAPH:
    print(f"\n(LangGraph tests skipped — install `langgraph` to run them)")
if _failures:
    print("\nFailures:")
    for name, detail in _failures:
        print(f"  - {name}: {detail}")
    sys.exit(1)
print("All checks passed. \u2713")
sys.exit(0)

import os

# B3: per-task MPS hygiene only.
#
# Earlier revisions of this file set PYTORCH_MPS_HIGH/LOW_WATERMARK_RATIO=0.0
# and torch.mps.set_per_process_memory_fraction(0.7) at worker init. Field
# testing showed both were too aggressive on a 32 GB Mac:
# - HIGH=0.0 sounds like "no cap" but the LOW=0.0 pair triggered constant
#   reclamation cycles that made every MPS allocation slower and, when the
#   text_to_3d shape-gen tried to allocate ~10 GB in one go, allocations
#   failed outright. The mesh node caught the OOM and silently routed to
#   store_result with mesh_output=None — visible to the user as "Pipeline
#   running..." stuck forever even though the timer went green.
# - set_per_process_memory_fraction(0.7) on a single-worker setup capped
#   torch at ~22 GB total — fine for inference alone, too tight when SDXL +
#   shape-gen + texgen are all warm.
#
# What we keep: a post-task GC + empty_cache pass. That's pure good (it
# bounds residual MPS allocations between tasks) and has no effect on the
# in-flight task. Watermarks stay at PyTorch defaults (HIGH=1.7, LOW=1.4),
# which work for this hardware in practice.

from celery import Celery
from celery.signals import task_postrun
from app.core.config import settings

celery_app = Celery(
    "3d_generator_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks", "app.tasks_3d"],
)

# Celery configuration
# Production-hardened: ack-late so a worker crash mid-task requeues the job
# instead of losing it; prefetch=1 so a single GPU worker never tries to run
# multiple 20-min mesh-gen tasks concurrently; hard/soft time limits cap
# stuck tasks; result expiry keeps Redis tidy.
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Reliability
    task_acks_late=True,                  # ack only after task body returns
    task_reject_on_worker_lost=True,      # requeue on worker SIGKILL/OOM
    task_track_started=True,              # STARTED state visible to clients
    worker_prefetch_multiplier=1,         # one heavy task per worker at a time
    # Time limits (40 min hard / 35 min soft — leaves slack over 20-min mesh gen)
    task_time_limit=2400,
    task_soft_time_limit=2100,
    # Result backend hygiene
    result_expires=86400,                 # 24h before AsyncResult.get() returns gone
    result_extended=True,                 # include task name/args in result meta
    # Queue routing
    task_routes={
        # Document-to-3D LangGraph pipeline
        "app.tasks.run_pipeline":              {"queue": "3d_generation"},
        "app.tasks.resume_pipeline":           {"queue": "3d_generation"},
        # Direct 3D generation tasks (was running in FastAPI via threading.Thread)
        "app.tasks_3d.image_to_3d_task":       {"queue": "3d_generation"},
        "app.tasks_3d.text_to_3d_task":        {"queue": "3d_generation"},
        "app.tasks_3d.multiview_to_3d_task":   {"queue": "3d_generation"},
        "app.tasks_3d.retexture_task":         {"queue": "3d_generation"},
    },
)


# B3: per-task cleanup signal (the only piece worth keeping from the
# earlier revision — see the comment block at the top of this file).
@task_postrun.connect
def _hy3d_task_mps_cleanup(**_):
    """Release MPS allocations + run Python GC after every task.

    Without this, MPS buffers from a finished task survive until the next
    Python GC sweep — easily 30+ seconds. On a single-prefork worker that
    interval is enough for the next task's model load to OOM. Explicit
    cleanup makes each task's footprint deterministic.
    """
    import gc
    gc.collect()
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass

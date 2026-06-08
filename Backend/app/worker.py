from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "3d_generator_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"]
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
        "app.tasks.process_document": {"queue": "document_processing"},
        "app.tasks.generate_3d_model": {"queue": "3d_generation"},
        "app.tasks.run_pipeline":      {"queue": "3d_generation"},
        "app.tasks.resume_pipeline":   {"queue": "3d_generation"},
    },
)
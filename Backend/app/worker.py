from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "3d_generator_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_routes={
        "app.tasks.process_document": {"queue": "document_processing"},
        "app.tasks.generate_3d_model": {"queue": "3d_generation"},
    },
)
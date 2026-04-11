import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api import routes
from app.api import routes_3d
from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load VectorStore first (lightweight), then Hunyuan3D models (heavy)."""
    logger.info("=== 3D Generator Backend starting ===")

    from app.core.hunyuan3d_config import Hunyuan3DSettings
    from app.services.vector_store import VectorStore

    hy3d_settings = Hunyuan3DSettings()

    # Step 1: VectorStore — lightweight, always available for gallery even if ML models fail
    try:
        vs_path = str(Path(hy3d_settings.cache_path).parent / "vector_store")
        threshold = float(os.environ.get("HY3D_CACHE_THRESHOLD", "0.85"))
        routes_3d._vector_store = VectorStore(persist_dir=vs_path, similarity_threshold=threshold)
        logger.info("VectorStore ready at %s", vs_path)
    except Exception as exc:
        logger.warning("VectorStore init failed: %s", exc)

    # Step 2: ML models — heavy, 3D generation endpoints return 503 if this fails
    try:
        from app.services.hunyuan3d_service import init_hunyuan3d

        logger.info("Device: %s", hy3d_settings.device)
        logger.info("Model: %s / %s", hy3d_settings.model_path, hy3d_settings.subfolder)
        Path(hy3d_settings.cache_path).mkdir(parents=True, exist_ok=True)
        init_hunyuan3d(hy3d_settings, vector_store=routes_3d._vector_store)
    except Exception as exc:
        logger.warning(
            "Hunyuan3D models could not be loaded: %s. "
            "3D generation endpoints will return 503. "
            "All other endpoints work normally.",
            exc,
        )

    yield
    logger.info("=== 3D Generator Backend shutting down ===")


tags_metadata = [
    {"name": "System", "description": "System health and status endpoints"},
    {"name": "Document Processing", "description": "Upload and process documents (PDF, Email) for 3D generation"},
    {"name": "Task Management", "description": "Monitor and manage asynchronous processing tasks"},
    {"name": "3D Models", "description": "Access and list generated 3D models"},
    {"name": "3D Generation", "description": "Generate 3D models from images, text, or multi-view inputs"},
    {"name": "PDF Tools", "description": "PDF utility operations including text extraction"},
]

app = FastAPI(
    title="3D Generator API",
    description="Unified API for document processing and 3D model generation using Hunyuan3D.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Existing routes (document processing, upload, extract, etc.)
app.include_router(routes.router, prefix="/api/v1")

# 3D generation routes (image-to-3d, text-to-3d, multiview-to-3d)
app.include_router(routes_3d.router)




@app.get("/", tags=["System"])
async def root():
    return {"message": "3D Generator API is running", "version": "2.0.0"}


@app.get("/health", tags=["System"])
async def health_check():
    try:
        from app.services.hunyuan3d_service import _service
        hy3d_ready = _service.is_ready if _service else False
    except Exception:
        hy3d_ready = False
    return {"status": "healthy", "hunyuan3d_ready": hy3d_ready}


# Serve generated 3D model files (MUST be after all route registrations)
_outputs_dir = Path("generated/3d_outputs")
_outputs_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/v1/outputs", StaticFiles(directory=str(_outputs_dir)), name="outputs")

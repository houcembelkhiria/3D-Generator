from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes
from app.core.config import settings

# Custom tags metadata for organized documentation
tags_metadata = [
    {
        "name": "System",
        "description": "System health and status endpoints",
    },
    {
        "name": "Document Processing",
        "description": "Upload and process documents (PDF, Email) for 3D generation",
    },
    {
        "name": "Task Management",
        "description": "Monitor and manage asynchronous processing tasks",
    },
    {
        "name": "3D Models",
        "description": "Access and list generated 3D models",
    },
    {
        "name": "PDF Tools",
        "description": "PDF utility operations including text extraction",
    },
]

app = FastAPI(
    title="3D Generator API",
    description="""
    # 3D Generator API
    
    A comprehensive API for generating 3D objects from documents using AI/ML processing.
    
    ## Features
    
    * **Document Upload**: Support for PDF and Email files
    * **Async Processing**: Background task processing with Celery
    * **3D Generation**: Convert documents to 3D models (GLB, OBJ, STL formats)
    * **PDF Tools**: Extract text and process PDF documents
    * **Real-time Monitoring**: Track task progress and status
    
    ## Architecture
    
    * **FastAPI**: High-performance async web framework
    * **Celery**: Distributed task queue for background processing
    * **Redis**: Message broker and result backend
    * **Docker**: Containerized deployment
    
    ## Getting Started
    
    1. Upload a document using `POST /api/v1/upload`
    2. Monitor processing status with `GET /api/v1/task/{task_id}`
    3. Download generated 3D models from `GET /api/v1/models`
    
    ## Supported File Types
    
    * PDF documents (`application/pdf`)
    * Email messages (`message/rfc822`)
    
    ## Rate Limits
    
    * Maximum file size: 50MB
    * Concurrent processing: Configurable via Celery workers
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    contact={
        "name": "3D Generator Team",
        "url": "https://github.com/your-org/3d-generator",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(routes.router, prefix="/api/v1")

@app.get("/", tags=["System"], summary="Root endpoint", response_description="API welcome message")
async def root():
    """
    Root endpoint that returns a welcome message.
    
    Use this to verify the API is running and accessible.
    """
    return {"message": "3D Generator API is running"}

@app.get("/health", tags=["System"], summary="Health check", response_description="Service health status")
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns the current health status of the API service.
    
    - **status**: Current health status (healthy/unhealthy)
    """
    return {"status": "healthy"}
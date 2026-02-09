from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import os
from app.worker import celery_app
from app.core.config import settings

router = APIRouter()

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskResult(BaseModel):
    task_id: str
    status: str
    result: Optional[dict]

@router.post("/upload", response_model=TaskResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload document for 3D generation processing"""
    
    # Validate file type
    allowed_types = ["application/pdf", "message/rfc822"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail="Only PDF and Email files are supported"
        )
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    file_extension = file.filename.split(".")[-1] if "." in file.filename else ""
    filename = f"{file_id}.{file_extension}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Save file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Start Celery task
    task = celery_app.send_task(
        "app.tasks.process_document",
        args=[file_path, file.content_type],
        task_id=file_id
    )
    
    return TaskResponse(
        task_id=file_id,
        status="processing",
        message="File uploaded successfully. Processing started."
    )

@router.get("/task/{task_id}", response_model=TaskResult)
async def get_task_status(task_id: str):
    """Get the status of a processing task"""
    task = celery_app.AsyncResult(task_id)
    
    if task.state == "PENDING":
        return TaskResult(
            task_id=task_id,
            status="pending",
            result=None
        )
    elif task.state == "PROCESSING":
        return TaskResult(
            task_id=task_id,
            status="processing",
            result=task.info
        )
    elif task.state == "SUCCESS":
        return TaskResult(
            task_id=task_id,
            status="completed",
            result=task.result
        )
    elif task.state == "FAILURE":
        return TaskResult(
            task_id=task_id,
            status="failed",
            result={"error": str(task.info)}
        )
    else:
        return TaskResult(
            task_id=task_id,
            status=task.state.lower(),
            result=task.info
        )

@router.get("/models")
async def list_generated_models():
    """List all generated 3D models"""
    models = []
    if os.path.exists(settings.GENERATED_DIR):
        for filename in os.listdir(settings.GENERATED_DIR):
            if filename.endswith((".glb", ".obj", ".stl")):
                file_path = os.path.join(settings.GENERATED_DIR, filename)
                models.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "created": os.path.getctime(file_path)
                })
    return {"models": models}
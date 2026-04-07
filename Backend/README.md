# 3D Generator Backend

This is the backend service for the 3D Generator application, built with FastAPI and Celery.

## Features

- FastAPI REST API
- Celery for background task processing
- Redis for task queue and caching
- Document processing (PDF/Email)
- 3D model generation pipeline

## Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the development server:
```bash
uvicorn app.main:app --reload
```

3. Run Celery worker:
```bash
celery -A app.worker worker --loglevel=info
```

## API Endpoints

- `POST /api/v1/upload` - Upload document for 3D generation
- `GET /api/v1/task/{task_id}` - Check task status
- `GET /api/v1/models` - List generated 3D models
- `POST /api/v1/extract-text/` - Extract text from PDF files

## API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
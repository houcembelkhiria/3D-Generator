# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3D Generator is a document-to-3D-model pipeline. Users upload PDFs/emails, text is extracted and processed by an LLM to generate 3D specifications, then 3D meshes are generated and converted to various formats (GLB, OBJ, STL, GLTF).

## Architecture

**Backend** (FastAPI + Celery + Redis): `Backend/app/`
- `main.py` — FastAPI app with CORS middleware
- `worker.py` — Celery app configuration
- `tasks.py` — Async task definitions (document processing, 3D generation)
- `api/routes.py` — All API endpoints
- `services/` — Business logic: document parsing, LLM inference (llama-cpp-python), mesh generation (TripoSR), format conversion, PBR materials, Unity HDRP adaptation
- `models/spec_models.py` — Pydantic models for 3D specifications
- `core/config.py` — Settings (Redis URLs, file size limits)

**Frontend** (React 19 + Vite + Tailwind): `Frontend/`
- `App.tsx` — Main orchestrator, pipeline state management
- `components/TextExtractor.tsx` — PDF upload & text extraction UI
- `components/PipelineVisualizer.tsx` — Visual pipeline step flow
- `components/Terminal.tsx` — Real-time log viewer
- `types.ts` — Shared enums (PipelineStep, GenerationMethod) and interfaces

**Pipeline flow**: Upload → FastAPI → Celery task → Document parsing → LLM spec generation → Mesh generation → Format conversion → File storage → Result delivery

**Celery queues**: `document_processing`, `3d_generation` (routed in worker.py)

## Common Commands

```bash
# Full stack (dev with hot reload)
docker-compose up --build

# Production
docker-compose -f docker-compose.prod.yml up --build

# Frontend only
cd Frontend && npm run dev        # Vite dev server on port 3000
cd Frontend && npm run build      # Production build

# Backend only (requires Redis running)
cd Backend && uvicorn app.main:app --reload --port 8000
cd Backend && celery -A app.worker worker --loglevel=info
cd Backend && celery -A app.worker beat --loglevel=info
```

## Ports

| Service  | Dev  | Docker |
|----------|------|--------|
| Frontend | 3000 | 9503   |
| Backend  | 8000 | 9502   |
| Redis    | —    | 9501   |

## Key API Endpoints

- `POST /api/v1/upload` — Document upload (PDF/Email)
- `GET /api/v1/task/{task_id}` — Poll task status
- `GET /api/v1/models` — List generated 3D models
- `GET /health` — Health check

## Integration: Hunyuan3D-2GP

The `Hunyuan3D-2GP/` directory contains a separate 3D generation model (text-to-image, shape generation, texture generation). It has its own API server (`api_server.py`) and Gradio app.

# 3D Generator - Docker Setup

This project uses Docker Compose to orchestrate multiple services:

- **FastAPI Backend** (port 9502) - Main API service
- **React Frontend** (port 9503) - Web dashboard
- **Redis** (port 9501) - Cache and message broker
- **Celery Worker** - Background task processing
- **Celery Beat** - Scheduled task processing

## Quick Start

### Development Mode (with live reloading)
```bash
# Build and start all services with volume mounts for development
docker-compose up --build
```

### Production Mode (fully containerized)
```bash
# Build and start all services with code baked into containers
docker-compose -f docker-compose.prod.yml up --build
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed differences between deployment options.

## Service URLs

- **Frontend**: http://localhost:9503
- **Backend API**: http://localhost:9502
- **API Docs**: http://localhost:9502/docs
- **ReDoc**: http://localhost:9502/redoc
- **Redis**: localhost:9501

## Development Commands

Using npm scripts:
```bash
# Start all services
npm run docker:dev

# Build services
npm run docker:build

# Stop services
npm run docker:down
```

## Project Structure

```
3D-Generator/
├── docker-compose.yml          # Docker Compose configuration
├── Backend/
│   ├── Dockerfile             # Backend Docker configuration
│   ├── requirements.txt       # Python dependencies
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── core/
│   │   │   └── config.py     # Configuration settings
│   │   ├── api/
│   │   │   └── routes.py     # API routes
│   │   ├── worker.py         # Celery worker configuration
│   │   └── tasks.py          # Background tasks
│   └── uploads/              # Uploaded files (mounted volume)
├── Frontend/
│   ├── Dockerfile            # Frontend Docker configuration
│   ├── nginx.conf           # Nginx configuration
│   └── dist/                # Built React app (in container)
└── generated/               # Generated 3D models (mounted volume)
```

## Troubleshooting

1. **Port conflicts**: Make sure ports 3000, 6379, and 8000 are available
2. **Permission issues**: Check file permissions for mounted volumes
3. **Build errors**: Clear Docker cache with `docker system prune -a`

## Environment Variables

The services use the following environment variables (configured in docker-compose.yml):

- `REDIS_URL`: Redis connection string
- `CELERY_BROKER_URL`: Celery message broker URL
- `CELERY_RESULT_BACKEND`: Celery result backend URL
- `VITE_API_URL`: Frontend API URL
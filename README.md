# 3D Generator - Docker Setup

This project uses Docker Compose to orchestrate multiple services:

- **FastAPI Backend** (port 8000) - Main API service
- **React Frontend** (port 3000) - Web dashboard
- **Redis** (port 6379) - Cache and message broker
- **Celery Worker** - Background task processing
- **Celery Beat** - Scheduled task processing

## Quick Start

1. **Build and start all services:**
```bash
docker-compose up --build
```

2. **Start in detached mode:**
```bash
docker-compose up -d
```

3. **View logs:**
```bash
docker-compose logs -f
```

4. **Stop all services:**
```bash
docker-compose down
```

5. **Stop and remove volumes:**
```bash
docker-compose down -v
```

## Service URLs

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Redis**: localhost:6379

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
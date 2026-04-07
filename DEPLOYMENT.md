# Deployment Options

This project supports two deployment configurations:

## Development Deployment (with live reloading)
```bash
# Uses docker-compose.yml with volume mounts for development
docker-compose up -d
```

Features:
- Code volume mounts for live development
- Hot reloading for frontend and backend
- Local file access for debugging
- Development environment variables

## Production Deployment (fully containerized)
```bash
# Uses docker-compose.prod.yml with no volume mounts
docker-compose -f docker-compose.prod.yml up -d
```

Features:
- All code baked into Docker images
- No external volume mounts for code
- Production environment variables
- Fully isolated and reproducible deployment

## Key Differences

| Aspect | Development | Production |
|--------|-------------|------------|
| Code Mounts | Yes (./Backend:/app, ./Frontend:/app) | No |
| Dependencies | Installed in container + local sync | Fully in container |
| Environment | Development | Production |
| Reloading | Hot reload enabled | Static deployment |
| File Access | Direct file system access | Container-only access |

## When to Use Each

**Development**: Use when actively developing and testing code changes
**Production**: Use for staging, demos, or final deployment where consistency is critical

Both configurations use the same underlying Docker images and maintain the same service architecture.
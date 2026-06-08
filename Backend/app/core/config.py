from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "3D Generator API v2"
    
    # Redis Settings
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Celery Settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    GENERATED_DIR: str = "generated"
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["http://localhost:3001", "http://localhost:8001"]

    # Unity MCP Integration
    # "sqlite" for local dev (no extra infra), "redis" for production
    EVENT_BUS_TYPE: str = "sqlite"
    # Shared secret between backend and unity-agent
    UNITY_AGENT_TOKEN: str = "dev-token"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

settings = Settings()
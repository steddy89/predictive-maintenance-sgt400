"""Application configuration using Pydantic Settings."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    ENVIRONMENT: str = "dev"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    ALLOWED_HOSTS: list[str] = ["*"]
    
    # Azure AI Foundry / ML
    AZURE_ML_ENDPOINT_ANOMALY: str = ""
    AZURE_ML_ENDPOINT_FORECAST: str = ""
    AZURE_ML_API_KEY_ANOMALY: str = ""
    AZURE_ML_API_KEY_FORECAST: str = ""
    
    # Azure Key Vault
    AZURE_KEY_VAULT_URL: str = ""
    
    # Microsoft Fabric
    FABRIC_WORKSPACE_ID: str = ""
    FABRIC_LAKEHOUSE_ID: str = ""
    FABRIC_CONNECTION_STRING: str = ""
    
    # Application Insights
    APPINSIGHTS_CONNECTION_STRING: str = ""
    
    # Alerting
    ALERT_WEBHOOK_URL: str = ""
    ANOMALY_THRESHOLD: float = 0.7
    FAILURE_PROBABILITY_THRESHOLD: float = 0.6
    
    # Model paths (local fallback)
    LOCAL_MODEL_DIR: str = "models"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

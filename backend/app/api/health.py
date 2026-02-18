"""Health check endpoint."""

from datetime import datetime

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Application health check.
    Used by Azure App Service health probes and load balancers.
    """
    services = {
        "api": "healthy",
        "fabric": "connected" if settings.FABRIC_CONNECTION_STRING else "not_configured",
        "ml_anomaly": "connected" if settings.AZURE_ML_ENDPOINT_ANOMALY else "local_fallback",
        "ml_forecast": "connected" if settings.AZURE_ML_ENDPOINT_FORECAST else "local_fallback",
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        timestamp=datetime.now(),
        services=services,
    )

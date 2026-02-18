"""
============================================================================
SGT400 Predictive Maintenance - FastAPI Backend
============================================================================
Production REST API for turbine monitoring, anomaly detection, 
failure prediction, and alert management.

Reference:
  - https://learn.microsoft.com/azure/app-service/quickstart-python
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.turbine import router as turbine_router
from app.api.alerts import router as alert_router
from app.api.predictions import router as prediction_router
from app.api.health import router as health_router
from app.core.config import settings
from app.core.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown events."""
    logger.info("Starting SGT400 Predictive Maintenance API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize ML model clients on startup
    from app.services.ml_client import MLModelClient
    app.state.ml_client = MLModelClient()
    await app.state.ml_client.initialize()
    
    yield
    
    # Cleanup
    logger.info("Shutting down API")
    await app.state.ml_client.close()


app = FastAPI(
    title="SGT400 Predictive Maintenance API",
    description=(
        "REST API for Siemens SGT400 Gas Turbine predictive maintenance. "
        "Provides real-time turbine status, anomaly detection, failure prediction, "
        "and alert management."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trusted hosts
if settings.ENVIRONMENT == "prod":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

# Register routers
app.include_router(health_router, prefix="", tags=["Health"])
app.include_router(turbine_router, prefix="/api/v1/turbine", tags=["Turbine"])
app.include_router(prediction_router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(alert_router, prefix="/api/v1/alerts", tags=["Alerts"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.ENVIRONMENT == "dev",
        workers=4,
        log_level="info",
    )

"""Structured logging configuration with Application Insights integration."""

import logging
import sys

from app.core.config import settings


def setup_logging():
    """Configure structured logging."""
    
    log_format = (
        "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    )
    
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Suppress noisy loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)
    
    # Application Insights integration
    if settings.APPINSIGHTS_CONNECTION_STRING:
        try:
            from opencensus.ext.azure.log_exporter import AzureLogHandler
            
            azure_handler = AzureLogHandler(
                connection_string=settings.APPINSIGHTS_CONNECTION_STRING
            )
            logging.getLogger().addHandler(azure_handler)
            logging.getLogger(__name__).info("Application Insights logging enabled")
        except ImportError:
            logging.getLogger(__name__).warning(
                "opencensus not installed. App Insights logging disabled."
            )

"""Alert management endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException

from app.models.schemas import AlertListResponse, Alert, AlertAcknowledge
from app.services.fabric_client import FabricDataService

router = APIRouter()
logger = logging.getLogger(__name__)
fabric_service = FabricDataService()


@router.get("/", response_model=AlertListResponse)
async def get_alerts(
    turbine_id: str = Query(default="SGT400-001"),
    hours: int = Query(default=168, ge=1, le=720),
    severity: str | None = Query(default=None, description="Filter by severity: INFO, WARNING, CRITICAL"),
    active_only: bool = Query(default=False),
):
    """Get alert history for a turbine."""
    try:
        alerts = await fabric_service.get_alert_history(turbine_id, hours, severity)
        
        if active_only:
            alerts = [a for a in alerts if not a.get("resolved", False)]
        
        return AlertListResponse(
            status="success",
            total=len(alerts),
            alerts=[Alert(**a) for a in alerts],
        )
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/acknowledge")
async def acknowledge_alert(body: AlertAcknowledge):
    """Acknowledge an alert."""
    logger.info(f"Alert {body.alert_id} acknowledged by {body.acknowledged_by}")
    return {
        "status": "success",
        "message": f"Alert {body.alert_id} acknowledged",
        "acknowledged_at": datetime.now().isoformat(),
    }


@router.get("/summary")
async def get_alert_summary(
    turbine_id: str = Query(default="SGT400-001"),
):
    """Get a summary of active alerts by severity."""
    alerts = await fabric_service.get_alert_history(turbine_id, hours=168)
    
    active = [a for a in alerts if not a.get("resolved", False)]
    
    summary = {
        "total_active": len(active),
        "critical": sum(1 for a in active if a["severity"] == "CRITICAL"),
        "warning": sum(1 for a in active if a["severity"] == "WARNING"),
        "info": sum(1 for a in active if a["severity"] == "INFO"),
        "last_alert": active[0]["timestamp"] if active else None,
    }
    
    return {"status": "success", "turbine_id": turbine_id, "summary": summary}

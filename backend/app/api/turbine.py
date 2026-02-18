"""Turbine status and sensor data endpoints."""

import logging
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException

from app.models.schemas import TurbineStatusResponse, TrendResponse, TrendPoint
from app.services.fabric_client import FabricDataService

router = APIRouter()
logger = logging.getLogger(__name__)
fabric_service = FabricDataService()


@router.get("/live")
async def get_live_reading(
    turbine_id: str = Query(default="SGT400-001", description="Turbine identifier"),
):
    """
    Return the next live data point from the dataset.

    Each call advances an internal pointer, cycling through the full
    CSV.  The timestamp is set to *now* so the dashboard feels real-time.
    Call this every 2 seconds for the live feed.
    """
    try:
        reading = await fabric_service.get_live_reading(turbine_id)
        return {"status": "success", "data": reading}
    except Exception as e:
        logger.error(f"Error fetching live reading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=TurbineStatusResponse)
async def get_turbine_status(
    turbine_id: str = Query(default="SGT400-001", description="Turbine identifier")
):
    """
    Get the latest turbine status including health score, risk level,
    and key sensor values.
    """
    try:
        status = await fabric_service.get_latest_status(turbine_id)
        return TurbineStatusResponse(
            status="success",
            data=status,
        )
    except Exception as e:
        logger.error(f"Error fetching turbine status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/readings")
async def get_sensor_readings(
    turbine_id: str = Query(default="SGT400-001"),
    hours: int = Query(default=24, ge=1, le=720),
):
    """Get recent sensor readings for a turbine."""
    try:
        readings = await fabric_service.get_recent_readings(turbine_id, hours)
        return {
            "status": "success",
            "turbine_id": turbine_id,
            "period_hours": hours,
            "count": len(readings),
            "data": readings,
        }
    except Exception as e:
        logger.error(f"Error fetching readings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trend/{sensor}", response_model=TrendResponse)
async def get_sensor_trend(
    sensor: str,
    turbine_id: str = Query(default="SGT400-001"),
    hours: int = Query(default=24, ge=1, le=720),
):
    """
    Get trend data for a specific sensor.
    
    Available sensors:
    - temperature_c
    - rpm
    - torque_nm
    - vibration_mm_s
    - power_output_mw
    - fuel_flow_kg_s
    - air_pressure_kpa
    - exhaust_gas_temp_c
    - oil_temp_c
    - health_score
    - anomaly_score
    """
    valid_sensors = [
        "temperature_c", "rpm", "torque_nm",
        "vibration_mm_s", "power_output_mw", "fuel_flow_kg_s",
        "air_pressure_kpa", "exhaust_gas_temp_c", "oil_temp_c",
        "health_score", "anomaly_score",
    ]
    
    if sensor not in valid_sensors:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sensor. Available: {valid_sensors}"
        )
    
    try:
        trend_data = await fabric_service.get_trend_data(turbine_id, sensor, hours)
        return TrendResponse(
            status="success",
            turbine_id=turbine_id,
            sensor=sensor,
            period_hours=hours,
            data=[TrendPoint(**p) for p in trend_data],
        )
    except Exception as e:
        logger.error(f"Error fetching trend data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

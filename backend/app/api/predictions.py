"""Prediction endpoints - anomaly detection, failure prediction, forecasting."""

import logging
from datetime import datetime, timedelta

from fastapi import APIRouter, Query, Request, HTTPException

from app.models.schemas import (
    AnomalyResponse, AnomalyResult,
    FailurePredictionResponse, FailurePrediction,
    ForecastResponse, SensorForecast, ForecastPoint,
    SensorBatch, RiskLevel,
)
from app.services.fabric_client import FabricDataService
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)
fabric_service = FabricDataService()


@router.get("/anomaly", response_model=AnomalyResponse)
async def get_anomaly_score(
    request: Request,
    turbine_id: str = Query(default="SGT400-001"),
):
    """
    Get the latest anomaly detection score for a turbine.
    Uses the deployed Isolation Forest model or local fallback.
    """
    try:
        readings = await fabric_service.get_recent_readings(turbine_id, hours=1)
        
        if not readings:
            raise HTTPException(status_code=404, detail="No recent readings found")
        
        ml_client = request.app.state.ml_client
        result = await ml_client.get_anomaly_score(readings[-5:])
        
        latest = result["predictions"][-1] if result.get("predictions") else {
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "raw_score": 0.0,
            "contributing_sensors": [],
        }
        
        return AnomalyResponse(
            status="success",
            turbine_id=turbine_id,
            timestamp=datetime.now(),
            result=AnomalyResult(**latest),
            threshold=settings.ANOMALY_THRESHOLD,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/failure", response_model=FailurePredictionResponse)
async def get_failure_prediction(
    turbine_id: str = Query(default="SGT400-001"),
):
    """
    Get failure prediction and Remaining Useful Life (RUL) estimate.
    """
    try:
        readings = await fabric_service.get_recent_readings(turbine_id, hours=168)
        
        if len(readings) < 12:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for failure prediction. Need at least 1 hour."
            )
        
        # Simple RUL estimation based on sensor trends
        import numpy as np
        
        vibrations = [r.get("vibration_mm_s", 2.5) for r in readings]
        temps = [r.get("exhaust_gas_temp_c", 545) for r in readings]
        
        vib_trend = np.polyfit(range(len(vibrations)), vibrations, 1)[0]
        temp_trend = np.polyfit(range(len(temps)), temps, 1)[0]
        
        latest_vib = vibrations[-1]
        latest_temp = temps[-1]
        
        # Degradation index
        vib_dev = max(0, (latest_vib - 2.5) / 2.5)
        temp_dev = max(0, (latest_temp - 545) / 50)
        degradation = min(1.0, (vib_dev * 0.5 + temp_dev * 0.5))
        
        # Failure probability
        failure_prob = min(1.0, float(np.exp(degradation * 3 - 2)))
        
        # RUL estimate
        if degradation < 0.7 and vib_trend >= 0:
            remaining = (0.7 - degradation) / max(vib_trend * 288, 0.001)
            rul_days = min(365, max(0, remaining))
        elif degradation >= 0.7:
            rul_days = max(0, (1.0 - degradation) * 10)
        else:
            rul_days = 180
        
        # Determine risk level
        if failure_prob > 0.8 or rul_days < 3:
            risk = RiskLevel.CRITICAL
        elif failure_prob > 0.6 or rul_days < 7:
            risk = RiskLevel.HIGH
        elif failure_prob > 0.3 or rul_days < 30:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW
        
        # Recommendation
        if rul_days <= 3:
            recommendation = "IMMEDIATE: Schedule emergency maintenance within 24 hours"
        elif rul_days <= 7:
            recommendation = "URGENT: Plan maintenance within the next 3-5 days"
        elif rul_days <= 14:
            recommendation = "WARNING: Schedule maintenance within 2 weeks"
        elif rul_days <= 30:
            recommendation = "ADVISORY: Monitor closely, plan maintenance in next window"
        else:
            recommendation = "NORMAL: Continue standard monitoring schedule"
        
        return FailurePredictionResponse(
            status="success",
            turbine_id=turbine_id,
            timestamp=datetime.now(),
            prediction=FailurePrediction(
                failure_probability=round(failure_prob, 4),
                rul_days=round(rul_days, 1),
                degradation_index=round(degradation, 4),
                degradation_rate=round(float(vib_trend), 6),
                confidence="MEDIUM",
                recommendation=recommendation,
                risk_level=risk,
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failure prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast", response_model=ForecastResponse)
async def get_sensor_forecast(
    request: Request,
    turbine_id: str = Query(default="SGT400-001"),
    horizon_minutes: int = Query(default=60, ge=5, le=360),
):
    """
    Get sensor value forecasts for the specified horizon.
    """
    try:
        readings = await fabric_service.get_recent_readings(turbine_id, hours=4)
        
        if len(readings) < 12:
            raise HTTPException(
                status_code=400,
                detail="Insufficient data for forecasting."
            )
        
        ml_client = request.app.state.ml_client
        result = await ml_client.get_forecast(readings)
        
        now = datetime.now()
        horizon_steps = horizon_minutes // 5
        
        sensors = []
        forecasts_data = result.get("forecasts", {})
        
        for sensor_name, values in forecasts_data.items():
            forecast_points = []
            for i, val in enumerate(values[:horizon_steps]):
                ts = now + timedelta(minutes=5 * (i + 1))
                forecast_points.append(ForecastPoint(
                    timestamp=ts,
                    value=round(float(val), 4),
                    lower_bound=round(float(val * 0.97), 4),
                    upper_bound=round(float(val * 1.03), 4),
                ))
            
            sensors.append(SensorForecast(
                sensor=sensor_name,
                forecast=forecast_points,
            ))
        
        return ForecastResponse(
            status="success",
            turbine_id=turbine_id,
            forecast_horizon_minutes=horizon_minutes,
            sensors=sensors,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

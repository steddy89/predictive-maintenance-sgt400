"""Pydantic models for API request and response schemas."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---- Enums ----
class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---- Sensor Data ----
class SensorReading(BaseModel):
    timestamp: datetime
    turbine_id: str = "SGT400-001"
    temperature_c: float = Field(..., ge=700, le=1100, description="Compressor/Turbine Temperature")
    rpm: float = Field(..., ge=13000, le=17000)
    torque_nm: float = Field(..., ge=2500, le=4500)
    vibration_mm_s: float = Field(..., ge=0, le=5)
    power_output_mw: float = Field(..., ge=60, le=140)
    fuel_flow_kg_s: float = Field(..., ge=1.5, le=4.0)
    air_pressure_kpa: float = Field(..., ge=90, le=220)
    exhaust_gas_temp_c: float = Field(..., ge=400, le=600)
    oil_temp_c: float = Field(..., ge=80, le=160)
    fault: int = Field(0, ge=0, le=1)


class SensorBatch(BaseModel):
    readings: list[SensorReading]


# ---- Turbine Status ----
class TurbineStatus(BaseModel):
    turbine_id: str
    timestamp: datetime
    health_score: float = Field(..., ge=0, le=100)
    risk_level: RiskLevel
    temperature_c: float
    rpm: float
    torque_nm: float
    vibration_mm_s: float
    power_output_mw: float
    fuel_flow_kg_s: float
    air_pressure_kpa: float
    exhaust_gas_temp_c: float
    oil_temp_c: float
    fault: int = 0
    active_alerts: int = 0
    last_maintenance: datetime | None = None


class TurbineStatusResponse(BaseModel):
    status: str = "success"
    data: TurbineStatus


# ---- Anomaly Detection ----
class AnomalyResult(BaseModel):
    is_anomaly: bool
    anomaly_score: float = Field(..., ge=0, le=1)
    raw_score: float
    contributing_sensors: list[str] = []


class AnomalyResponse(BaseModel):
    status: str = "success"
    turbine_id: str
    timestamp: datetime
    result: AnomalyResult
    threshold: float


# ---- Failure Prediction ----
class FailurePrediction(BaseModel):
    failure_probability: float = Field(..., ge=0, le=1)
    rul_days: float | None = None
    degradation_index: float
    degradation_rate: float | None = None
    confidence: str
    recommendation: str
    risk_level: RiskLevel


class FailurePredictionResponse(BaseModel):
    status: str = "success"
    turbine_id: str
    timestamp: datetime
    prediction: FailurePrediction


# ---- Forecasting ----
class ForecastPoint(BaseModel):
    timestamp: datetime
    value: float
    lower_bound: float | None = None
    upper_bound: float | None = None


class SensorForecast(BaseModel):
    sensor: str
    forecast: list[ForecastPoint]


class ForecastResponse(BaseModel):
    status: str = "success"
    turbine_id: str
    forecast_horizon_minutes: int
    sensors: list[SensorForecast]


# ---- Alerts ----
class Alert(BaseModel):
    id: str
    timestamp: datetime
    turbine_id: str
    alarm_code: str
    description: str
    severity: AlertSeverity
    sensor_value: float | None = None
    threshold: float | None = None
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: datetime | None = None


class AlertListResponse(BaseModel):
    status: str = "success"
    total: int
    alerts: list[Alert]


class AlertAcknowledge(BaseModel):
    alert_id: str
    acknowledged_by: str


# ---- Health Check ----
class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    timestamp: datetime
    services: dict[str, str]


# ---- Trend Data ----
class TrendPoint(BaseModel):
    timestamp: datetime
    value: float


class TrendResponse(BaseModel):
    status: str = "success"
    turbine_id: str
    sensor: str
    period_hours: int
    data: list[TrendPoint]

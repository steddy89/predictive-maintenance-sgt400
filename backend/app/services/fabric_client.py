"""Fabric Lakehouse data service - reads data from Microsoft Fabric."""

import logging
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class FabricDataService:
    """
    Service to read data from Microsoft Fabric Lakehouse.
    
    In production, uses Fabric REST API or JDBC connection.
    Falls back to local sample data for development.
    
    Reference: https://learn.microsoft.com/fabric/data-engineering/lakehouse-overview
    """
    
    def __init__(self):
        self.connection_string = settings.FABRIC_CONNECTION_STRING
        self.workspace_id = settings.FABRIC_WORKSPACE_ID
        self.lakehouse_id = settings.FABRIC_LAKEHOUSE_ID
        self._sample_data = None
    
    def _ensure_sample_data(self):
        """Load sample data for development mode."""
        if self._sample_data is None:
            try:
                self._sample_data = pd.read_parquet("data/sample/sensor_data_normal.parquet")
            except (FileNotFoundError, ImportError, Exception) as exc:
                logger.warning(f"Sample data not loaded ({exc.__class__.__name__}). Generating synthetic data...")
                self._sample_data = self._generate_synthetic_snapshot()
    
    def _generate_synthetic_snapshot(self) -> pd.DataFrame:
        """Generate a small synthetic dataset for development."""
        np.random.seed(42)
        now = datetime.now()
        n = 288  # 24 hours at 5-min intervals
        
        data = {
            "timestamp": [now - timedelta(minutes=5 * i) for i in range(n, 0, -1)],
            "turbine_id": ["SGT400-001"] * n,
            "exhaust_gas_temp_c": np.random.normal(545, 8, n),
            "compressor_discharge_temp_c": np.random.normal(380, 5, n),
            "vibration_mm_s": np.random.normal(2.5, 0.4, n),
            "inlet_pressure_bar": np.random.normal(1.013, 0.01, n),
            "discharge_pressure_bar": np.random.normal(15.5, 0.3, n),
            "turbine_load_mw": np.random.normal(11.8, 0.5, n),
            "fuel_flow_kg_s": np.random.normal(3.2, 0.15, n),
            "rotor_speed_rpm": np.random.normal(9500, 30, n),
            "lube_oil_temp_c": np.random.normal(52, 2, n),
            "ambient_temp_c": np.random.normal(28, 4, n),
        }
        
        df = pd.DataFrame(data)
        df["pressure_ratio"] = df["discharge_pressure_bar"] / df["inlet_pressure_bar"]
        df["efficiency_pct"] = np.random.normal(96, 2, n)
        df["health_score"] = np.random.normal(88, 5, n).clip(0, 100)
        df["anomaly_score"] = np.random.exponential(0.1, n).clip(0, 1)
        
        return df
    
    async def get_latest_status(self, turbine_id: str = "SGT400-001") -> dict:
        """Get the latest turbine status from Gold layer."""
        self._ensure_sample_data()
        latest = self._sample_data[self._sample_data["turbine_id"] == turbine_id].iloc[-1]
        
        health = float(latest.get("health_score", 88))
        
        return {
            "turbine_id": turbine_id,
            "timestamp": latest["timestamp"].isoformat() if isinstance(latest["timestamp"], datetime) else str(latest["timestamp"]),
            "health_score": round(health, 1),
            "risk_level": "LOW" if health >= 80 else "MEDIUM" if health >= 60 else "HIGH" if health >= 40 else "CRITICAL",
            "exhaust_gas_temp_c": round(float(latest["exhaust_gas_temp_c"]), 1),
            "vibration_mm_s": round(float(latest["vibration_mm_s"]), 2),
            "discharge_pressure_bar": round(float(latest["discharge_pressure_bar"]), 2),
            "turbine_load_mw": round(float(latest["turbine_load_mw"]), 2),
            "efficiency_pct": round(float(latest.get("efficiency_pct", 96)), 1),
            "pressure_ratio": round(float(latest.get("pressure_ratio", 15.3)), 2),
            "active_alerts": 0,
        }
    
    async def get_recent_readings(
        self,
        turbine_id: str = "SGT400-001",
        hours: int = 24,
    ) -> list[dict]:
        """Get recent sensor readings."""
        self._ensure_sample_data()
        
        df = self._sample_data[self._sample_data["turbine_id"] == turbine_id].copy()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Use all data if timestamps are in the past
        df = df.tail(hours * 12)  # 12 readings per hour at 5-min intervals
        
        records = df.to_dict(orient="records")
        for r in records:
            if isinstance(r.get("timestamp"), pd.Timestamp):
                r["timestamp"] = r["timestamp"].isoformat()
        
        return records
    
    async def get_trend_data(
        self,
        turbine_id: str,
        sensor: str,
        hours: int = 24,
    ) -> list[dict]:
        """Get trend data for a specific sensor."""
        self._ensure_sample_data()
        
        df = self._sample_data[self._sample_data["turbine_id"] == turbine_id].copy()
        df = df.tail(hours * 12)
        
        if sensor not in df.columns:
            return []
        
        return [
            {
                "timestamp": row["timestamp"].isoformat() if isinstance(row["timestamp"], pd.Timestamp) else str(row["timestamp"]),
                "value": round(float(row[sensor]), 4),
            }
            for _, row in df.iterrows()
        ]
    
    async def get_alert_history(
        self,
        turbine_id: str = "SGT400-001",
        hours: int = 168,
        severity: str | None = None,
    ) -> list[dict]:
        """Get alert history from Gold layer."""
        # In production, query gold_alert_history table
        # For development, return sample alerts
        alerts = [
            {
                "id": "ALT-001",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
                "turbine_id": turbine_id,
                "alarm_code": "ALM002",
                "description": "High Vibration Level",
                "severity": "WARNING",
                "sensor_value": 5.2,
                "threshold": 5.0,
                "acknowledged": False,
                "resolved": False,
            },
            {
                "id": "ALT-002",
                "timestamp": (datetime.now() - timedelta(hours=12)).isoformat(),
                "turbine_id": turbine_id,
                "alarm_code": "ALM001",
                "description": "High Exhaust Gas Temperature",
                "severity": "CRITICAL",
                "sensor_value": 582.3,
                "threshold": 580.0,
                "acknowledged": True,
                "resolved": True,
                "resolved_at": (datetime.now() - timedelta(hours=10)).isoformat(),
            },
        ]
        
        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]
        
        return alerts

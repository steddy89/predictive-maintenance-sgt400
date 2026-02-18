"""
Fabric Lakehouse data service — reads real gas turbine fault-detection data.

Dataset: gas_turbine_fault_detection.csv  (1 386 rows × 10 columns)
Columns: Temperature (°C), RPM, Torque (Nm), Vibrations (mm/s),
         Power Output (MW), Fuel Flow Rate (kg/s), Air Pressure (kPa),
         Exhaust Gas Temperature (°C), Oil Temperature (°C), Fault
"""

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

# Column rename mapping — CSV headers → internal snake_case names
_COL_MAP = {
    "Temperature (°C)": "temperature_c",
    "RPM": "rpm",
    "Torque (Nm)": "torque_nm",
    "Vibrations (mm/s)": "vibration_mm_s",
    "Power Output (MW)": "power_output_mw",
    "Fuel Flow Rate (kg/s)": "fuel_flow_kg_s",
    "Air Pressure (kPa)": "air_pressure_kpa",
    "Exhaust Gas Temperature (°C)": "exhaust_gas_temp_c",
    "Oil Temperature (°C)": "oil_temp_c",
    "Fault": "fault",
}

# Sensor thresholds used for health score & alerting
SENSOR_THRESHOLDS = {
    "temperature_c":       {"warn": 970,  "crit": 1050,  "mean": 901, "std": 49},
    "rpm":                 {"warn": 15800, "crit": 16200, "mean": 15022, "std": 490},
    "torque_nm":           {"warn": 3800, "crit": 4100,   "mean": 3494, "std": 204},
    "vibration_mm_s":      {"warn": 2.8,  "crit": 3.2,    "mean": 1.98, "std": 0.49},
    "power_output_mw":     {"warn_low": 80, "crit_low": 72, "mean": 99.5, "std": 10.3},
    "fuel_flow_kg_s":      {"warn": 3.0,  "crit": 3.3,    "mean": 2.51, "std": 0.32},
    "air_pressure_kpa":    {"warn": 185,  "crit": 200,    "mean": 150, "std": 19.4},
    "exhaust_gas_temp_c":  {"warn": 540,  "crit": 570,    "mean": 499, "std": 28.8},
    "oil_temp_c":          {"warn": 135,  "crit": 145,    "mean": 120, "std": 10.0},
}


class FabricDataService:
    """Service to read real turbine sensor data from CSV (or Fabric in prod)."""

    def __init__(self):
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    def _ensure_data(self):
        """Load the CSV dataset, add synthetic timestamps, cache in memory."""
        if self._df is not None:
            return

        csv_candidates = [
            Path(__file__).resolve().parents[2] / "data" / "gas_turbine_fault_detection.csv",
            Path("data") / "gas_turbine_fault_detection.csv",
        ]

        for path in csv_candidates:
            if path.exists():
                logger.info(f"Loading dataset from {path}")
                df = pd.read_csv(str(path))
                break
        else:
            raise FileNotFoundError(
                "gas_turbine_fault_detection.csv not found in any expected location"
            )

        # Rename to snake_case
        df.rename(columns=_COL_MAP, inplace=True)

        # Assign synthetic timestamps (most-recent = now, 1-min intervals backwards)
        now = datetime.now()
        n = len(df)
        df["timestamp"] = pd.date_range(end=now, periods=n, freq="min")
        df["turbine_id"] = "SGT400-001"

        # Compute derived columns — vectorised (fast)
        df["health_score"] = self._compute_health_vectorised(df)
        df["anomaly_score"] = self._compute_anomaly_score_vectorised(df)

        self._df = df
        logger.info(f"Dataset loaded: {n} rows, {len(df.columns)} columns, "
                     f"faults={int(df['fault'].sum())}")

    @staticmethod
    def _compute_health_vectorised(df: pd.DataFrame) -> pd.Series:
        """Vectorised 0-100 health score from sensor z-scores."""
        penalties = pd.Series(0.0, index=df.index)
        for sensor, thresh in SENSOR_THRESHOLDS.items():
            if sensor not in df.columns:
                continue
            z = ((df[sensor] - thresh["mean"]) / thresh["std"]).abs()
            penalties += np.where(z > 3, 25, np.where(z > 2, 10, np.where(z > 1.5, 3, 0)))
        penalties += np.where(df.get("fault", 0) == 1, 30, 0)
        return (100 - penalties).clip(0, 100)

    @staticmethod
    def _compute_anomaly_score_vectorised(df: pd.DataFrame) -> pd.Series:
        """Vectorised 0-1 anomaly score from mean sensor z-scores."""
        z_sum = pd.Series(0.0, index=df.index)
        count = 0
        for sensor, thresh in SENSOR_THRESHOLDS.items():
            if sensor not in df.columns:
                continue
            z_sum += ((df[sensor] - thresh["mean"]) / thresh["std"]).abs()
            count += 1
        avg_z = z_sum / max(count, 1)
        return (avg_z / 5.0).clip(0, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get_latest_status(self, turbine_id: str = "SGT400-001") -> dict:
        self._ensure_data()
        latest = self._df.iloc[-1]
        health = float(latest["health_score"])
        fault_count = int(self._df.tail(288)["fault"].sum())

        return {
            "turbine_id": turbine_id,
            "timestamp": latest["timestamp"].isoformat(),
            "health_score": round(health, 1),
            "risk_level": (
                "LOW" if health >= 80 else
                "MEDIUM" if health >= 60 else
                "HIGH" if health >= 40 else "CRITICAL"
            ),
            "temperature_c": round(float(latest["temperature_c"]), 1),
            "rpm": round(float(latest["rpm"]), 0),
            "torque_nm": round(float(latest["torque_nm"]), 1),
            "vibration_mm_s": round(float(latest["vibration_mm_s"]), 3),
            "power_output_mw": round(float(latest["power_output_mw"]), 2),
            "fuel_flow_kg_s": round(float(latest["fuel_flow_kg_s"]), 3),
            "air_pressure_kpa": round(float(latest["air_pressure_kpa"]), 1),
            "exhaust_gas_temp_c": round(float(latest["exhaust_gas_temp_c"]), 1),
            "oil_temp_c": round(float(latest["oil_temp_c"]), 1),
            "fault": int(latest["fault"]),
            "active_alerts": fault_count,
        }

    async def get_recent_readings(
        self,
        turbine_id: str = "SGT400-001",
        hours: int = 24,
    ) -> list[dict]:
        self._ensure_data()
        n_rows = min(len(self._df), hours * 60)  # 1-min intervals
        df = self._df.tail(n_rows).copy()
        records = df.to_dict(orient="records")
        for r in records:
            if isinstance(r.get("timestamp"), (datetime, pd.Timestamp)):
                r["timestamp"] = r["timestamp"].isoformat()
        return records

    async def get_trend_data(
        self,
        turbine_id: str,
        sensor: str,
        hours: int = 24,
    ) -> list[dict]:
        self._ensure_data()
        n_rows = min(len(self._df), hours * 60)  # 1-min intervals
        df = self._df.tail(n_rows)

        if sensor not in df.columns:
            return []

        return [
            {
                "timestamp": (
                    row["timestamp"].isoformat()
                    if isinstance(row["timestamp"], (datetime, pd.Timestamp))
                    else str(row["timestamp"])
                ),
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
        """Generate alerts from actual fault records in the dataset."""
        self._ensure_data()
        n_rows = min(len(self._df), hours * 60)  # 1-min intervals
        faulty = self._df.tail(n_rows)
        faulty = faulty[faulty["fault"] == 1].tail(20)

        alerts = []
        for idx, (_, row) in enumerate(faulty.iterrows()):
            # Determine which sensor is most anomalous
            worst_sensor = ""
            worst_z = 0
            for sensor, thresh in SENSOR_THRESHOLDS.items():
                val = row.get(sensor)
                if val is None:
                    continue
                z = abs(val - thresh["mean"]) / thresh["std"] if thresh["std"] else 0
                if z > worst_z:
                    worst_z = z
                    worst_sensor = sensor

            sev = "CRITICAL" if worst_z > 3 else "WARNING"
            ts = row["timestamp"]

            alerts.append({
                "id": f"ALT-{idx + 1:03d}",
                "timestamp": ts.isoformat() if isinstance(ts, (datetime, pd.Timestamp)) else str(ts),
                "turbine_id": turbine_id,
                "alarm_code": f"ALM-{worst_sensor[:6].upper()}",
                "description": f"Fault detected — {worst_sensor} deviation (z={worst_z:.1f}σ)",
                "severity": sev,
                "sensor_value": round(float(row.get(worst_sensor, 0)), 2),
                "threshold": round(SENSOR_THRESHOLDS.get(worst_sensor, {}).get("mean", 0), 2),
                "acknowledged": idx > 5,
                "resolved": idx > 10,
                "resolved_at": None,
            })

        if severity:
            alerts = [a for a in alerts if a["severity"] == severity]

        return alerts

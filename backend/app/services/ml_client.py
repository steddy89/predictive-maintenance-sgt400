"""ML Model Client - Connects to Azure AI Foundry endpoints for inference."""

import json
import logging
from datetime import datetime

import httpx
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class MLModelClient:
    """Client for Azure AI Foundry managed online endpoints."""
    
    def __init__(self):
        self.http_client: httpx.AsyncClient | None = None
        self.anomaly_endpoint = settings.AZURE_ML_ENDPOINT_ANOMALY
        self.forecast_endpoint = settings.AZURE_ML_ENDPOINT_FORECAST
        self.anomaly_key = settings.AZURE_ML_API_KEY_ANOMALY
        self.forecast_key = settings.AZURE_ML_API_KEY_FORECAST
    
    async def initialize(self):
        """Initialize async HTTP client."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        logger.info("ML Model Client initialized")
    
    async def close(self):
        """Close HTTP client."""
        if self.http_client:
            await self.http_client.aclose()
    
    async def get_anomaly_score(self, sensor_data: list[dict]) -> dict:
        """
        Call anomaly detection endpoint.
        
        Args:
            sensor_data: List of sensor reading dictionaries
        
        Returns:
            Anomaly detection result
        """
        if self.anomaly_endpoint and self.anomaly_key:
            return await self._call_endpoint(
                self.anomaly_endpoint,
                self.anomaly_key,
                {"data": sensor_data},
            )
        else:
            # Local fallback using statistical method
            return self._local_anomaly_detection(sensor_data)
    
    async def get_forecast(self, sensor_data: list[dict]) -> dict:
        """
        Call forecasting endpoint.
        
        Args:
            sensor_data: List of recent sensor readings (sequence)
        
        Returns:
            Forecast result
        """
        if self.forecast_endpoint and self.forecast_key:
            return await self._call_endpoint(
                self.forecast_endpoint,
                self.forecast_key,
                {"data": sensor_data},
            )
        else:
            return self._local_forecast(sensor_data)
    
    async def _call_endpoint(self, endpoint: str, api_key: str, payload: dict) -> dict:
        """Call an Azure ML managed online endpoint."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        
        try:
            response = await self.http_client.post(
                endpoint,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"ML endpoint error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"ML endpoint connection error: {str(e)}")
            raise
    
    def _local_anomaly_detection(self, sensor_data: list[dict]) -> dict:
        """
        Local statistical anomaly detection fallback.
        Uses z-score method when Azure ML endpoint is not configured.
        """
        baselines = {
            "exhaust_gas_temp_c": (545, 8),
            "vibration_mm_s": (2.5, 0.4),
            "discharge_pressure_bar": (15.5, 0.3),
            "turbine_load_mw": (11.8, 0.5),
            "fuel_flow_kg_s": (3.2, 0.15),
            "rotor_speed_rpm": (9500, 30),
            "lube_oil_temp_c": (52, 2),
        }
        
        results = []
        for reading in sensor_data:
            z_scores = []
            contributors = []
            
            for sensor, (mean, std) in baselines.items():
                if sensor in reading and std > 0:
                    z = abs(reading[sensor] - mean) / std
                    z_scores.append(z)
                    if z > 2.5:
                        contributors.append(sensor)
            
            avg_z = np.mean(z_scores) if z_scores else 0
            anomaly_score = min(1.0, avg_z / 5.0)
            
            results.append({
                "is_anomaly": anomaly_score > settings.ANOMALY_THRESHOLD,
                "anomaly_score": round(float(anomaly_score), 4),
                "raw_score": round(float(-avg_z), 4),
                "contributing_sensors": contributors,
            })
        
        return {
            "predictions": results,
            "summary": {
                "total_records": len(results),
                "anomalies_detected": sum(1 for r in results if r["is_anomaly"]),
                "max_anomaly_score": max(r["anomaly_score"] for r in results) if results else 0,
                "mean_anomaly_score": np.mean([r["anomaly_score"] for r in results]) if results else 0,
            },
        }
    
    def _local_forecast(self, sensor_data: list[dict]) -> dict:
        """Local linear extrapolation forecast fallback."""
        if len(sensor_data) < 2:
            return {"error": "Need at least 2 data points for forecasting"}
        
        sensors = ["exhaust_gas_temp_c", "vibration_mm_s", "discharge_pressure_bar",
                    "turbine_load_mw", "fuel_flow_kg_s"]
        
        forecasts = {}
        for sensor in sensors:
            values = [r.get(sensor, 0) for r in sensor_data[-12:]]
            if len(values) >= 2:
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                
                future_x = np.arange(len(values), len(values) + 12)
                predictions = np.polyval(coeffs, future_x)
                forecasts[sensor] = predictions.tolist()
        
        return {"forecasts": forecasts}

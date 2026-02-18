"""
============================================================================
Microsoft Fabric Data Agent — SGT400 Predictive Maintenance
============================================================================

Implements a Data Agent that connects to Microsoft Fabric Lakehouse
(or local CSV fallback) using agentic tool-calling patterns.

Architecture (Azure AI Foundry pattern):
  ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
  │  Orchestrator│────▶│  Fabric Data    │────▶│  Fabric Lakehouse│
  │    Agent     │◀────│    Agent        │◀────│  (or local CSV)  │
  └──────────────┘     └─────────────────┘     └──────────────────┘

The Data Agent exposes "tools" that the orchestrator can invoke:
  • query_sensor_data    — SQL-like query on the sensor telemetry
  • compute_statistics   — summary stats for any window
  • detect_anomaly_window— anomaly scan over a time range
  • get_fault_analysis   — fault pattern analysis
  • get_correlation      — sensor-sensor cross-correlation

Reference:
  https://learn.microsoft.com/fabric/data-science/data-agent
  https://learn.microsoft.com/azure/ai-services/agents/overview
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from app.services.fabric_client import FabricDataService, SENSOR_THRESHOLDS

logger = logging.getLogger(__name__)


class FabricDataAgent:
    """
    Microsoft Fabric Data Agent for turbine telemetry analytics.

    Follows the Azure AI Foundry agent tool-calling contract:
    each public method is a 'tool' the orchestrator can invoke.
    """

    AGENT_NAME = "FabricDataAgent"
    AGENT_DESCRIPTION = (
        "Connects to the Microsoft Fabric Lakehouse to query SGT-400 gas "
        "turbine sensor telemetry.  Can run statistical analytics, anomaly "
        "scans, fault-pattern analysis, and cross-correlation studies on "
        "the live data stream."
    )

    # Tool definitions following Azure AI Foundry function-calling schema
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "query_sensor_data",
                "description": (
                    "Query the Fabric Lakehouse for sensor telemetry rows. "
                    "Supports filtering by sensor name, time window, and fault status."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor": {
                            "type": "string",
                            "description": "Sensor column name (e.g. temperature_c, rpm, vibration_mm_s)",
                        },
                        "last_n_rows": {
                            "type": "integer",
                            "description": "Number of most-recent rows to return (default 20)",
                        },
                        "fault_only": {
                            "type": "boolean",
                            "description": "If true, return only rows where fault==1",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_statistics",
                "description": (
                    "Compute summary statistics (mean, std, min, max, percentiles) "
                    "for one or all sensors over a specified window."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor": {
                            "type": "string",
                            "description": "Sensor name, or 'all' for every sensor",
                        },
                        "last_n_rows": {
                            "type": "integer",
                            "description": "Number of rows to include (default: all)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "detect_anomaly_window",
                "description": (
                    "Run statistical anomaly detection over a window of data. "
                    "Returns rows whose z-score exceeds the threshold."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor": {
                            "type": "string",
                            "description": "Sensor to check",
                        },
                        "z_threshold": {
                            "type": "number",
                            "description": "Z-score threshold (default 2.5)",
                        },
                        "last_n_rows": {
                            "type": "integer",
                            "description": "Window size (default 100)",
                        },
                    },
                    "required": ["sensor"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_fault_analysis",
                "description": (
                    "Analyse fault patterns in the dataset: fault rate, "
                    "which sensors deviate most during faults, fault clusters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "last_n_rows": {
                            "type": "integer",
                            "description": "Analysis window (default: all rows)",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_correlation",
                "description": (
                    "Compute Pearson correlation between two sensors or "
                    "between a sensor and the fault column."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sensor_a": {
                            "type": "string",
                            "description": "First sensor",
                        },
                        "sensor_b": {
                            "type": "string",
                            "description": "Second sensor (or 'fault')",
                        },
                    },
                    "required": ["sensor_a", "sensor_b"],
                },
            },
        },
    ]

    def __init__(self, fabric_service: FabricDataService | None = None):
        self._fabric = fabric_service or FabricDataService()
        self._fabric._ensure_data()
        logger.info(f"[{self.AGENT_NAME}] Initialised with {len(self._fabric._df)} rows")

    # ------------------------------------------------------------------
    # Tool dispatcher (Azure AI Foundry pattern)
    # ------------------------------------------------------------------
    async def invoke_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict:
        """Route a tool call from the orchestrator to the correct method."""
        dispatch = {
            "query_sensor_data": self.query_sensor_data,
            "compute_statistics": self.compute_statistics,
            "detect_anomaly_window": self.detect_anomaly_window,
            "get_fault_analysis": self.get_fault_analysis,
            "get_correlation": self.get_correlation,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            result = await handler(**arguments)
            return {"tool": tool_name, "result": result}
        except Exception as e:
            logger.error(f"[{self.AGENT_NAME}] Tool {tool_name} failed: {e}")
            return {"tool": tool_name, "error": str(e)}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------
    async def query_sensor_data(
        self,
        sensor: str | None = None,
        last_n_rows: int = 20,
        fault_only: bool = False,
    ) -> dict:
        """Query the Fabric Lakehouse for sensor telemetry rows."""
        df = self._fabric._df.copy()
        if fault_only:
            df = df[df["fault"] == 1]

        df = df.tail(last_n_rows)

        sensors = [sensor] if sensor and sensor in df.columns else [
            "temperature_c", "rpm", "torque_nm", "vibration_mm_s",
            "power_output_mw", "fuel_flow_kg_s", "air_pressure_kpa",
            "exhaust_gas_temp_c", "oil_temp_c",
        ]

        cols = ["timestamp", "fault", "health_score", "anomaly_score"] + [
            s for s in sensors if s in df.columns
        ]
        subset = df[cols].copy()
        subset["timestamp"] = subset["timestamp"].astype(str)
        records = subset.to_dict(orient="records")

        return {
            "rows_returned": len(records),
            "total_available": len(self._fabric._df),
            "filters": {"sensor": sensor, "fault_only": fault_only},
            "data": records,
        }

    async def compute_statistics(
        self,
        sensor: str | None = None,
        last_n_rows: int | None = None,
    ) -> dict:
        """Compute summary statistics for sensors."""
        df = self._fabric._df
        if last_n_rows:
            df = df.tail(last_n_rows)

        sensors = (
            [sensor] if sensor and sensor != "all" and sensor in df.columns
            else [
                "temperature_c", "rpm", "torque_nm", "vibration_mm_s",
                "power_output_mw", "fuel_flow_kg_s", "air_pressure_kpa",
                "exhaust_gas_temp_c", "oil_temp_c", "health_score", "anomaly_score",
            ]
        )

        stats = {}
        for s in sensors:
            if s not in df.columns:
                continue
            col = df[s].astype(float)
            stats[s] = {
                "mean": round(float(col.mean()), 4),
                "std": round(float(col.std()), 4),
                "min": round(float(col.min()), 4),
                "max": round(float(col.max()), 4),
                "p25": round(float(col.quantile(0.25)), 4),
                "p50": round(float(col.quantile(0.50)), 4),
                "p75": round(float(col.quantile(0.75)), 4),
                "count": int(col.count()),
            }

        return {
            "window_rows": len(df),
            "statistics": stats,
        }

    async def detect_anomaly_window(
        self,
        sensor: str,
        z_threshold: float = 2.5,
        last_n_rows: int = 100,
    ) -> dict:
        """Detect anomalies in a sensor using z-score method."""
        if sensor not in self._fabric._df.columns:
            return {"error": f"Unknown sensor: {sensor}"}

        thresh = SENSOR_THRESHOLDS.get(sensor, {})
        mean = thresh.get("mean", 0)
        std = thresh.get("std", 1)

        df = self._fabric._df.tail(last_n_rows).copy()
        df["z_score"] = ((df[sensor] - mean) / std).abs()
        anomalies = df[df["z_score"] > z_threshold]

        anomaly_records = []
        for _, row in anomalies.iterrows():
            anomaly_records.append({
                "timestamp": str(row["timestamp"]),
                "value": round(float(row[sensor]), 4),
                "z_score": round(float(row["z_score"]), 4),
                "fault": int(row["fault"]),
                "health_score": round(float(row["health_score"]), 2),
            })

        return {
            "sensor": sensor,
            "z_threshold": z_threshold,
            "window_size": last_n_rows,
            "total_anomalies": len(anomaly_records),
            "anomaly_rate": round(len(anomaly_records) / max(last_n_rows, 1), 4),
            "baseline": {"mean": mean, "std": std},
            "anomalies": anomaly_records[:20],  # cap output
        }

    async def get_fault_analysis(
        self,
        last_n_rows: int | None = None,
    ) -> dict:
        """Analyse fault patterns in the dataset."""
        df = self._fabric._df if not last_n_rows else self._fabric._df.tail(last_n_rows)

        total = len(df)
        faults = df[df["fault"] == 1]
        normals = df[df["fault"] == 0]
        fault_count = len(faults)

        # Which sensors deviate most during faults?
        sensor_deviation = {}
        sensors = [
            "temperature_c", "rpm", "torque_nm", "vibration_mm_s",
            "power_output_mw", "fuel_flow_kg_s", "air_pressure_kpa",
            "exhaust_gas_temp_c", "oil_temp_c",
        ]
        for s in sensors:
            if s not in df.columns:
                continue
            fault_mean = float(faults[s].mean()) if fault_count > 0 else 0
            normal_mean = float(normals[s].mean()) if len(normals) > 0 else 0
            normal_std = float(normals[s].std()) if len(normals) > 1 else 1
            z_diff = abs(fault_mean - normal_mean) / max(normal_std, 0.001)
            sensor_deviation[s] = {
                "fault_mean": round(fault_mean, 4),
                "normal_mean": round(normal_mean, 4),
                "z_difference": round(z_diff, 4),
                "significant": z_diff > 1.5,
            }

        # Sort by significance
        ranked = sorted(
            sensor_deviation.items(), key=lambda x: x[1]["z_difference"], reverse=True
        )

        # Fault clusters — transitions from 0→1
        fault_starts = []
        prev = 0
        for idx, row in df.iterrows():
            if row["fault"] == 1 and prev == 0:
                fault_starts.append(str(row["timestamp"]))
            prev = int(row["fault"])

        return {
            "total_rows": total,
            "fault_count": fault_count,
            "normal_count": total - fault_count,
            "fault_rate": round(fault_count / max(total, 1), 4),
            "sensor_deviation_ranking": [
                {"sensor": s, **d} for s, d in ranked
            ],
            "fault_cluster_count": len(fault_starts),
            "fault_cluster_starts": fault_starts[:10],
            "avg_health_during_fault": round(float(faults["health_score"].mean()), 2) if fault_count > 0 else None,
            "avg_health_during_normal": round(float(normals["health_score"].mean()), 2) if len(normals) > 0 else None,
        }

    async def get_correlation(
        self,
        sensor_a: str,
        sensor_b: str,
    ) -> dict:
        """Compute correlation between two columns."""
        df = self._fabric._df
        valid_cols = list(df.columns)

        if sensor_a not in valid_cols:
            return {"error": f"Unknown column: {sensor_a}"}
        if sensor_b not in valid_cols:
            return {"error": f"Unknown column: {sensor_b}"}

        corr = float(df[sensor_a].astype(float).corr(df[sensor_b].astype(float)))

        interpretation = (
            "strong positive" if corr > 0.7 else
            "moderate positive" if corr > 0.3 else
            "weak/no" if corr > -0.3 else
            "moderate negative" if corr > -0.7 else
            "strong negative"
        )

        return {
            "sensor_a": sensor_a,
            "sensor_b": sensor_b,
            "pearson_r": round(corr, 4) if not math.isnan(corr) else 0.0,
            "interpretation": interpretation,
            "data_points": int(df[sensor_a].count()),
        }

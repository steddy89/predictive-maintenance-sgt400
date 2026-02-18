"""
============================================================================
Azure AI Foundry Diagnostic Agent — SGT400 Predictive Maintenance
============================================================================

An autonomous AI agent that reasons about turbine sensor data, identifies
root causes of anomalies, and generates actionable maintenance recommendations.

Architecture (Azure AI Foundry Agent Service pattern):
  ┌──────────────────────────────────────────────────────┐
  │                 Diagnostic Agent                       │
  │  ┌───────────┐ ┌────────────┐ ┌───────────────────┐ │
  │  │  Planner  │→│  Tool      │→│  Response         │ │
  │  │  (reason) │ │  Executor  │ │  Synthesiser      │ │
  │  └───────────┘ └────────────┘ └───────────────────┘ │
  │        ↑              │                               │
  │        │         ┌────┴────┐                          │
  │        └─────────│ Fabric  │                          │
  │                  │ Data    │                          │
  │                  │ Agent   │                          │
  │                  └─────────┘                          │
  └──────────────────────────────────────────────────────┘

The Diagnostic Agent follows the Azure AI Foundry agent pattern:
  1. Receives a user question or auto-triggered event
  2. Plans which tools/data it needs (chain-of-thought)
  3. Invokes Fabric Data Agent tools to gather evidence
  4. Synthesises a structured diagnostic response

Reference:
  https://learn.microsoft.com/azure/ai-services/agents/overview
  https://learn.microsoft.com/azure/ai-foundry/agents
"""

import json
import logging
from datetime import datetime
from typing import Any

from app.agents.fabric_data_agent import FabricDataAgent
from app.services.fabric_client import FabricDataService, SENSOR_THRESHOLDS

logger = logging.getLogger(__name__)


# ── System prompt for the Diagnostic Agent ──────────────────────────────

DIAGNOSTIC_SYSTEM_PROMPT = """You are the SGT-400 Gas Turbine Diagnostic Agent, 
an AI specialist in industrial gas turbine predictive maintenance. You are part of 
the Azure AI Foundry Agent Service and Microsoft Fabric ecosystem.

Your capabilities:
1. DIAGNOSE sensor anomalies using statistical analysis from the Fabric Data Agent
2. IDENTIFY root causes by correlating sensor patterns during fault events
3. PREDICT failure probability based on degradation trends
4. RECOMMEND maintenance actions with priority and timeline
5. ANALYSE historical patterns to identify recurring issues

You have access to the Fabric Data Agent which can:
- query_sensor_data: Retrieve raw sensor telemetry from the Fabric Lakehouse
- compute_statistics: Calculate statistical summaries
- detect_anomaly_window: Find anomalous readings using z-score analysis
- get_fault_analysis: Analyse fault patterns and sensor deviations
- get_correlation: Compute cross-correlation between sensors

When responding, structure your analysis as:
1. **Observation**: What the data shows
2. **Analysis**: What it means for turbine health
3. **Root Cause**: Most likely cause of any issues
4. **Recommendation**: Specific maintenance actions
5. **Risk Level**: LOW / MEDIUM / HIGH / CRITICAL

Always cite specific sensor values and thresholds in your analysis.
Format sensor values with units (e.g. 924.8°C, 15651 RPM, 1.675 mm/s).
"""


class DiagnosticAgent:
    """
    Azure AI Foundry Diagnostic Agent for SGT-400 turbine analysis.

    Uses chain-of-thought reasoning with Fabric Data Agent tools
    to produce expert-level diagnostic assessments.
    """

    AGENT_NAME = "DiagnosticAgent"
    AGENT_DESCRIPTION = (
        "Azure AI Foundry agent that performs expert-level diagnostic "
        "analysis of SGT-400 gas turbine sensor data. Uses reasoning "
        "and tool-calling to identify anomalies, root causes, and "
        "maintenance recommendations."
    )

    # Pre-defined diagnostic workflows (agent plans)
    DIAGNOSTIC_PLANS = {
        "health_assessment": [
            {"tool": "compute_statistics", "args": {"sensor": "all", "last_n_rows": 100}},
            {"tool": "get_fault_analysis", "args": {"last_n_rows": 100}},
        ],
        "anomaly_investigation": [
            {"tool": "detect_anomaly_window", "args": {"sensor": "temperature_c", "last_n_rows": 200}},
            {"tool": "detect_anomaly_window", "args": {"sensor": "vibration_mm_s", "last_n_rows": 200}},
            {"tool": "detect_anomaly_window", "args": {"sensor": "exhaust_gas_temp_c", "last_n_rows": 200}},
            {"tool": "get_fault_analysis", "args": {"last_n_rows": 200}},
        ],
        "sensor_deep_dive": [
            {"tool": "query_sensor_data", "args": {"last_n_rows": 50}},
            {"tool": "compute_statistics", "args": {"sensor": "all"}},
        ],
        "fault_root_cause": [
            {"tool": "get_fault_analysis", "args": {}},
            {"tool": "get_correlation", "args": {"sensor_a": "vibration_mm_s", "sensor_b": "fault"}},
            {"tool": "get_correlation", "args": {"sensor_a": "temperature_c", "sensor_b": "fault"}},
            {"tool": "get_correlation", "args": {"sensor_a": "exhaust_gas_temp_c", "sensor_b": "fault"}},
            {"tool": "get_correlation", "args": {"sensor_a": "oil_temp_c", "sensor_b": "fault"}},
        ],
        "correlation_analysis": [
            {"tool": "get_correlation", "args": {"sensor_a": "temperature_c", "sensor_b": "rpm"}},
            {"tool": "get_correlation", "args": {"sensor_a": "vibration_mm_s", "sensor_b": "torque_nm"}},
            {"tool": "get_correlation", "args": {"sensor_a": "fuel_flow_kg_s", "sensor_b": "power_output_mw"}},
            {"tool": "get_correlation", "args": {"sensor_a": "exhaust_gas_temp_c", "sensor_b": "oil_temp_c"}},
        ],
    }

    def __init__(self, fabric_service: FabricDataService | None = None):
        self._fabric_service = fabric_service or FabricDataService()
        self._data_agent = FabricDataAgent(self._fabric_service)
        self._conversation_history: list[dict] = []
        logger.info(f"[{self.AGENT_NAME}] Initialised with Fabric Data Agent")

    # ------------------------------------------------------------------
    # Agent reasoning — plan → execute → synthesise
    # ------------------------------------------------------------------
    async def process_message(self, user_message: str) -> dict:
        """
        Main entry point: process a user message through the agent pipeline.

        Steps (Azure AI Foundry agent pattern):
          1. Plan: determine which tools/data we need
          2. Execute: invoke Fabric Data Agent tools
          3. Synthesise: produce a structured diagnostic response
        """
        start_time = datetime.now()

        # Save user message
        self._conversation_history.append({
            "role": "user",
            "content": user_message,
            "timestamp": start_time.isoformat(),
        })

        # Step 1: PLAN — determine the diagnostic workflow
        plan = self._plan(user_message)
        logger.info(f"[{self.AGENT_NAME}] Plan: {plan['workflow']} "
                     f"({len(plan['tool_calls'])} tools)")

        # Step 2: EXECUTE — invoke tools via Fabric Data Agent
        tool_results = []
        for tc in plan["tool_calls"]:
            result = await self._data_agent.invoke_tool(tc["tool"], tc["args"])
            tool_results.append(result)

        # Step 3: SYNTHESISE — produce the diagnostic response
        response = self._synthesise(user_message, plan, tool_results)

        # Record conversation
        elapsed = (datetime.now() - start_time).total_seconds()
        response["agent_metadata"] = {
            "agent": self.AGENT_NAME,
            "workflow": plan["workflow"],
            "tools_invoked": [tc["tool"] for tc in plan["tool_calls"]],
            "processing_time_s": round(elapsed, 3),
            "timestamp": datetime.now().isoformat(),
            "fabric_data_agent": self._data_agent.AGENT_NAME,
        }

        self._conversation_history.append({
            "role": "assistant",
            "content": response["diagnosis"],
            "timestamp": datetime.now().isoformat(),
        })

        return response

    # ------------------------------------------------------------------
    # Step 1: PLAN — match user intent to a diagnostic workflow
    # ------------------------------------------------------------------
    def _plan(self, message: str) -> dict:
        """Chain-of-thought planning: determine which tools to call."""
        msg_lower = message.lower()

        # Intent classification (would use Azure AI in production)
        if any(w in msg_lower for w in ["fault", "root cause", "why", "cause", "failure"]):
            workflow = "fault_root_cause"
        elif any(w in msg_lower for w in ["anomaly", "anomalies", "abnormal", "unusual", "outlier"]):
            workflow = "anomaly_investigation"
        elif any(w in msg_lower for w in ["correlat", "relationship", "relate", "depend"]):
            workflow = "correlation_analysis"
        elif any(w in msg_lower for w in ["sensor", "reading", "data", "show", "current", "latest"]):
            workflow = "sensor_deep_dive"
        else:
            workflow = "health_assessment"

        # Handle sensor-specific queries
        extra_tools = []
        for sensor_name in SENSOR_THRESHOLDS:
            pretty = sensor_name.replace("_", " ")
            if pretty in msg_lower or sensor_name in msg_lower:
                extra_tools.append({
                    "tool": "detect_anomaly_window",
                    "args": {"sensor": sensor_name, "last_n_rows": 200},
                })
                extra_tools.append({
                    "tool": "compute_statistics",
                    "args": {"sensor": sensor_name},
                })
                break

        tool_calls = list(self.DIAGNOSTIC_PLANS[workflow])
        if extra_tools:
            tool_calls = extra_tools + tool_calls

        return {
            "workflow": workflow,
            "intent": msg_lower,
            "tool_calls": tool_calls,
        }

    # ------------------------------------------------------------------
    # Step 3: SYNTHESISE — turn raw tool outputs into diagnostics
    # ------------------------------------------------------------------
    def _synthesise(
        self,
        user_message: str,
        plan: dict,
        tool_results: list[dict],
    ) -> dict:
        """Synthesise tool results into a structured diagnostic response."""
        workflow = plan["workflow"]

        if workflow == "health_assessment":
            return self._synthesise_health(tool_results)
        elif workflow == "anomaly_investigation":
            return self._synthesise_anomaly(tool_results)
        elif workflow == "fault_root_cause":
            return self._synthesise_fault_root_cause(tool_results)
        elif workflow == "correlation_analysis":
            return self._synthesise_correlation(tool_results)
        elif workflow == "sensor_deep_dive":
            return self._synthesise_sensor_dive(tool_results)
        else:
            return self._synthesise_health(tool_results)

    def _synthesise_health(self, results: list[dict]) -> dict:
        """Synthesise a health assessment."""
        stats = {}
        fault_info = {}
        for r in results:
            if r.get("tool") == "compute_statistics":
                stats = r.get("result", {}).get("statistics", {})
            elif r.get("tool") == "get_fault_analysis":
                fault_info = r.get("result", {})

        # Build diagnosis text
        health_stats = stats.get("health_score", {})
        avg_health = health_stats.get("mean", 0)
        min_health = health_stats.get("min", 0)
        fault_rate = fault_info.get("fault_rate", 0)
        fault_count = fault_info.get("fault_count", 0)

        if avg_health >= 85 and fault_rate < 0.1:
            risk = "LOW"
            status_text = "HEALTHY"
        elif avg_health >= 65 and fault_rate < 0.3:
            risk = "MEDIUM"
            status_text = "DEGRADED"
        elif avg_health >= 40:
            risk = "HIGH"
            status_text = "AT RISK"
        else:
            risk = "CRITICAL"
            status_text = "CRITICAL"

        # Top deviating sensors during faults
        deviation_ranking = fault_info.get("sensor_deviation_ranking", [])
        top_sensors = deviation_ranking[:3] if deviation_ranking else []

        # Build output
        lines = [
            f"## Turbine Health Assessment: **{status_text}**\n",
            f"**Average Health Score**: {avg_health:.1f}% (min: {min_health:.1f}%)",
            f"**Fault Rate**: {fault_rate*100:.1f}% ({fault_count} fault events)\n",
        ]

        if top_sensors:
            lines.append("### Sensors Most Affected During Faults")
            for s in top_sensors:
                lines.append(
                    f"- **{s['sensor'].replace('_', ' ').title()}**: "
                    f"z-difference = {s['z_difference']:.2f}σ "
                    f"(fault mean: {s['fault_mean']:.1f}, "
                    f"normal mean: {s['normal_mean']:.1f})"
                )

        lines.append(f"\n### Recommendation")
        if risk == "LOW":
            lines.append("Continue standard monitoring. No immediate action required.")
        elif risk == "MEDIUM":
            lines.append("Increase monitoring frequency. Schedule inspection in next maintenance window.")
        elif risk == "HIGH":
            lines.append("**Schedule maintenance within 7 days.** Monitor critical sensors closely.")
        else:
            lines.append("**IMMEDIATE ACTION REQUIRED.** Prepare for emergency shutdown if conditions worsen.")

        return {
            "diagnosis": "\n".join(lines),
            "risk_level": risk,
            "status": status_text,
            "metrics": {
                "avg_health": round(avg_health, 1),
                "min_health": round(min_health, 1),
                "fault_rate": round(fault_rate, 4),
                "fault_count": fault_count,
            },
            "top_deviation_sensors": [s["sensor"] for s in top_sensors],
            "raw_data": {"statistics": stats, "fault_analysis": fault_info},
        }

    def _synthesise_anomaly(self, results: list[dict]) -> dict:
        """Synthesise an anomaly investigation."""
        anomaly_summaries = []
        fault_info = {}

        for r in results:
            if r.get("tool") == "detect_anomaly_window":
                res = r.get("result", {})
                anomaly_summaries.append({
                    "sensor": res.get("sensor", "unknown"),
                    "total_anomalies": res.get("total_anomalies", 0),
                    "anomaly_rate": res.get("anomaly_rate", 0),
                    "window_size": res.get("window_size", 0),
                })
            elif r.get("tool") == "get_fault_analysis":
                fault_info = r.get("result", {})

        # Sort by anomaly rate
        anomaly_summaries.sort(key=lambda x: x["anomaly_rate"], reverse=True)

        max_rate = max((a["anomaly_rate"] for a in anomaly_summaries), default=0)
        if max_rate > 0.3:
            risk = "CRITICAL"
        elif max_rate > 0.15:
            risk = "HIGH"
        elif max_rate > 0.05:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        lines = [f"## Anomaly Investigation Report\n"]

        if not anomaly_summaries:
            lines.append("No anomalies detected. System operating within normal parameters.")
        else:
            lines.append("### Anomaly Summary by Sensor\n")
            lines.append("| Sensor | Anomalies | Rate | Status |")
            lines.append("|--------|-----------|------|--------|")
            for a in anomaly_summaries:
                status = (
                    "🔴 CRITICAL" if a["anomaly_rate"] > 0.3
                    else "🟡 WARNING" if a["anomaly_rate"] > 0.05
                    else "🟢 NORMAL"
                )
                lines.append(
                    f"| {a['sensor'].replace('_', ' ').title()} "
                    f"| {a['total_anomalies']} "
                    f"| {a['anomaly_rate']*100:.1f}% "
                    f"| {status} |"
                )

        fault_rate = fault_info.get("fault_rate", 0)
        lines.append(f"\n**Fault correlation**: {fault_rate*100:.1f}% of readings are fault-flagged")

        lines.append(f"\n### Risk Level: **{risk}**")
        if risk in ("CRITICAL", "HIGH"):
            lines.append("⚠ Immediate investigation recommended. Check the highest-anomaly sensors.")
        elif risk == "MEDIUM":
            lines.append("Monitor closely. Trend may indicate early degradation.")
        else:
            lines.append("System nominal. Continue standard monitoring.")

        return {
            "diagnosis": "\n".join(lines),
            "risk_level": risk,
            "anomaly_summary": anomaly_summaries,
            "raw_data": {"fault_analysis": fault_info},
        }

    def _synthesise_fault_root_cause(self, results: list[dict]) -> dict:
        """Synthesise root cause analysis from fault + correlation data."""
        fault_info = {}
        correlations = []

        for r in results:
            if r.get("tool") == "get_fault_analysis":
                fault_info = r.get("result", {})
            elif r.get("tool") == "get_correlation":
                correlations.append(r.get("result", {}))

        deviation_ranking = fault_info.get("sensor_deviation_ranking", [])
        fault_rate = fault_info.get("fault_rate", 0)
        fault_count = fault_info.get("fault_count", 0)

        # Sort correlations by absolute value
        correlations.sort(key=lambda x: abs(x.get("pearson_r", 0)), reverse=True)

        risk = "HIGH" if fault_rate > 0.2 else "MEDIUM" if fault_rate > 0.1 else "LOW"

        lines = [
            f"## Root Cause Analysis\n",
            f"**Total Faults**: {fault_count} ({fault_rate*100:.1f}% of all readings)\n",
        ]

        # Top deviators
        if deviation_ranking:
            lines.append("### Primary Fault Indicators (by statistical deviation)\n")
            for i, d in enumerate(deviation_ranking[:5]):
                marker = "🔴" if d.get("significant") else "⚪"
                lines.append(
                    f"{i+1}. {marker} **{d['sensor'].replace('_', ' ').title()}** — "
                    f"z-difference: {d['z_difference']:.2f}σ "
                    f"(fault avg: {d['fault_mean']:.1f}, normal avg: {d['normal_mean']:.1f})"
                )

        # Correlations
        if correlations:
            lines.append("\n### Sensor-Fault Correlation\n")
            lines.append("| Sensor | Pearson r | Interpretation |")
            lines.append("|--------|-----------|----------------|")
            for c in correlations:
                lines.append(
                    f"| {c.get('sensor_a', '?').replace('_', ' ').title()} "
                    f"| {c.get('pearson_r', 0):.4f} "
                    f"| {c.get('interpretation', '')} |"
                )

        # Root cause determination
        top_cause = deviation_ranking[0] if deviation_ranking else None
        if top_cause:
            lines.append(f"\n### Most Likely Root Cause")
            lines.append(
                f"**{top_cause['sensor'].replace('_', ' ').title()}** shows the "
                f"largest deviation during fault events ({top_cause['z_difference']:.2f}σ). "
                f"This sensor should be the primary focus of investigation."
            )

        lines.append(f"\n### Recommendation")
        if risk == "HIGH":
            lines.append("**Schedule immediate inspection** of the top-deviating sensors. Consider partial shutdown for diagnostics.")
        elif risk == "MEDIUM":
            lines.append("Plan inspection during next maintenance window. Increase monitoring frequency on flagged sensors.")
        else:
            lines.append("Low fault activity. Continue standard predictive maintenance schedule.")

        return {
            "diagnosis": "\n".join(lines),
            "risk_level": risk,
            "top_fault_indicators": [d["sensor"] for d in deviation_ranking[:3]],
            "correlations": correlations,
            "raw_data": {"fault_analysis": fault_info},
        }

    def _synthesise_correlation(self, results: list[dict]) -> dict:
        """Synthesise correlation analysis."""
        correlations = []
        for r in results:
            if r.get("tool") == "get_correlation":
                correlations.append(r.get("result", {}))

        correlations.sort(key=lambda x: abs(x.get("pearson_r", 0)), reverse=True)

        lines = [
            "## Sensor Correlation Analysis\n",
            "| Sensor Pair | Pearson r | Interpretation |",
            "|-------------|-----------|----------------|",
        ]
        for c in correlations:
            a = c.get("sensor_a", "?").replace("_", " ").title()
            b = c.get("sensor_b", "?").replace("_", " ").title()
            lines.append(
                f"| {a} ↔ {b} "
                f"| {c.get('pearson_r', 0):.4f} "
                f"| {c.get('interpretation', '')} |"
            )

        strong = [c for c in correlations if abs(c.get("pearson_r", 0)) > 0.5]
        if strong:
            lines.append("\n### Key Findings")
            for c in strong:
                a = c["sensor_a"].replace("_", " ").title()
                b = c["sensor_b"].replace("_", " ").title()
                lines.append(
                    f"- **{a}** and **{b}** have a {c['interpretation']} "
                    f"correlation (r={c['pearson_r']:.4f}). "
                    f"Changes in one may predict changes in the other."
                )

        return {
            "diagnosis": "\n".join(lines),
            "risk_level": "LOW",
            "correlations": correlations,
        }

    def _synthesise_sensor_dive(self, results: list[dict]) -> dict:
        """Synthesise sensor deep-dive."""
        data_rows = []
        stats = {}

        for r in results:
            res = r.get("result", {})
            if r.get("tool") == "query_sensor_data":
                data_rows = res.get("data", [])
            elif r.get("tool") == "compute_statistics":
                stats = res.get("statistics", {})

        lines = [
            f"## Sensor Data Deep Dive\n",
            f"**Rows Retrieved**: {len(data_rows)}\n",
        ]

        if stats:
            lines.append("### Current Statistics\n")
            lines.append("| Sensor | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            for sensor, s in stats.items():
                if sensor in ("health_score", "anomaly_score"):
                    continue
                lines.append(
                    f"| {sensor.replace('_', ' ').title()} "
                    f"| {s['mean']:.1f} "
                    f"| {s['std']:.1f} "
                    f"| {s['min']:.1f} "
                    f"| {s['max']:.1f} |"
                )

        # Check for out-of-range sensors
        warnings = []
        for sensor, s in stats.items():
            thresh = SENSOR_THRESHOLDS.get(sensor, {})
            if thresh.get("warn") and s.get("max", 0) > thresh["warn"]:
                warnings.append(f"⚠ **{sensor.replace('_', ' ').title()}** max ({s['max']:.1f}) exceeds warning threshold ({thresh['warn']})")
            if thresh.get("crit") and s.get("max", 0) > thresh["crit"]:
                warnings.append(f"🔴 **{sensor.replace('_', ' ').title()}** max ({s['max']:.1f}) exceeds CRITICAL threshold ({thresh['crit']})")

        if warnings:
            lines.append("\n### ⚠ Threshold Alerts\n")
            lines.extend(warnings)

        risk = "HIGH" if any("CRITICAL" in w for w in warnings) else \
               "MEDIUM" if warnings else "LOW"

        return {
            "diagnosis": "\n".join(lines),
            "risk_level": risk,
            "statistics": stats,
            "sample_rows": data_rows[:5],
        }

    # ------------------------------------------------------------------
    # Conversation management
    # ------------------------------------------------------------------
    def get_conversation_history(self) -> list[dict]:
        """Return the conversation history."""
        return self._conversation_history

    def clear_conversation(self):
        """Reset the conversation."""
        self._conversation_history = []

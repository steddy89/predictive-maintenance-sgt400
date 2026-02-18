"""
============================================================================
Orchestrator Agent — SGT400 Predictive Maintenance
============================================================================

Coordinates the Fabric Data Agent and Diagnostic Agent into a unified
multi-agent system following Azure AI Foundry Agent Service patterns.

Architecture:
  ┌─────────────────────────────────────────────────┐
  │               Orchestrator Agent                  │
  │                                                   │
  │   User Message ──► Intent Router                  │
  │                        │                          │
  │             ┌──────────┼──────────┐               │
  │             ▼          ▼          ▼               │
  │        Data Query   Diagnosis  Auto-Check         │
  │             │          │          │               │
  │      Fabric Data    Diagnostic   Both             │
  │        Agent         Agent      Agents            │
  │             │          │          │               │
  │             └──────────┼──────────┘               │
  │                        ▼                          │
  │                Response Builder                   │
  └─────────────────────────────────────────────────┘

The Orchestrator:
  1. Receives all user/system messages
  2. Routes to the correct specialist agent(s)
  3. Merges responses into a unified output
  4. Maintains conversation state
  5. Provides auto-triggered diagnostics (event-driven)
"""

import json
import logging
from datetime import datetime
from typing import Any

from app.agents.fabric_data_agent import FabricDataAgent
from app.agents.diagnostic_agent import DiagnosticAgent
from app.services.fabric_client import FabricDataService

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """
    Central orchestrator that coordinates the multi-agent system.

    Follows the Azure AI Foundry multi-agent pattern where a supervisory
    agent routes work to specialist agents.
    """

    AGENT_NAME = "OrchestratorAgent"

    # Suggested prompts for the chat UI
    SUGGESTED_PROMPTS = [
        {"text": "What is the current turbine health?", "category": "health"},
        {"text": "Analyze anomalies in recent sensor data", "category": "anomaly"},
        {"text": "What are the root causes of faults?", "category": "fault"},
        {"text": "Show sensor correlation analysis", "category": "correlation"},
        {"text": "Show me the latest sensor readings", "category": "data"},
        {"text": "Which sensors deviate most during faults?", "category": "fault"},
    ]

    def __init__(self, fabric_service: FabricDataService | None = None):
        self._fabric_service = fabric_service or FabricDataService()
        self._data_agent = FabricDataAgent(self._fabric_service)
        self._diagnostic_agent = DiagnosticAgent(self._fabric_service)
        self._conversation: list[dict] = []
        logger.info(f"[{self.AGENT_NAME}] Multi-agent system initialised")
        logger.info(f"  ├── {self._data_agent.AGENT_NAME} (5 tools)")
        logger.info(f"  └── {self._diagnostic_agent.AGENT_NAME}")

    # ------------------------------------------------------------------
    # Main chat interface
    # ------------------------------------------------------------------
    async def chat(self, user_message: str) -> dict:
        """
        Process a user message through the multi-agent pipeline.

        Returns a unified response dict with diagnosis, data, metadata.
        """
        start = datetime.now()

        # Classify intent
        intent = self._classify_intent(user_message)
        logger.info(f"[{self.AGENT_NAME}] Intent: {intent} for '{user_message[:60]}...'")

        # Route to agents
        if intent == "data_query":
            response = await self._handle_data_query(user_message)
        elif intent in ("health", "anomaly", "fault", "correlation"):
            response = await self._diagnostic_agent.process_message(user_message)
        else:
            response = await self._diagnostic_agent.process_message(user_message)

        elapsed = (datetime.now() - start).total_seconds()

        # Wrap in orchestrator envelope
        output = {
            "message": response.get("diagnosis", ""),
            "risk_level": response.get("risk_level", "LOW"),
            "intent": intent,
            "agents_used": self._agents_for_intent(intent),
            "processing_time_s": round(elapsed, 3),
            "timestamp": datetime.now().isoformat(),
            "metadata": response.get("agent_metadata", {}),
            "data": {
                k: v for k, v in response.items()
                if k not in ("diagnosis", "risk_level", "agent_metadata", "raw_data")
            },
        }

        # Save conversation
        self._conversation.append({"role": "user", "content": user_message, "ts": start.isoformat()})
        self._conversation.append({"role": "assistant", "content": output["message"], "ts": datetime.now().isoformat()})

        return output

    # ------------------------------------------------------------------
    # Auto-triggered diagnostics (called from live endpoints)
    # ------------------------------------------------------------------
    async def auto_diagnose(self, live_reading: dict) -> dict | None:
        """
        Automatically run diagnostics when a live reading has high risk.
        Called by the /live endpoint when anomaly or fault triggers.

        Returns a mini-diagnostic or None if system is healthy.
        """
        is_fault = live_reading.get("fault", False)
        anomaly = live_reading.get("anomaly_score", 0)
        health = live_reading.get("health_score", 100)

        if not is_fault and anomaly < 0.6 and health > 60:
            return None  # System healthy, no auto-diagnosis needed

        # Build contextual alert
        alert_type = "FAULT" if is_fault else "ANOMALY" if anomaly >= 0.6 else "DEGRADED"

        # Quick stats from data agent
        quick_stats = await self._data_agent.invoke_tool(
            "compute_statistics", {"sensor": "all", "last_n_rows": 20}
        )

        sensors = live_reading.get("sensors", {})
        lines = [
            f"⚡ **Auto-Diagnostic Alert: {alert_type}**\n",
            f"Health: {health:.0f}% | Anomaly Score: {anomaly:.2f} | Fault: {'YES' if is_fault else 'NO'}\n",
            "**Current Sensor Readings:**",
        ]

        for name, val in sensors.items():
            lines.append(f"- {name.replace('_', ' ').title()}: {val:.1f}")

        return {
            "alert_type": alert_type,
            "message": "\n".join(lines),
            "risk_level": "CRITICAL" if is_fault else "HIGH",
            "live_reading": live_reading,
            "quick_stats": quick_stats.get("result", {}),
        }

    # ------------------------------------------------------------------
    # Data query handler (direct Fabric Data Agent call)
    # ------------------------------------------------------------------
    async def _handle_data_query(self, message: str) -> dict:
        """Handle simple data retrieval queries."""
        msg = message.lower()

        # Pick the best tool
        if any(w in msg for w in ["stat", "average", "mean", "summary"]):
            result = await self._data_agent.invoke_tool(
                "compute_statistics", {"sensor": "all", "last_n_rows": 100}
            )
        elif any(w in msg for w in ["anomal", "outlier"]):
            result = await self._data_agent.invoke_tool(
                "detect_anomaly_window", {"sensor": "vibration_mm_s", "last_n_rows": 200}
            )
        elif any(w in msg for w in ["fault", "failure"]):
            result = await self._data_agent.invoke_tool("get_fault_analysis", {})
        else:
            result = await self._data_agent.invoke_tool(
                "query_sensor_data", {"last_n_rows": 20}
            )

        data = result.get("result", {})
        lines = [f"## Fabric Data Agent Response\n"]

        if "statistics" in data:
            lines.append("### Sensor Statistics\n")
            lines.append("| Sensor | Mean | Std | Min | Max |")
            lines.append("|--------|------|-----|-----|-----|")
            for sensor, s in data["statistics"].items():
                lines.append(
                    f"| {sensor.replace('_', ' ').title()} "
                    f"| {s['mean']:.1f} | {s['std']:.1f} "
                    f"| {s['min']:.1f} | {s['max']:.1f} |"
                )
        elif "data" in data:
            lines.append(f"Retrieved {data.get('total_rows', 0)} rows of sensor data.")
        else:
            lines.append(json.dumps(data, indent=2, default=str))

        return {
            "diagnosis": "\n".join(lines),
            "risk_level": "LOW",
            "agent_metadata": {
                "agent": self._data_agent.AGENT_NAME,
                "tool": result.get("tool", "unknown"),
                "timestamp": datetime.now().isoformat(),
            },
        }

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------
    def _classify_intent(self, message: str) -> str:
        """Classify user message into an intent category."""
        msg = message.lower()
        if any(w in msg for w in ["health", "status", "overview", "how is", "how's"]):
            return "health"
        if any(w in msg for w in ["anomaly", "anomalies", "unusual", "abnormal", "outlier"]):
            return "anomaly"
        if any(w in msg for w in ["fault", "root cause", "why", "failure", "cause", "diagnos"]):
            return "fault"
        if any(w in msg for w in ["correlat", "relationship", "relate", "depend"]):
            return "correlation"
        if any(w in msg for w in ["data", "reading", "sensor", "show", "latest", "current", "value"]):
            return "data_query"
        return "health"  # default

    def _agents_for_intent(self, intent: str) -> list[str]:
        if intent == "data_query":
            return [self._data_agent.AGENT_NAME]
        return [self._data_agent.AGENT_NAME, self._diagnostic_agent.AGENT_NAME]

    # ------------------------------------------------------------------
    # System info
    # ------------------------------------------------------------------
    def get_system_info(self) -> dict:
        """Return multi-agent system status."""
        return {
            "orchestrator": self.AGENT_NAME,
            "agents": [
                {
                    "name": self._data_agent.AGENT_NAME,
                    "description": self._data_agent.AGENT_DESCRIPTION,
                    "tools": [t["function"]["name"] for t in self._data_agent.TOOLS],
                },
                {
                    "name": self._diagnostic_agent.AGENT_NAME,
                    "description": self._diagnostic_agent.AGENT_DESCRIPTION,
                    "tools": list(self._diagnostic_agent.DIAGNOSTIC_PLANS.keys()),
                },
            ],
            "available_workflows": list(self._diagnostic_agent.DIAGNOSTIC_PLANS.keys()),
            "suggested_prompts": self.SUGGESTED_PROMPTS,
            "conversation_length": len(self._conversation),
        }

    def get_conversation(self) -> list[dict]:
        return self._conversation

    def clear_conversation(self):
        self._conversation = []
        self._diagnostic_agent.clear_conversation()

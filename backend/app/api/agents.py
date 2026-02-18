"""
============================================================================
Agent API Endpoints — SGT400 Predictive Maintenance
============================================================================

REST API for the multi-agent system (Azure AI Foundry Agent Service pattern).

Endpoints:
  POST /api/v1/agent/chat          — Send a message, get agent response
  GET  /api/v1/agent/status        — Agent system health & metadata
  GET  /api/v1/agent/tools         — List available agent tools
  POST /api/v1/agent/tool          — Invoke a specific Fabric Data Agent tool
  GET  /api/v1/agent/conversation  — Get conversation history
  DELETE /api/v1/agent/conversation — Clear conversation
  POST /api/v1/agent/auto-diagnose — Auto-diagnose from live reading
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["Agent"])

# Global reference set by main.py on startup
_orchestrator = None


def set_orchestrator(orchestrator):
    """Called by main.py lifespan to inject the orchestrator instance."""
    global _orchestrator
    _orchestrator = orchestrator


def _get_orchestrator():
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Agent system not initialised")
    return _orchestrator


# ── Request / Response schemas ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="User message")

class ChatResponse(BaseModel):
    message: str
    risk_level: str
    intent: str
    agents_used: list[str]
    processing_time_s: float
    timestamp: str
    metadata: dict = {}
    data: dict = {}

class ToolRequest(BaseModel):
    tool_name: str = Field(..., description="Name of the Fabric Data Agent tool to invoke")
    arguments: dict = Field(default_factory=dict, description="Tool arguments")

class ToolResponse(BaseModel):
    tool: str
    status: str
    result: dict = {}

class AutoDiagnoseRequest(BaseModel):
    live_reading: dict = Field(..., description="Live sensor reading from /api/v1/turbine/live")


# ── Endpoints ───────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest):
    """
    Send a natural language message to the AI agent system.

    The Orchestrator routes the message to the appropriate specialist agent(s):
    - Fabric Data Agent: raw data queries, statistics, anomaly detection
    - Diagnostic Agent: health assessment, root cause analysis, recommendations

    Example messages:
    - "What is the current turbine health?"
    - "Analyze anomalies in recent sensor data"
    - "What are the root causes of faults?"
    - "Show sensor correlation analysis"
    """
    orch = _get_orchestrator()
    try:
        result = await orch.chat(req.message)
        return ChatResponse(**result)
    except Exception as e:
        logger.error(f"Agent chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@router.get("/status")
async def agent_status():
    """
    Get the multi-agent system status, available agents, tools, and suggested prompts.
    """
    orch = _get_orchestrator()
    return orch.get_system_info()


@router.get("/tools")
async def agent_tools():
    """
    List all available Fabric Data Agent tools with their schemas
    (following Azure AI Foundry function-calling format).
    """
    orch = _get_orchestrator()
    data_agent = orch._data_agent
    return {
        "agent": data_agent.AGENT_NAME,
        "tools": data_agent.TOOLS,
        "total": len(data_agent.TOOLS),
    }


@router.post("/tool", response_model=ToolResponse)
async def invoke_tool(req: ToolRequest):
    """
    Directly invoke a Fabric Data Agent tool by name.

    Available tools:
    - query_sensor_data: Retrieve raw sensor telemetry
    - compute_statistics: Calculate statistical summaries
    - detect_anomaly_window: Find anomalous readings
    - get_fault_analysis: Analyse fault patterns
    - get_correlation: Compute sensor correlation
    """
    orch = _get_orchestrator()
    try:
        result = await orch._data_agent.invoke_tool(req.tool_name, req.arguments)
        return ToolResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Tool invoke error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation")
async def get_conversation():
    """Get the conversation history."""
    orch = _get_orchestrator()
    return {
        "messages": orch.get_conversation(),
        "total": len(orch.get_conversation()),
    }


@router.delete("/conversation")
async def clear_conversation():
    """Clear the conversation history."""
    orch = _get_orchestrator()
    orch.clear_conversation()
    return {"status": "cleared"}


@router.post("/auto-diagnose")
async def auto_diagnose(req: AutoDiagnoseRequest):
    """
    Auto-triggered diagnostic analysis from a live reading.
    Returns diagnosis only if the reading indicates issues (fault/anomaly/degraded).
    Returns null if the system is healthy.
    """
    orch = _get_orchestrator()
    try:
        result = await orch.auto_diagnose(req.live_reading)
        if result is None:
            return {"triggered": False, "message": "System healthy — no auto-diagnosis needed"}
        return {"triggered": True, **result}
    except Exception as e:
        logger.error(f"Auto-diagnose error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

"""
============================================================================
Microsoft Fabric Data Pipeline Orchestrator
============================================================================
Programmatic orchestration of the medallion architecture pipeline
using the Microsoft Fabric REST API.

This script can be run standalone or scheduled as a Fabric notebook job.

Reference:
  - https://learn.microsoft.com/rest/api/fabric/core/items
  - https://learn.microsoft.com/fabric/data-engineering/lakehouse-overview
============================================================================
"""

import os
import json
import time
import logging
from datetime import datetime

import httpx
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FABRIC_API_BASE = "https://api.fabric.microsoft.com/v1"


class FabricPipelineOrchestrator:
    """Orchestrates the medallion pipeline via Fabric REST API."""

    def __init__(
        self,
        workspace_id: str | None = None,
        lakehouse_id: str | None = None,
    ):
        self.workspace_id = workspace_id or os.getenv("FABRIC_WORKSPACE_ID")
        self.lakehouse_id = lakehouse_id or os.getenv("FABRIC_LAKEHOUSE_ID")

        self.credential = DefaultAzureCredential()
        self._token = None

    def _get_token(self) -> str:
        """Get or refresh the Fabric API access token."""
        if self._token is None:
            token = self.credential.get_token("https://api.fabric.microsoft.com/.default")
            self._token = token.token
        return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Content-Type": "application/json",
        }

    def run_notebook(
        self,
        notebook_name: str,
        parameters: dict | None = None,
        timeout_minutes: int = 60,
    ) -> dict:
        """Execute a Fabric notebook and wait for completion."""
        url = (
            f"{FABRIC_API_BASE}/workspaces/{self.workspace_id}"
            f"/items/{notebook_name}/jobs/instances?jobType=RunNotebook"
        )

        payload = {}
        if parameters:
            payload["executionData"] = {
                "parameters": {
                    k: {"value": str(v), "type": "string"}
                    for k, v in parameters.items()
                }
            }

        logger.info(f"Starting notebook: {notebook_name}")

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            resp.raise_for_status()

            # Poll for completion
            location = resp.headers.get("Location")
            if location:
                return self._poll_job(client, location, timeout_minutes)

        return {"status": "submitted"}

    def _poll_job(
        self, client: httpx.Client, location_url: str, timeout_minutes: int
    ) -> dict:
        """Poll a job until completion."""
        deadline = time.time() + timeout_minutes * 60
        poll_interval = 15

        while time.time() < deadline:
            resp = client.get(location_url, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status", "Unknown")

            logger.info(f"  Job status: {status}")

            if status in ("Completed", "Succeeded"):
                return {"status": "success", "details": data}
            elif status in ("Failed", "Cancelled"):
                return {"status": "failed", "details": data}

            time.sleep(poll_interval)

        return {"status": "timeout"}

    def run_full_pipeline(self) -> dict:
        """Execute the complete Bronze → Silver → Gold pipeline."""
        pipeline_start = datetime.now()
        results = {"start_time": pipeline_start.isoformat(), "stages": {}}

        # Load pipeline config
        config_path = os.path.join(os.path.dirname(__file__), "fabric_pipeline.json")
        with open(config_path) as f:
            config = json.load(f)

        for stage in config["stages"]:
            stage_name = stage["stage_id"]
            logger.info(f"\n{'=' * 60}")
            logger.info(f"STAGE: {stage['name']}")
            logger.info(f"{'=' * 60}")

            try:
                result = self.run_notebook(
                    notebook_name=stage["notebook_path"],
                    parameters=stage.get("parameters"),
                    timeout_minutes=stage.get("timeout_minutes", 60),
                )
                results["stages"][stage_name] = result
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                results["stages"][stage_name] = {"status": "error", "error": str(e)}

                # Check retry policy
                retries = stage.get("retry_policy", {}).get("max_retries", 0)
                retry_interval = stage.get("retry_policy", {}).get(
                    "retry_interval_seconds", 60
                )
                for attempt in range(1, retries + 1):
                    logger.info(f"  Retry {attempt}/{retries} in {retry_interval}s")
                    time.sleep(retry_interval)
                    try:
                        result = self.run_notebook(
                            notebook_name=stage["notebook_path"],
                            parameters=stage.get("parameters"),
                            timeout_minutes=stage.get("timeout_minutes", 60),
                        )
                        results["stages"][stage_name] = result
                        break
                    except Exception as re:
                        logger.error(f"  Retry {attempt} failed: {re}")

        pipeline_end = datetime.now()
        results["end_time"] = pipeline_end.isoformat()
        results["duration_seconds"] = (pipeline_end - pipeline_start).total_seconds()

        logger.info(f"\nPipeline completed in {results['duration_seconds']:.1f}s")
        return results


if __name__ == "__main__":
    orchestrator = FabricPipelineOrchestrator()
    results = orchestrator.run_full_pipeline()
    print(json.dumps(results, indent=2))

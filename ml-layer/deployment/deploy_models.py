"""
============================================================================
Azure AI Foundry - Model Deployment Script
============================================================================
Deploys trained models as managed online endpoints in Azure AI Foundry.

Reference:
  - https://learn.microsoft.com/azure/machine-learning/how-to-deploy-online-endpoints
  - https://learn.microsoft.com/azure/ai-foundry/how-to/deploy-models-managed
"""

import os
import time
import logging
from datetime import datetime

from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    CodeConfiguration,
    ProbeSettings,
)
from azure.identity import DefaultAzureCredential

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """Deploys ML models to Azure AI Foundry managed online endpoints."""
    
    def __init__(
        self,
        subscription_id: str | None = None,
        resource_group: str | None = None,
        workspace_name: str | None = None,
    ):
        self.subscription_id = subscription_id or os.getenv("AZURE_SUBSCRIPTION_ID")
        self.resource_group = resource_group or os.getenv("AZURE_RESOURCE_GROUP")
        self.workspace_name = workspace_name or os.getenv("AZURE_ML_WORKSPACE")
        
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name,
        )
        logger.info(f"Connected to workspace: {self.workspace_name}")
    
    def create_endpoint(self, endpoint_name: str, description: str = "") -> ManagedOnlineEndpoint:
        """Create a managed online endpoint."""
        
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description=description or f"SGT400 Predictive Maintenance - {endpoint_name}",
            auth_mode="key",
            tags={
                "project": "predictive-maintenance-sgt400",
                "created": datetime.now().isoformat(),
            },
        )
        
        logger.info(f"Creating endpoint '{endpoint_name}'...")
        result = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        logger.info(f"Endpoint created: {result.scoring_uri}")
        
        return result
    
    def deploy_anomaly_model(
        self,
        endpoint_name: str = "sgt400-anomaly-endpoint",
        model_name: str = "sgt400-anomaly-detection",
        model_version: str = "1",
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
    ) -> ManagedOnlineDeployment:
        """Deploy the anomaly detection model."""
        
        # Create endpoint if not exists
        try:
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
            logger.info(f"Using existing endpoint: {endpoint_name}")
        except Exception:
            endpoint = self.create_endpoint(
                endpoint_name,
                "SGT400 Anomaly Detection - Isolation Forest"
            )
        
        # Get registered model
        model = self.ml_client.models.get(model_name, version=model_version)
        
        # Define environment
        env = Environment(
            name="sgt400-anomaly-env",
            conda_file="deployment/conda_anomaly.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
            description="Environment for SGT400 anomaly detection model",
        )
        
        # Create deployment
        deployment = ManagedOnlineDeployment(
            name="anomaly-blue",
            endpoint_name=endpoint_name,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code="deployment/",
                scoring_script="score_anomaly.py",
            ),
            instance_type=instance_type,
            instance_count=instance_count,
            liveness_probe=ProbeSettings(initial_delay=300),
            readiness_probe=ProbeSettings(initial_delay=300),
            tags={
                "model_type": "IsolationForest",
                "version": model_version,
            },
        )
        
        logger.info(f"Deploying model '{model_name}' to '{endpoint_name}'...")
        result = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        # Set 100% traffic to this deployment
        endpoint.traffic = {"anomaly-blue": 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")
        return result
    
    def deploy_forecasting_model(
        self,
        endpoint_name: str = "sgt400-forecast-endpoint",
        model_name: str = "sgt400-sensor-forecasting",
        model_version: str = "1",
        instance_type: str = "Standard_DS3_v2",
        instance_count: int = 1,
    ) -> ManagedOnlineDeployment:
        """Deploy the LSTM forecasting model."""
        
        try:
            endpoint = self.ml_client.online_endpoints.get(endpoint_name)
        except Exception:
            endpoint = self.create_endpoint(
                endpoint_name,
                "SGT400 Sensor Forecasting - LSTM"
            )
        
        model = self.ml_client.models.get(model_name, version=model_version)
        
        env = Environment(
            name="sgt400-forecast-env",
            conda_file="deployment/conda_forecast.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
        )
        
        deployment = ManagedOnlineDeployment(
            name="forecast-blue",
            endpoint_name=endpoint_name,
            model=model,
            environment=env,
            code_configuration=CodeConfiguration(
                code="deployment/",
                scoring_script="score_forecast.py",
            ),
            instance_type=instance_type,
            instance_count=instance_count,
            liveness_probe=ProbeSettings(initial_delay=600),
            readiness_probe=ProbeSettings(initial_delay=600),
        )
        
        result = self.ml_client.online_deployments.begin_create_or_update(deployment).result()
        
        endpoint.traffic = {"forecast-blue": 100}
        self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        
        logger.info(f"Forecasting model deployed: {endpoint.scoring_uri}")
        return result
    
    def test_endpoint(self, endpoint_name: str, sample_data: dict) -> dict:
        """Test an endpoint with sample data."""
        import json
        
        response = self.ml_client.online_endpoints.invoke(
            endpoint_name=endpoint_name,
            request_file=None,
            request=json.dumps(sample_data),
        )
        
        return json.loads(response)
    
    def list_endpoints(self):
        """List all online endpoints in the workspace."""
        endpoints = self.ml_client.online_endpoints.list()
        
        print(f"\n{'Name':<35} {'State':<15} {'URI'}")
        print("-" * 100)
        for ep in endpoints:
            print(f"{ep.name:<35} {ep.provisioning_state:<15} {ep.scoring_uri or 'N/A'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy SGT400 ML models")
    parser.add_argument("--action", choices=["deploy-anomaly", "deploy-forecast", "deploy-all", "list"],
                       default="list")
    parser.add_argument("--subscription-id", type=str)
    parser.add_argument("--resource-group", type=str)
    parser.add_argument("--workspace", type=str)
    args = parser.parse_args()
    
    deployer = ModelDeployer(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace_name=args.workspace,
    )
    
    if args.action == "list":
        deployer.list_endpoints()
    elif args.action == "deploy-anomaly":
        deployer.deploy_anomaly_model()
    elif args.action == "deploy-forecast":
        deployer.deploy_forecasting_model()
    elif args.action == "deploy-all":
        deployer.deploy_anomaly_model()
        deployer.deploy_forecasting_model()

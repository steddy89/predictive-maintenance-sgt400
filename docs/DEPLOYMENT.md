# SGT400 Predictive Maintenance - Deployment Guide

## Prerequisites

| Requirement                | Version / Detail                             |
|----------------------------|----------------------------------------------|
| Azure Subscription         | With Contributor access                      |
| Azure CLI                  | >= 2.55                                      |
| Docker Desktop             | >= 24.0                                      |
| Node.js                    | >= 20 LTS                                    |
| Python                     | >= 3.12                                      |
| Microsoft Fabric           | Workspace with Lakehouse enabled             |
| GitHub Account             | For CI/CD (Actions)                          |

---

## 1. Local Development Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/<org>/predictive-maintenance-sgt400.git
cd predictive-maintenance-sgt400
```

### 1.2 Environment Variables

Copy the template and fill in values:

```bash
cp .env.template .env
```

Key variables:

| Variable                     | Description                                  |
|------------------------------|----------------------------------------------|
| `AZURE_ML_ANOMALY_ENDPOINT`  | Anomaly detection ML endpoint URL            |
| `AZURE_ML_FORECAST_ENDPOINT` | Forecast BiLSTM ML endpoint URL              |
| `AZURE_ML_API_KEY`           | ML endpoint key (dev only, use MI in prod)   |
| `FABRIC_WORKSPACE_ID`        | Microsoft Fabric workspace GUID              |
| `FABRIC_LAKEHOUSE_ID`        | Fabric Lakehouse GUID                        |
| `KEY_VAULT_URL`              | Azure Key Vault URL                          |
| `APPINSIGHTS_CONNECTION_STRING` | Application Insights connection string    |

### 1.3 Run with Docker Compose (Recommended)

```bash
docker compose up --build
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API docs: http://localhost:8000/docs

### 1.4 Run Backend Standalone

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 1.5 Run Frontend Standalone

```bash
cd frontend
npm install
npm run dev
```

Frontend dev server starts at http://localhost:5173 with hot module replacement.

### 1.6 Generate Sample Data

```bash
python scripts/generate_sample_data.py --days 30 --output data-layer/sample_output
```

---

## 2. Azure Infrastructure Deployment

### 2.1 Deploy with Bicep

```bash
az login
az account set --subscription <SUBSCRIPTION_ID>

# Create resource group
az group create --name rg-sgt400-prod --location eastus2

# Deploy infrastructure
az deployment group create \
  --resource-group rg-sgt400-prod \
  --template-file infra/main.bicep \
  --parameters infra/main.bicepparam
```

This creates all resources defined in the architecture:
- Log Analytics + Application Insights
- Key Vault (Standard)
- Container Registry (Basic)
- App Service Plan (PremiumV3 / P1v3)
- Backend + Frontend App Services
- Storage Account
- Azure ML Workspace
- User-Assigned Managed Identity with RBAC

### 2.2 Store Secrets in Key Vault

```bash
KV_NAME="kv-sgt400-prod"

az keyvault secret set --vault-name $KV_NAME \
  --name "AzureMLAnomalyEndpoint" \
  --value "<ML_ANOMALY_ENDPOINT>"

az keyvault secret set --vault-name $KV_NAME \
  --name "AzureMLForecastEndpoint" \
  --value "<ML_FORECAST_ENDPOINT>"

az keyvault secret set --vault-name $KV_NAME \
  --name "AzureMLApiKey" \
  --value "<ML_API_KEY>"
```

---

## 3. Microsoft Fabric Setup

### 3.1 Create Lakehouse

1. Open Microsoft Fabric portal (https://app.fabric.microsoft.com)
2. Navigate to your workspace
3. Create a new **Lakehouse** named `sgt400_lakehouse`
4. Note the Workspace ID and Lakehouse ID

### 3.2 Upload Notebooks

Upload the three PySpark notebooks into the Fabric workspace:

```
data-layer/notebooks/01_bronze_ingestion.py
data-layer/notebooks/02_silver_transform.py
data-layer/notebooks/03_gold_aggregation.py
```

### 3.3 Configure Pipeline

Use the configuration in `data-layer/pipelines/fabric_pipeline.json` to set up
a **Fabric Data Pipeline** with three sequential activities:

1. **Bronze Ingestion** — runs `01_bronze_ingestion` notebook
2. **Silver Transform** — runs `02_silver_transform` notebook (depends on Bronze)
3. **Gold Aggregation** — runs `03_gold_aggregation` notebook (depends on Silver)

Schedule: hourly (cron `0 * * * *`)

### 3.4 Initial Data Load

Upload sample data generated in step 1.6 to the Lakehouse `Files/raw/sensor_data` path.

---

## 4. ML Model Deployment (Azure AI Foundry)

### 4.1 Train Models

```bash
cd ml-layer
python training/train_pipeline.py \
  --data-source local \
  --data-path ../data-layer/sample_output \
  --experiment-name sgt400-training-v1 \
  --register
```

### 4.2 Deploy to Managed Endpoints

```bash
python deployment/deploy_models.py
```

This will:
1. Create managed online endpoints (`sgt400-anomaly-endpoint`, `sgt400-forecast-endpoint`)
2. Deploy the latest registered model versions
3. Configure auto-scaling (1-3 instances)

### 4.3 Verify Endpoints

```bash
# Test anomaly endpoint
az ml online-endpoint invoke \
  --name sgt400-anomaly-endpoint \
  --request-file deployment/sample_request_anomaly.json

# Test forecast endpoint
az ml online-endpoint invoke \
  --name sgt400-forecast-endpoint \
  --request-file deployment/sample_request_forecast.json
```

---

## 5. CI/CD Pipeline (GitHub Actions)

### 5.1 Configure GitHub Secrets

| Secret                           | Value                                 |
|----------------------------------|---------------------------------------|
| `AZURE_CLIENT_ID`               | Managed Identity / Service Principal  |
| `AZURE_TENANT_ID`               | Azure AD Tenant ID                    |
| `AZURE_SUBSCRIPTION_ID`         | Subscription ID                       |
| `ACR_LOGIN_SERVER`              | `<name>.azurecr.io`                   |
| `BACKEND_APP_NAME`              | App Service name (backend)            |
| `FRONTEND_APP_NAME`             | App Service name (frontend)           |

### 5.2 Pipeline Stages

The CI/CD pipeline (`.github/workflows/ci-cd.yml`) runs on pushes to `main`:

```
lint → test-backend → test-frontend → build-push → deploy
```

1. **lint**: Ruff (Python) + ESLint (TypeScript)
2. **test-backend**: pytest with coverage
3. **test-frontend**: Vitest
4. **build-push**: Docker build → push to ACR
5. **deploy**: Update App Service container images

### 5.3 OIDC Authentication

The pipeline uses OpenID Connect (OIDC) for passwordless Azure authentication.
Configure the federated credential:

```bash
az ad app federated-credential create \
  --id <APP_REGISTRATION_ID> \
  --parameters '{
    "name": "github-actions",
    "issuer": "https://token.actions.githubusercontent.com",
    "subject": "repo:<org>/<repo>:ref:refs/heads/main",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

---

## 6. Monitoring & Operations

### 6.1 Application Insights

- Navigate to the Application Insights resource in Azure Portal
- **Live Metrics**: Real-time request/dependency monitoring
- **Application Map**: Service dependency visualization
- **Failures**: Exception tracking and analysis

### 6.2 Alert Rules (Azure Monitor)

Set up alert rules for:

| Metric                         | Condition               | Action              |
|--------------------------------|-------------------------|---------------------|
| Backend response time          | P95 > 2s               | Email + Teams       |
| Backend error rate             | > 5% in 5 min          | Email + PagerDuty   |
| ML endpoint latency            | P95 > 5s               | Email               |
| Container memory               | > 85%                  | Auto-scale          |

### 6.3 ML Model Monitoring

Run drift detection periodically:

```bash
python ml-layer/monitoring/drift_detection.py \
  --baseline-path data/baseline_stats.json \
  --current-data-path data/latest_readings.csv
```

Investigate when:
- KS-test p-value < 0.05 for any feature
- PSI > 0.2 for any feature
- Anomaly model false-positive rate increases

---

## 7. Scaling & Production Hardening

### 7.1 Horizontal Scaling

| Component      | Scaling Strategy                              |
|----------------|-----------------------------------------------|
| Backend API    | App Service auto-scale (CPU > 70%)            |
| ML Endpoints   | AzureML managed auto-scale (1-5 instances)    |
| Fabric Spark   | Dynamic executor allocation (1-8)             |
| Frontend       | CDN + static hosting (Azure Static Web Apps)  |

### 7.2 High Availability

- App Service: minimum 2 instances in production
- ML Endpoints: minimum 2 instances with zone redundancy
- Fabric: built-in HA within the Fabric capacity

### 7.3 Backup & Recovery

- Delta tables: time-travel for point-in-time recovery (90-day retention)
- Key Vault: soft-delete enabled (90-day retention)
- MLflow: model versioning with full artifact history
- Infrastructure: Bicep templates in source control = infrastructure as code

---

## Troubleshooting

| Issue                              | Solution                                  |
|------------------------------------|-------------------------------------------|
| ML endpoint returns 401            | Verify API key / Managed Identity RBAC    |
| Fabric notebook timeout            | Increase executor count or node size      |
| Frontend shows "Disconnected"      | Check backend health at `/api/v1/health`  |
| Docker build fails                 | Verify `requirements.txt` / `package.json`|
| Drift detected                     | Retrain models with latest data           |
| High anomaly false-positive rate   | Adjust contamination parameter            |

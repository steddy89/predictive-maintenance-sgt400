# SGT-400 Predictive Maintenance System

Production-ready predictive maintenance application for a **Siemens SGT-400 Industrial Gas Turbine** used in power-plant operations. Built on **Microsoft Fabric**, **Azure AI Foundry**, **FastAPI**, and **React**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AZURE CLOUD INFRASTRUCTURE                       │
│  (Bicep IaC: Key Vault, ACR, App Insights, Managed Identity, RBAC)     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────────── DATA LAYER (Microsoft Fabric) ─────────────────┐ │
│  │                                                                     │ │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐                      │ │
│  │  │  BRONZE   │───▶│  SILVER   │───▶│   GOLD   │                     │ │
│  │  │ Raw IoT   │    │ Cleaned + │    │ KPIs +   │                     │ │
│  │  │ Ingestion │    │ Features  │    │ Health   │                     │ │
│  │  │ (Delta)   │    │ (Delta)   │    │ Scores   │                     │ │
│  │  └──────────┘    └──────────┘    └──────────┘                      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────── ML LAYER (Azure AI Foundry) ───────────────────────┐ │
│  │                                                                     │ │
│  │  ┌────────────────┐  ┌────────────────┐  ┌──────────────────┐      │ │
│  │  │ Isolation      │  │ BiLSTM         │  │ Exponential      │      │ │
│  │  │ Forest         │  │ Forecaster     │  │ Degradation      │      │ │
│  │  │ (Anomaly)      │  │ (Time-Series)  │  │ (RUL Estimator)  │      │ │
│  │  └────────────────┘  └────────────────┘  └──────────────────┘      │ │
│  │         │                    │                     │                │ │
│  │         └────────────┬───────┘─────────────────────┘                │ │
│  │                      ▼                                              │ │
│  │            Managed Online Endpoints                                 │ │
│  │            (MLflow + Azure ML SDK)                                  │ │
│  │                                                                     │ │
│  │  ┌──────────────────┐                                               │ │
│  │  │ Drift Detection  │  (KS-test, PSI monitoring)                   │ │
│  │  └──────────────────┘                                               │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────── BACKEND (FastAPI) ─────────────────────────────────┐ │
│  │  /api/turbine/*   /api/predictions/*   /api/alerts/*   /health     │ │
│  │  Pydantic schemas │ ML client service │ Fabric client │ Logging    │ │
│  │  Docker container │ App Service (Linux) │ App Insights integration │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────── ALERTING ENGINE ───────────────────────────────────┐ │
│  │  Threshold rules │ ML score rules │ Cooldown │ Teams webhooks      │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────── FRONTEND (React + Vite) ───────────────────────────┐ │
│  │  KPI Panel │ Trend Charts (Recharts) │ Alerts │ Maintenance Recs   │ │
│  │  Tailwind CSS │ TypeScript │ Polling │ nginx container             │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌──────────────── DEVOPS (GitHub Actions) ───────────────────────────┐ │
│  │  Lint → Test → Build → Push ACR → Deploy App Service               │ │
│  │  OIDC auth │ Multi-stage Docker │ Health-check verification        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
predictive-maintenance-sgt400/
├── .github/workflows/
│   └── ci-cd.yml                  # GitHub Actions pipeline
├── infra/
│   ├── main.bicep                 # Azure IaC (Key Vault, ACR, App Service, ML…)
│   └── main.bicepparam            # Parameter file
├── data-layer/
│   ├── notebooks/
│   │   ├── 01_bronze_ingestion.py # Raw IoT → Bronze Delta tables
│   │   ├── 02_silver_transform.py # Cleaning, feature engineering
│   │   └── 03_gold_aggregation.py # KPIs, health scores, ML datasets
│   ├── pipelines/
│   │   ├── fabric_pipeline.json   # Pipeline config (schedule, quality gates)
│   │   └── pipeline_orchestrator.py # Fabric REST API orchestrator
│   └── schemas/
│       └── lakehouse_config.json
├── ml-layer/
│   ├── models/
│   │   ├── anomaly_detection.py   # Isolation Forest + MLflow
│   │   └── forecasting_rul.py     # BiLSTM forecaster + RUL estimator
│   ├── deployment/
│   │   ├── deploy_models.py       # Azure AI Foundry deployment
│   │   ├── score_anomaly.py       # Anomaly scoring entry point
│   │   ├── score_forecast.py      # Forecast scoring entry point
│   │   ├── conda_anomaly.yml      # Anomaly model conda environment
│   │   └── conda_forecast.yml     # Forecast model conda environment
│   ├── training/
│   │   └── train_pipeline.py      # End-to-end training orchestration
│   └── monitoring/
│       └── drift_detection.py     # KS-test, PSI drift monitoring
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI application
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings
│   │   │   └── logging_config.py  # Structured logging + App Insights
│   │   ├── models/
│   │   │   └── schemas.py         # Request / response models
│   │   ├── services/
│   │   │   ├── ml_client.py       # Azure ML endpoint client
│   │   │   └── fabric_client.py   # Fabric Lakehouse data service
│   │   └── api/
│   │       ├── health.py          # /health
│   │       ├── turbine.py         # /api/turbine/*
│   │       ├── predictions.py     # /api/predictions/*
│   │       └── alerts.py          # /api/alerts/*
│   ├── tests/
│   │   └── test_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                # Root component
│   │   ├── main.tsx               # Entry point
│   │   ├── index.css              # Tailwind + globals
│   │   ├── types/api.ts           # TypeScript interfaces
│   │   ├── services/api.ts        # Axios API layer
│   │   ├── hooks/usePolling.ts    # Polling custom hook
│   │   ├── utils/
│   │   │   └── formatters.ts     # Sensor value & UI formatting
│   │   └── components/
│   │       ├── Header.tsx
│   │       ├── KPIPanel.tsx
│   │       ├── TrendCharts.tsx
│   │       ├── AlertPanel.tsx
│   │       └── MaintenancePanel.tsx
│   ├── Dockerfile                 # Multi-stage nginx build
│   ├── nginx.conf                 # SPA routing + API proxy
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
├── alerting/
│   ├── __init__.py
│   └── alert_engine.py            # Threshold + ML-score rules, webhooks
├── scripts/
│   └── generate_sample_data.py    # Synthetic SGT-400 data generator
├── docs/
│   ├── ARCHITECTURE.md            # Detailed system architecture
│   └── DEPLOYMENT.md              # Step-by-step deployment guide
├── docker-compose.yml             # Local dev stack
├── .env.template                  # Environment variable template
├── .gitignore
└── README.md                      # ← You are here
```

---

## Key Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Infrastructure** | Azure Bicep | Declarative IaC for all Azure resources |
| **Data** | Microsoft Fabric Lakehouse | Bronze/Silver/Gold medallion architecture with PySpark & Delta |
| **ML** | Azure AI Foundry (Azure ML) | Model training, MLflow registry, managed online endpoints |
| **Backend** | FastAPI (Python 3.12) | REST API with async endpoints, Pydantic validation |
| **Frontend** | React 18 + Vite + TypeScript | Real-time dashboard with Recharts, Tailwind CSS |
| **Alerting** | Custom Python engine | Threshold & ML-score rules, Teams/Slack webhooks |
| **DevOps** | GitHub Actions | CI/CD: lint → test → build → push ACR → deploy App Service |
| **Security** | Azure Key Vault + Managed Identity | RBAC-based secret management, no stored credentials |
| **Monitoring** | Application Insights | Structured logging, distributed tracing |

---

## Microsoft Learn References

- [Azure ML Managed Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online)
- [Microsoft Fabric Lakehouse](https://learn.microsoft.com/en-us/fabric/data-engineering/lakehouse-overview)
- [Medallion Architecture](https://learn.microsoft.com/en-us/fabric/onelake/onelake-medallion-lakehouse-architecture)
- [Azure Bicep Documentation](https://learn.microsoft.com/en-us/azure/azure-resource-manager/bicep/)
- [Azure App Service Container Deployment](https://learn.microsoft.com/en-us/azure/app-service/deploy-ci-cd-custom-container)
- [Azure Key Vault RBAC](https://learn.microsoft.com/en-us/azure/key-vault/general/rbac-guide)
- [Azure Monitor Alerts Overview](https://learn.microsoft.com/en-us/azure/azure-monitor/alerts/alerts-overview)
- [MLflow on Azure ML](https://learn.microsoft.com/en-us/azure/machine-learning/concept-mlflow)

---

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 20+
- Docker & Docker Compose
- Azure CLI (`az`) with an active subscription
- Microsoft Fabric workspace (for data layer)

### 1. Clone & configure

```bash
git clone <repo-url>
cd predictive-maintenance-sgt400
cp .env.template .env
# Edit .env with your Azure resource values
```

### 2. Generate sample data

```bash
pip install pandas numpy pyarrow
python scripts/generate_sample_data.py --output-dir ./sample_data --days 30
```

### 3. Run locally with Docker Compose

```bash
docker compose up --build
```

- **Backend API**: http://localhost:8000/docs (Swagger UI)
- **Frontend Dashboard**: http://localhost:3000

### 4. Deploy infrastructure

```bash
az deployment group create \
  --resource-group rg-predictive-maintenance \
  --template-file infra/main.bicep \
  --parameters infra/main.bicepparam
```

### 5. Train & deploy ML models

```bash
cd ml-layer
pip install -r ../backend/requirements.txt
python models/anomaly_detection.py        # Train anomaly model
python models/forecasting_rul.py          # Train forecaster
python deployment/deploy_models.py \
  --subscription-id <sub> \
  --resource-group <rg> \
  --workspace-name <ws>
```

---

## Sensor Coverage (SGT-400)

| Sensor | Unit | Normal Range | Warning | Critical |
|--------|------|-------------|---------|----------|
| Exhaust Gas Temperature | °C | 480–530 | >540 | >560 |
| Vibration | mm/s | 2–7 | >10 | >15 |
| Compressor Inlet Temp | °C | 15–35 | — | — |
| Compressor Discharge Pressure | bar | 14–17 | — | — |
| Bearing Temperature | °C | 60–95 | >110 | >120 |
| Lube Oil Pressure | bar | 2.5–4.0 | <2.0 | <1.5 |
| Power Output | MW | 10–13 | <11 | <9 |
| Fuel Flow Rate | kg/s | 0.7–1.0 | — | — |
| Turbine Speed | RPM | 9,200–9,600 | — | — |
| Ambient Temperature | °C | –10–45 | — | — |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health check |
| GET | `/api/turbine/status` | Current turbine status + health score |
| GET | `/api/turbine/readings` | Recent sensor readings |
| GET | `/api/turbine/trend/{sensor}` | Historical trend for a sensor |
| POST | `/api/predictions/anomaly` | Run anomaly detection |
| POST | `/api/predictions/failure` | Failure probability + RUL |
| POST | `/api/predictions/forecast` | Sensor value forecasting |
| GET | `/api/alerts/` | Active alerts list |
| POST | `/api/alerts/acknowledge` | Acknowledge an alert |
| GET | `/api/alerts/summary` | Alert statistics summary |

---

## License

Internal use – Microsoft confidential.

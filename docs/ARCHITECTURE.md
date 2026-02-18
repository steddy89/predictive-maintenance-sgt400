# SGT400 Predictive Maintenance - System Architecture

## Overview

The Predictive Maintenance platform for Siemens SGT-400 Industrial Gas Turbines provides
real-time health monitoring, anomaly detection, failure forecasting, and remaining useful life
(RUL) estimation. It is built on a modern cloud-native stack centered on **Microsoft Fabric**
and **Azure AI Foundry**.

---

## High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  IoT Edge / Historian (OPC-UA)                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                                   │
│  │ Vibration │  │  Temps   │  │ Pressure │  … 14 sensor channels             │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘                                   │
│        └──────────────┼──────────────┘                                       │
│                       ▼                                                       │
│               Event Hub / IoT Hub                                            │
└───────────────────────┬──────────────────────────────────────────────────────┘
                        │
          ┌─────────────▼─────────────┐
          │   Microsoft Fabric        │
          │   Lakehouse               │
          │                           │
          │  ┌─────────────────────┐  │
          │  │  Bronze (Raw)       │  │   Raw Parquet / streaming ingestion
          │  │  Delta Tables       │  │
          │  └────────┬────────────┘  │
          │           ▼               │
          │  ┌─────────────────────┐  │
          │  │  Silver (Cleaned)   │  │   Dedup, impute, rolling features,
          │  │  Feature Store      │  │   z-score flags, 50+ features
          │  └────────┬────────────┘  │
          │           ▼               │
          │  ┌─────────────────────┐  │
          │  │  Gold (KPIs)        │  │   Hourly aggregates, health scores,
          │  │  Model-ready data   │  │   alert history, ML datasets
          │  └────────┬────────────┘  │
          └───────────┼───────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  ┌───────────┐ ┌───────────┐ ┌───────────┐
  │  Anomaly  │ │ BiLSTM    │ │   RUL     │
  │ Detection │ │ Forecast  │ │ Estimator │   Azure AI Foundry
  │ (IsoFor.) │ │ (24h-out) │ │ (Exp.Deg) │   Managed Endpoints
  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
        └──────────────┼─────────────┘
                       ▼
                ┌──────────────┐
                │  FastAPI     │         Backend API
                │  Backend     │         (Python 3.12)
                │  ┌────────┐  │
                │  │ML Client│ │─── httpx async calls to ML endpoints
                │  │Fabric   │ │─── Fabric Lakehouse reads
                │  │Client   │ │
                │  └────────┘  │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │  React 18    │         Frontend Dashboard
                │  TypeScript  │         (Vite + Tailwind + Recharts)
                │  SPA         │
                └──────────────┘
```

---

## Component Details

### 1. Data Layer — Microsoft Fabric Lakehouse

| Layer    | Table / Path                     | Format     | Retention |
|----------|----------------------------------|------------|-----------|
| Bronze   | `bronze_sensor_readings`         | Delta Lake | 90 days   |
| Bronze   | `bronze_alarm_logs`              | Delta Lake | 1 year    |
| Silver   | `silver_sensor_features`         | Delta Lake | 1 year    |
| Gold     | `gold_turbine_hourly_kpis`       | Delta Lake | 2 years   |
| Gold     | `gold_turbine_latest_status`     | Delta Lake | Current   |
| Gold     | `gold_alert_history`             | Delta Lake | 2 years   |
| Gold     | `gold_ml_training_data`          | Delta Lake | Versioned |

**Processing Pipeline:**
- **Bronze**: Auto Loader (Structured Streaming) or batch Parquet ingestion
- **Silver**: Deduplication, range validation, forward-fill imputation,
  rolling statistics (1h / 12h windows), rate of change, cross-sensor ratios
- **Gold**: Hourly KPIs, 4-component health score, alert aggregation, feature-complete ML datasets

### 2. ML Layer — Azure AI Foundry

| Model               | Algorithm         | Input               | Output                | Endpoint          |
|---------------------|-------------------|----------------------|-----------------------|-------------------|
| Anomaly Detection   | Isolation Forest  | 14 sensor features   | Score (0-1) + labels  | `anomaly-detect`  |
| Failure Forecast    | BiLSTM (Keras)    | 144-step sequences   | 24h sensor forecast   | `forecast-bilstm` |
| RUL Estimator       | Exponential Decay  | Sensor degradation   | Days remaining        | (backend inline)  |

**MLflow tracking** records all experiments, metrics, and model artifacts.

### 3. Backend — FastAPI

```
backend/app/
├── main.py                 # App factory, lifespan, middleware
├── core/
│   ├── config.py           # Pydantic Settings (env-driven)
│   └── logging_config.py   # Structured logging + App Insights
├── api/
│   ├── health.py           # GET /health
│   ├── turbine.py          # GET /status, /readings, /trend/{sensor}
│   ├── predictions.py      # GET /anomaly, /failure, /forecast
│   └── alerts.py           # GET /, POST /acknowledge, GET /summary
├── models/
│   └── schemas.py          # Pydantic v2 request/response models
└── services/
    ├── ml_client.py        # Async httpx client for ML endpoints
    └── fabric_client.py    # Fabric Lakehouse data service
```

**Key design decisions:**
- Async throughout (httpx, asyncio)
- Local fallback mode with synthetic data for development
- TrustedHostMiddleware + CORS configuration
- Health endpoint checks downstream service connectivity

### 4. Frontend — React 18 + TypeScript

```
frontend/src/
├── App.tsx                 # Root layout, polling orchestration
├── types/api.ts            # TypeScript interfaces mirroring backend schemas
├── components/
│   ├── Header.tsx          # Branding, risk badge, connection indicator
│   ├── KPIPanel.tsx        # Health score, anomaly score, RUL, risk level
│   ├── TrendCharts.tsx     # Multi-sensor trend visualization (Recharts)
│   ├── AlertPanel.tsx      # Active alert list with acknowledge actions
│   └── MaintenancePanel.tsx# Failure predictions, recommendations, sensors
├── services/api.ts         # Axios HTTP client for backend API
├── hooks/usePolling.ts     # Generic polling hook with error handling
└── utils/formatters.ts     # Sensor formatting, risk color mapping
```

**Polling strategy:**
- KPIs & anomaly: 15-second interval
- Trend data: 30-second interval
- Alerts: 20-second interval

### 5. Alert Engine

Seven default rules covering temperature, vibration, pressure, load, and
efficiency thresholds. Each rule has a configurable cooldown period.

Notification channels: Microsoft Teams (Adaptive Cards via webhook).

### 6. Infrastructure — Azure (Bicep IaC)

| Resource                   | SKU / Tier     | Purpose                          |
|----------------------------|----------------|----------------------------------|
| Log Analytics Workspace    | PerGB2018      | Central logging                  |
| Application Insights       | —              | APM, request tracing             |
| User-Assigned Managed ID   | —              | Passwordless RBAC                |
| Key Vault                  | Standard       | Secrets management               |
| Container Registry         | Basic          | Docker image storage             |
| App Service Plan           | PremiumV3 (P1) | Compute for web apps             |
| App Service (Backend)      | Linux / Docker | FastAPI container                |
| App Service (Frontend)     | Linux / Docker | Nginx + React SPA                |
| Storage Account            | Standard LRS   | ML artifacts, data staging       |
| ML Workspace               | —              | Azure AI Foundry workspace       |

RBAC assignments:
- `AcrPull` on Container Registry
- `Key Vault Secrets User` on Key Vault
- `Storage Blob Data Contributor` on Storage

---

## Security Architecture

| Concern               | Implementation                                      |
|-----------------------|-----------------------------------------------------|
| Authentication        | Managed Identity (DefaultAzureCredential)           |
| Secret management     | Azure Key Vault (no secrets in code or env vars)    |
| Network               | TrustedHostMiddleware, CORS allow-list              |
| API security          | HTTPS-only, App Service built-in auth               |
| Data encryption       | At-rest (Azure Storage), in-transit (TLS 1.2+)      |
| Supply chain          | Ruff + Dependabot + image scanning in CI            |

---

## Data Flow

```
1.  Sensors → Event Hub → Fabric (Bronze)       [5-min intervals]
2.  Bronze → Silver (PySpark cleaning)           [hourly pipeline]
3.  Silver → Gold (KPIs, health scores)          [hourly pipeline]
4.  Gold → ML Training (on-demand)               [weekly / on-drift]
5.  Trained models → Azure ML Endpoints          [CI/CD deployment]
6.  Dashboard request → FastAPI → ML + Fabric    [real-time]
7.  Alert Engine evaluates rules → Teams webhook  [on each poll]
```

## Monitoring & Observability

- **Application Insights** for request tracing and dependency tracking
- **Structured JSON logging** with correlation IDs
- **ML model drift detection** with KS-test and PSI metrics
- **Data quality gates** at each medallion layer
- **Alert SLA monitoring** for pipeline execution times

---

## Sensor Reference

| Sensor                         | Unit   | Normal Range     | Critical Threshold |
|--------------------------------|--------|------------------|--------------------|
| Turbine Inlet Temperature      | °C     | 900 – 1100       | > 1150             |
| Exhaust Temperature            | °C     | 450 – 550        | > 600              |
| Compressor Discharge Pressure  | bar    | 14 – 18          | > 20               |
| Vibration (Bearing DE)         | mm/s   | 0.5 – 3.0        | > 7.0              |
| Vibration (Bearing NDE)        | mm/s   | 0.5 – 3.0        | > 7.0              |
| Lube Oil Temperature           | °C     | 45 – 65          | > 80               |
| Lube Oil Pressure              | bar    | 1.5 – 3.0        | < 1.0              |
| Fuel Flow Rate                 | kg/s   | 2.0 – 5.0        | > 6.0              |
| Turbine Speed (Shaft)          | RPM    | 9000 – 11000     | > 11500            |
| Power Output                   | MW     | 8 – 13           | < 5                |
| Compressor Efficiency          | %      | 82 – 90          | < 75               |
| Turbine Efficiency             | %      | 85 – 93          | < 78               |
| NOx Emissions                  | ppm    | 10 – 25          | > 40               |
| CO Emissions                   | ppm    | 2 – 10           | > 20               |

import axios from 'axios';
import type {
  TurbineStatus,
  AnomalyResult,
  FailurePrediction,
  Alert,
  TrendPoint,
  SensorForecast,
} from '../types/api';

const API_BASE = import.meta.env.VITE_API_URL || '/api/v1';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ---- Turbine ----
export async function fetchTurbineStatus(turbineId = 'SGT400-001'): Promise<TurbineStatus> {
  const { data } = await api.get(`/turbine/status`, { params: { turbine_id: turbineId } });
  return data.data;
}

export async function fetchSensorTrend(
  sensor: string,
  turbineId = 'SGT400-001',
  hours = 24
): Promise<TrendPoint[]> {
  const { data } = await api.get(`/turbine/trend/${sensor}`, {
    params: { turbine_id: turbineId, hours },
  });
  return data.data;
}

// ---- Predictions ----
export async function fetchAnomalyScore(turbineId = 'SGT400-001'): Promise<{
  result: AnomalyResult;
  threshold: number;
}> {
  const { data } = await api.get(`/predictions/anomaly`, {
    params: { turbine_id: turbineId },
  });
  return { result: data.result, threshold: data.threshold };
}

export async function fetchFailurePrediction(turbineId = 'SGT400-001'): Promise<FailurePrediction> {
  const { data } = await api.get(`/predictions/failure`, {
    params: { turbine_id: turbineId },
  });
  return data.prediction;
}

export async function fetchForecast(
  turbineId = 'SGT400-001',
  horizonMinutes = 60
): Promise<SensorForecast[]> {
  const { data } = await api.get(`/predictions/forecast`, {
    params: { turbine_id: turbineId, horizon_minutes: horizonMinutes },
  });
  return data.sensors;
}

// ---- Alerts ----
export async function fetchAlerts(
  turbineId = 'SGT400-001',
  hours = 168,
  activeOnly = false
): Promise<Alert[]> {
  const { data } = await api.get(`/alerts/`, {
    params: { turbine_id: turbineId, hours, active_only: activeOnly },
  });
  return data.alerts;
}

export async function fetchAlertSummary(turbineId = 'SGT400-001'): Promise<{
  total_active: number;
  critical: number;
  warning: number;
  info: number;
}> {
  const { data } = await api.get(`/alerts/summary`, {
    params: { turbine_id: turbineId },
  });
  return data.summary;
}

export async function acknowledgeAlert(alertId: string, user: string): Promise<void> {
  await api.post(`/alerts/acknowledge`, {
    alert_id: alertId,
    acknowledged_by: user,
  });
}

// ---- Health ----
export async function fetchHealth(): Promise<{
  status: string;
  version: string;
  environment: string;
  services: Record<string, string>;
}> {
  const { data } = await axios.get(
    `${import.meta.env.VITE_API_URL || ''}/health`
  );
  return data;
}

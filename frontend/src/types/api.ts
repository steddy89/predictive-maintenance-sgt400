/** API types matching the FastAPI backend schemas */

export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type AlertSeverity = 'INFO' | 'WARNING' | 'CRITICAL';

export interface TurbineStatus {
  turbine_id: string;
  timestamp: string;
  health_score: number;
  risk_level: RiskLevel;
  exhaust_gas_temp_c: number;
  vibration_mm_s: number;
  discharge_pressure_bar: number;
  turbine_load_mw: number;
  efficiency_pct: number;
  pressure_ratio: number;
  active_alerts: number;
  last_maintenance?: string;
}

export interface AnomalyResult {
  is_anomaly: boolean;
  anomaly_score: number;
  raw_score: number;
  contributing_sensors: string[];
  top_contributing_sensors?: { sensor: string; contribution: number }[];
}

export interface FailurePrediction {
  failure_probability: number;
  rul_days: number | null;
  estimated_rul_hours?: number | null;
  degradation_index: number;
  degradation_rate: number | null;
  confidence: string | number;
  recommendation: string;
  recommendations?: string[];
  risk_level: RiskLevel;
}

export interface Alert {
  id: string;
  alert_id?: string;
  timestamp: string;
  turbine_id: string;
  alarm_code: string;
  description: string;
  message?: string;
  severity: AlertSeverity;
  sensor_name?: string;
  sensor_value: number | null;
  value?: number;
  threshold: number | null;
  acknowledged: boolean;
  resolved: boolean;
  resolved_at?: string;
}

export interface TrendPoint {
  timestamp: string;
  value: number;
}

export interface ForecastPoint {
  timestamp: string;
  value: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface SensorForecast {
  sensor: string;
  forecast: ForecastPoint[];
}

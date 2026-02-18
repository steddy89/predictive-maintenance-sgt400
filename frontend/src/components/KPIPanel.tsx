import {
  Gauge,
  ShieldAlert,
  AlertTriangle,
  Thermometer,
  Clock,
} from 'lucide-react';
import type { TurbineStatus, AnomalyResult, FailurePrediction } from '../types/api';

/* ---------- helpers ---------- */

function riskColor(level: string) {
  switch (level) {
    case 'LOW':
      return 'text-turbine-green';
    case 'MEDIUM':
      return 'text-turbine-yellow';
    case 'HIGH':
      return 'text-turbine-orange';
    case 'CRITICAL':
      return 'text-turbine-red';
    default:
      return 'text-gray-400';
  }
}

function scoreBar(value: number, max = 1, color: string) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="w-full h-2 bg-gray-700 rounded-full mt-2">
      <div
        className={`h-2 rounded-full ${color}`}
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

/* ---------- Card wrapper ---------- */

interface CardProps {
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  className?: string;
}

function Card({ title, icon, children, className = '' }: CardProps) {
  return (
    <div
      className={`bg-gray-900 border border-gray-800 rounded-xl p-5 ${className}`}
    >
      <div className="flex items-center gap-2 text-gray-400 text-xs uppercase tracking-wider mb-3">
        {icon}
        {title}
      </div>
      {children}
    </div>
  );
}

/* ---------- KPI Panel ---------- */

interface KPIPanelProps {
  status: TurbineStatus | null;
  anomaly: AnomalyResult | null;
  failure: FailurePrediction | null;
  alertCount: number;
  loading: boolean;
}

export function KPIPanel({
  status,
  anomaly,
  failure,
  alertCount,
  loading,
}: KPIPanelProps) {
  if (loading && !status) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div
            key={i}
            className="bg-gray-900 border border-gray-800 rounded-xl p-5 h-32 animate-pulse"
          />
        ))}
      </div>
    );
  }

  const healthScore = status?.health_score ?? 0;
  const anomalyScore = anomaly?.anomaly_score ?? 0;
  const failureProb = failure?.failure_probability ?? 0;
  const rul = failure?.estimated_rul_hours ?? null;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
      {/* Health Score */}
      <Card
        title="Health Score"
        icon={<Gauge className="h-4 w-4" />}
      >
        <p className={`text-3xl font-bold ${riskColor(status?.risk_level ?? 'LOW')}`}>
          {(healthScore * 100).toFixed(1)}
          <span className="text-lg font-normal text-gray-500">%</span>
        </p>
        {scoreBar(healthScore, 1, 'bg-turbine-green')}
      </Card>

      {/* Anomaly Score */}
      <Card
        title="Anomaly Score"
        icon={<ShieldAlert className="h-4 w-4" />}
      >
        <p
          className={`text-3xl font-bold ${
            anomalyScore > 0.7
              ? 'text-turbine-red'
              : anomalyScore > 0.4
                ? 'text-turbine-yellow'
                : 'text-turbine-green'
          }`}
        >
          {anomalyScore.toFixed(3)}
        </p>
        {scoreBar(
          anomalyScore,
          1,
          anomalyScore > 0.7
            ? 'bg-turbine-red'
            : anomalyScore > 0.4
              ? 'bg-turbine-yellow'
              : 'bg-turbine-green',
        )}
      </Card>

      {/* Failure Probability */}
      <Card
        title="Failure Risk"
        icon={<AlertTriangle className="h-4 w-4" />}
      >
        <p
          className={`text-3xl font-bold ${
            failureProb > 0.6
              ? 'text-turbine-red'
              : failureProb > 0.3
                ? 'text-turbine-orange'
                : 'text-turbine-green'
          }`}
        >
          {(failureProb * 100).toFixed(1)}
          <span className="text-lg font-normal text-gray-500">%</span>
        </p>
        {scoreBar(
          failureProb,
          1,
          failureProb > 0.6 ? 'bg-turbine-red' : 'bg-turbine-orange',
        )}
      </Card>

      {/* RUL */}
      <Card
        title="Est. RUL"
        icon={<Clock className="h-4 w-4" />}
      >
        {rul !== null ? (
          <p className="text-3xl font-bold text-turbine-blue">
            {rul.toFixed(0)}
            <span className="text-lg font-normal text-gray-500"> hrs</span>
          </p>
        ) : (
          <p className="text-2xl font-bold text-gray-500">&mdash;</p>
        )}
        <p className="text-xs text-gray-500 mt-2">
          {failure?.risk_level ?? 'N/A'} risk
        </p>
      </Card>

      {/* Active Alerts */}
      <Card
        title="Active Alerts"
        icon={<Thermometer className="h-4 w-4" />}
      >
        <p
          className={`text-3xl font-bold ${
            alertCount > 5
              ? 'text-turbine-red'
              : alertCount > 0
                ? 'text-turbine-yellow'
                : 'text-turbine-green'
          }`}
        >
          {alertCount}
        </p>
        <p className="text-xs text-gray-500 mt-2">Last 7 days</p>
      </Card>
    </div>
  );
}

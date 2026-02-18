import { Wrench, ArrowUpCircle, AlertTriangle } from 'lucide-react';
import type { FailurePrediction, AnomalyResult } from '../types/api';

interface MaintenancePanelProps {
  failure: FailurePrediction | null;
  anomaly: AnomalyResult | null;
}

export function MaintenancePanel({ failure, anomaly }: MaintenancePanelProps) {
  // Build recommendations from either array or single recommendation string
  const recommendations: string[] = failure?.recommendations
    ?? (failure?.recommendation ? [failure.recommendation] : []);
  // Build contributing sensors from top_contributing_sensors or contributing_sensors
  const topSensors: { sensor: string; contribution: number }[] =
    anomaly?.top_contributing_sensors ??
    (anomaly?.contributing_sensors?.map((s, i) => ({
      sensor: s,
      contribution: Math.max(0.05, 1.0 / (i + 1)),
    })) ?? []);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Maintenance Recommendations */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2 mb-4">
          <Wrench className="h-4 w-4" /> Maintenance Recommendations
        </h2>

        {recommendations.length === 0 ? (
          <p className="text-gray-600 text-sm py-4 text-center">
            No maintenance actions required at this time.
          </p>
        ) : (
          <ul className="space-y-3">
            {recommendations.map((rec, i) => (
              <li
                key={i}
                className="flex items-start gap-3 text-sm text-gray-300"
              >
                <ArrowUpCircle className="h-4 w-4 text-turbine-blue flex-shrink-0 mt-0.5" />
                <span>{rec}</span>
              </li>
            ))}
          </ul>
        )}

        {failure && (
          <div className="mt-4 pt-4 border-t border-gray-800 grid grid-cols-2 gap-4 text-xs text-gray-400">
            <div>
              <span className="block text-gray-500">Risk Level</span>
              <span
                className={`font-semibold ${
                  failure.risk_level === 'CRITICAL'
                    ? 'text-turbine-red'
                    : failure.risk_level === 'HIGH'
                      ? 'text-turbine-orange'
                      : 'text-turbine-green'
                }`}
              >
                {failure.risk_level}
              </span>
            </div>
            <div>
              <span className="block text-gray-500">Confidence</span>
              <span className="font-semibold text-gray-300">
                {typeof failure.confidence === 'number'
                  ? `${(failure.confidence * 100).toFixed(0)}%`
                  : failure.confidence}
              </span>
            </div>
            <div>
              <span className="block text-gray-500">Failure Probability</span>
              <span className="font-semibold">
                {(failure.failure_probability * 100).toFixed(1)}%
              </span>
            </div>
            <div>
              <span className="block text-gray-500">Est. RUL</span>
              <span className="font-semibold text-turbine-blue">
                {failure.rul_days?.toFixed(1) ?? '—'} days
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Contributing Sensors */}
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2 mb-4">
          <AlertTriangle className="h-4 w-4" /> Top Contributing Sensors
        </h2>

        {topSensors.length === 0 ? (
          <p className="text-gray-600 text-sm py-4 text-center">
            All sensors within normal operating parameters.
          </p>
        ) : (
          <div className="space-y-3">
            {topSensors.map((sensor) => (
              <div key={sensor.sensor} className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-gray-300 font-mono">{sensor.sensor}</span>
                  <span
                    className={`font-bold ${
                      sensor.contribution > 0.3
                        ? 'text-turbine-red'
                        : sensor.contribution > 0.15
                          ? 'text-turbine-yellow'
                          : 'text-gray-400'
                    }`}
                  >
                    {(sensor.contribution * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-gray-700 rounded-full">
                  <div
                    className={`h-1.5 rounded-full ${
                      sensor.contribution > 0.3
                        ? 'bg-turbine-red'
                        : sensor.contribution > 0.15
                          ? 'bg-turbine-yellow'
                          : 'bg-turbine-green'
                    }`}
                    style={{ width: `${Math.min(sensor.contribution * 100, 100)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {anomaly && (
          <div className="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-400">
            <div className="flex justify-between">
              <span>Anomaly Classification</span>
              <span
                className={`font-semibold ${
                  anomaly.is_anomaly ? 'text-turbine-red' : 'text-turbine-green'
                }`}
              >
                {anomaly.is_anomaly ? 'ANOMALY DETECTED' : 'NORMAL'}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

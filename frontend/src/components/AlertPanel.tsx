import { Bell, CheckCircle } from 'lucide-react';
import { format } from 'date-fns';
import type { Alert } from '../types/api';
import { acknowledgeAlert } from '../services/api';
import { useState } from 'react';

const severityStyles: Record<string, string> = {
  INFO: 'border-l-turbine-blue bg-turbine-blue/5',
  WARNING: 'border-l-turbine-yellow bg-turbine-yellow/5',
  HIGH: 'border-l-turbine-orange bg-turbine-orange/5',
  CRITICAL: 'border-l-turbine-red bg-turbine-red/5',
};

const severityBadge: Record<string, string> = {
  INFO: 'bg-turbine-blue/20 text-turbine-blue',
  WARNING: 'bg-turbine-yellow/20 text-turbine-yellow',
  HIGH: 'bg-turbine-orange/20 text-turbine-orange',
  CRITICAL: 'bg-turbine-red/20 text-turbine-red',
};

interface AlertPanelProps {
  alerts: Alert[];
}

export function AlertPanel({ alerts }: AlertPanelProps) {
  const [acknowledgedIds, setAcknowledgedIds] = useState<Set<string>>(new Set());

  async function handleAck(alertId: string) {
    try {
      await acknowledgeAlert(alertId, 'operator');
      setAcknowledgedIds((prev) => new Set(prev).add(alertId));
    } catch {
      // silently ignore – poll will refresh
    }
  }

  const sorted = [...alerts].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  );

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider flex items-center gap-2">
          <Bell className="h-4 w-4" /> Alerts
        </h2>
        <span className="text-xs text-gray-500">{alerts.length} active</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
        {sorted.length === 0 ? (
          <p className="text-gray-600 text-sm text-center py-8">
            No active alerts
          </p>
        ) : (
          sorted.map((alert) => {
            const acked = alert.acknowledged || acknowledgedIds.has(alert.alert_id);
            return (
              <div
                key={alert.alert_id}
                className={`border-l-4 rounded-lg p-3 ${
                  severityStyles[alert.severity] ?? 'border-l-gray-600'
                } ${acked ? 'opacity-50' : ''}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${
                          severityBadge[alert.severity] ?? ''
                        }`}
                      >
                        {alert.severity}
                      </span>
                      <span className="text-[10px] text-gray-500 font-mono">
                        {format(new Date(alert.timestamp), 'MMM dd HH:mm')}
                      </span>
                    </div>
                    <p className="text-xs text-gray-300 leading-relaxed">
                      {alert.message}
                    </p>
                    {alert.sensor_name && (
                      <span className="text-[10px] text-gray-500 mt-1 inline-block font-mono">
                        {alert.sensor_name}
                        {alert.value !== undefined && ` = ${alert.value.toFixed(2)}`}
                        {alert.threshold !== undefined &&
                          ` (threshold: ${alert.threshold})`}
                      </span>
                    )}
                  </div>
                  {!acked && (
                    <button
                      onClick={() => handleAck(alert.alert_id)}
                      className="text-gray-500 hover:text-turbine-green transition-colors flex-shrink-0"
                      title="Acknowledge"
                    >
                      <CheckCircle className="h-4 w-4" />
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

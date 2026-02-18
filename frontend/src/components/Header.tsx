import { Activity, Wifi, WifiOff } from 'lucide-react';
import type { TurbineStatus } from '../types/api';

interface HeaderProps {
  turbineId: string;
  status: TurbineStatus | null;
}

export function Header({ turbineId, status }: HeaderProps) {
  const isOnline = status !== null;
  const riskColor: Record<string, string> = {
    LOW: 'bg-turbine-green',
    MEDIUM: 'bg-turbine-yellow',
    HIGH: 'bg-turbine-orange',
    CRITICAL: 'bg-turbine-red',
  };

  return (
    <header className="bg-gray-900 border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-[1600px] mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-6 w-6 text-turbine-blue" />
          <div>
            <h1 className="text-lg font-semibold leading-none">
              Predictive Maintenance Dashboard
            </h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Siemens SGT-400 Industrial Gas Turbine
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Turbine ID Badge */}
          <span className="bg-gray-800 text-gray-300 text-xs font-mono px-3 py-1 rounded-md">
            {turbineId}
          </span>

          {/* Risk Level pill */}
          {status && (
            <span
              className={`text-xs font-bold px-3 py-1 rounded-full text-gray-950 ${
                riskColor[status.risk_level] ?? 'bg-gray-600'
              }`}
            >
              {status.risk_level} RISK
            </span>
          )}

          {/* Connection indicator */}
          <div className="flex items-center gap-1.5 text-xs">
            {isOnline ? (
              <>
                <Wifi className="h-4 w-4 text-turbine-green" />
                <span className="text-turbine-green">Connected</span>
              </>
            ) : (
              <>
                <WifiOff className="h-4 w-4 text-turbine-red" />
                <span className="text-turbine-red">Offline</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

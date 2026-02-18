import { useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from 'recharts';
import { usePolling } from '../hooks/usePolling';
import { fetchSensorTrend } from '../services/api';
import type { TrendPoint } from '../types/api';

/* ---------- sensor config ---------- */

interface SensorCfg {
  key: string;
  label: string;
  unit: string;
  color: string;
  warnHi?: number;
  critHi?: number;
}

const SENSORS: SensorCfg[] = [
  {
    key: 'exhaust_gas_temp_c',
    label: 'Exhaust Gas Temp',
    unit: '°C',
    color: '#ef4444',
    warnHi: 540,
    critHi: 560,
  },
  {
    key: 'vibration_mm_s',
    label: 'Vibration',
    unit: 'mm/s',
    color: '#f59e0b',
    warnHi: 10,
    critHi: 15,
  },
  {
    key: 'compressor_inlet_temp_c',
    label: 'Compressor Inlet Temp',
    unit: '°C',
    color: '#3b82f6',
  },
  {
    key: 'bearing_temp_c',
    label: 'Bearing Temp',
    unit: '°C',
    color: '#8b5cf6',
    warnHi: 110,
    critHi: 120,
  },
  {
    key: 'power_output_mw',
    label: 'Power Output',
    unit: 'MW',
    color: '#22c55e',
  },
  {
    key: 'fuel_flow_rate_kg_s',
    label: 'Fuel Flow Rate',
    unit: 'kg/s',
    color: '#06b6d4',
  },
];

/* ---------- Formatters ---------- */

function formatTime(ts: string) {
  const d = new Date(ts);
  return `${d.getHours().toString().padStart(2, '0')}:${d
    .getMinutes()
    .toString()
    .padStart(2, '0')}`;
}

/* ---------- Single chart ---------- */

function SensorChart({ cfg, points }: { cfg: SensorCfg; points: TrendPoint[] }) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-300 mb-3">
        {cfg.label}{' '}
        <span className="text-gray-500 font-normal">({cfg.unit})</span>
      </h3>
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={points}>
          <CartesianGrid strokeDasharray="3 3" stroke="#333" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatTime}
            stroke="#666"
            tick={{ fontSize: 10 }}
          />
          <YAxis stroke="#666" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a1a2e',
              border: '1px solid #333',
              borderRadius: 8,
              fontSize: 12,
            }}
            labelFormatter={formatTime}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            name={cfg.label}
            stroke={cfg.color}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 3 }}
          />
          {cfg.warnHi && (
            <ReferenceLine
              y={cfg.warnHi}
              stroke="#f59e0b"
              strokeDasharray="4 4"
              label={{ value: 'Warn', fill: '#f59e0b', fontSize: 10 }}
            />
          )}
          {cfg.critHi && (
            <ReferenceLine
              y={cfg.critHi}
              stroke="#ef4444"
              strokeDasharray="4 4"
              label={{ value: 'Crit', fill: '#ef4444', fontSize: 10 }}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ---------- TrendCharts container ---------- */

interface TrendChartsProps {
  turbineId: string;
}

export function TrendCharts({ turbineId }: TrendChartsProps) {
  const [selected, setSelected] = useState<string[]>([
    'exhaust_gas_temp_c',
    'vibration_mm_s',
    'power_output_mw',
    'bearing_temp_c',
  ]);

  const toggleSensor = (key: string) => {
    setSelected((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          Live Sensor Trends
        </h2>
        <div className="flex gap-1 flex-wrap">
          {SENSORS.map((s) => (
            <button
              key={s.key}
              onClick={() => toggleSensor(s.key)}
              className={`text-[11px] px-2.5 py-1 rounded-full border transition-colors ${
                selected.includes(s.key)
                  ? 'border-turbine-blue bg-turbine-blue/20 text-turbine-blue'
                  : 'border-gray-700 text-gray-500 hover:text-gray-300'
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
        {SENSORS.filter((s) => selected.includes(s.key)).map((cfg) => (
          <TrendChartWrapper key={cfg.key} turbineId={turbineId} cfg={cfg} />
        ))}
      </div>
    </div>
  );
}

/* ---------- Wrapper that manages polling per sensor ---------- */

function TrendChartWrapper({
  turbineId,
  cfg,
}: {
  turbineId: string;
  cfg: SensorCfg;
}) {
  const fetcher = useCallback(
    () => fetchSensorTrend(turbineId, cfg.key, 168),
    [turbineId, cfg.key],
  );
  const { data: points, loading } = usePolling(fetcher, 60000);

  if (loading && !points) {
    return (
      <div className="bg-gray-900 border border-gray-800 rounded-xl p-4 h-[260px] animate-pulse" />
    );
  }

  return <SensorChart cfg={cfg} points={points ?? []} />;
}

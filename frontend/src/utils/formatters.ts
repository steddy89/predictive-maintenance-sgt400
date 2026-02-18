/**
 * Format a number with appropriate precision based on sensor type.
 */
export function formatSensorValue(value: number, sensor: string): string {
  if (sensor.includes('temp')) return `${value.toFixed(1)}°C`;
  if (sensor.includes('vibration')) return `${value.toFixed(2)} mm/s`;
  if (sensor.includes('pressure')) return `${value.toFixed(2)} bar`;
  if (sensor.includes('load') || sensor.includes('mw')) return `${value.toFixed(2)} MW`;
  if (sensor.includes('fuel_flow')) return `${value.toFixed(3)} kg/s`;
  if (sensor.includes('rpm') || sensor.includes('speed')) return `${Math.round(value)} RPM`;
  if (sensor.includes('efficiency')) return `${value.toFixed(1)}%`;
  return value.toFixed(2);
}

/**
 * Map a risk level string to a Tailwind color class.
 */
export function riskLevelColor(level: string): string {
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

/**
 * Map a risk level to a background color class.
 */
export function riskLevelBg(level: string): string {
  switch (level) {
    case 'LOW':
      return 'bg-turbine-green/20';
    case 'MEDIUM':
      return 'bg-turbine-yellow/20';
    case 'HIGH':
      return 'bg-turbine-orange/20';
    case 'CRITICAL':
      return 'bg-turbine-red/20';
    default:
      return 'bg-gray-700/20';
  }
}

/**
 * Format a timestamp for display in charts.
 */
export function formatChartTime(ts: string): string {
  const d = new Date(ts);
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
}

/**
 * Format a timestamp for display in alerts/tables.
 */
export function formatAlertTime(ts: string): string {
  const d = new Date(ts);
  const month = d.toLocaleString('en', { month: 'short' });
  const day = d.getDate().toString().padStart(2, '0');
  const hours = d.getHours().toString().padStart(2, '0');
  const minutes = d.getMinutes().toString().padStart(2, '0');
  return `${month} ${day} ${hours}:${minutes}`;
}

/**
 * Clamp a value between min and max.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Calculate a percentage score bar width.
 */
export function scoreToPercent(value: number, max = 1): number {
  return clamp((value / max) * 100, 0, 100);
}

/**
 * Classify anomaly score into human-readable label.
 */
export function classifyAnomalyScore(score: number): {
  label: string;
  color: string;
} {
  if (score > 0.7) return { label: 'CRITICAL', color: 'text-turbine-red' };
  if (score > 0.4) return { label: 'WARNING', color: 'text-turbine-yellow' };
  if (score > 0.2) return { label: 'ELEVATED', color: 'text-turbine-orange' };
  return { label: 'NORMAL', color: 'text-turbine-green' };
}

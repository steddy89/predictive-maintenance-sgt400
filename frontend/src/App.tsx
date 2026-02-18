import { useState, useCallback } from 'react';
import { Header } from './components/Header';
import { KPIPanel } from './components/KPIPanel';
import { TrendCharts } from './components/TrendCharts';
import { AlertPanel } from './components/AlertPanel';
import { MaintenancePanel } from './components/MaintenancePanel';
import { usePolling } from './hooks/usePolling';
import {
  fetchTurbineStatus,
  fetchAnomalyScore,
  fetchFailurePrediction,
  fetchAlerts,
} from './services/api';

function App() {
  const [turbineId] = useState('SGT400-001');

  const statusFetcher = useCallback(() => fetchTurbineStatus(turbineId), [turbineId]);
  const anomalyFetcher = useCallback(() => fetchAnomalyScore(turbineId), [turbineId]);
  const failureFetcher = useCallback(() => fetchFailurePrediction(turbineId), [turbineId]);
  const alertFetcher = useCallback(() => fetchAlerts(turbineId, 168, true), [turbineId]);

  const { data: status, loading: statusLoading } = usePolling(statusFetcher, 15000);
  const { data: anomaly } = usePolling(anomalyFetcher, 30000);
  const { data: failure } = usePolling(failureFetcher, 60000);
  const { data: alerts } = usePolling(alertFetcher, 30000);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <Header turbineId={turbineId} status={status} />

      <main className="max-w-[1600px] mx-auto px-4 py-6 space-y-6">
        {/* KPI Panel */}
        <KPIPanel
          status={status}
          anomaly={anomaly?.result ?? null}
          failure={failure}
          alertCount={alerts?.length ?? 0}
          loading={statusLoading}
        />

        {/* Charts & Alerts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <TrendCharts turbineId={turbineId} />
          </div>
          <div>
            <AlertPanel alerts={alerts ?? []} />
          </div>
        </div>

        {/* Maintenance Recommendations */}
        <MaintenancePanel failure={failure} anomaly={anomaly?.result ?? null} />
      </main>

      <footer className="text-center text-gray-600 text-xs py-4 border-t border-gray-800">
        SGT400 Predictive Maintenance Dashboard v1.0 | Powered by Azure AI Foundry & Microsoft Fabric
      </footer>
    </div>
  );
}

export default App;

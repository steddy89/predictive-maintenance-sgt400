"""
Model drift detection and monitoring for deployed SGT400 models.

Reference:
  - https://learn.microsoft.com/azure/machine-learning/how-to-monitor-model-performance
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor deployed model performance and detect data drift."""
    
    def __init__(self, baseline_data: pd.DataFrame, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.baseline_stats = self._compute_stats(baseline_data)
        self.drift_history = []
    
    def _compute_stats(self, df: pd.DataFrame) -> dict:
        """Compute statistical profile of dataset."""
        stats_dict = {}
        for col in self.feature_columns:
            if col in df.columns:
                values = df[col].dropna()
                stats_dict[col] = {
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median()),
                    "p05": float(values.quantile(0.05)),
                    "p95": float(values.quantile(0.95)),
                    "distribution": values.values,
                }
        return stats_dict
    
    def detect_drift(
        self,
        current_data: pd.DataFrame,
        significance_level: float = 0.05,
    ) -> dict:
        """
        Detect data drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Recent production data
            significance_level: P-value threshold for drift detection
        
        Returns:
            Drift report dictionary
        """
        drift_results = {}
        drifted_features = []
        
        for col in self.feature_columns:
            if col not in current_data.columns or col not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[col]["distribution"]
            current = current_data[col].dropna().values
            
            if len(current) < 10:
                continue
            
            # KS test
            ks_stat, p_value = stats.ks_2samp(baseline, current)
            
            # Population Stability Index (PSI)
            psi = self._compute_psi(baseline, current)
            
            # Mean shift
            mean_shift = abs(current.mean() - self.baseline_stats[col]["mean"]) / (
                self.baseline_stats[col]["std"] + 1e-10
            )
            
            is_drifted = p_value < significance_level or psi > 0.2
            
            drift_results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": round(float(p_value), 6),
                "psi": round(float(psi), 4),
                "mean_shift_sigma": round(float(mean_shift), 4),
                "is_drifted": is_drifted,
                "current_mean": round(float(current.mean()), 4),
                "baseline_mean": round(float(self.baseline_stats[col]["mean"]), 4),
            }
            
            if is_drifted:
                drifted_features.append(col)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_features": len(drift_results),
            "drifted_features": drifted_features,
            "drift_count": len(drifted_features),
            "drift_percentage": round(len(drifted_features) / max(len(drift_results), 1) * 100, 1),
            "overall_status": "DRIFT_DETECTED" if drifted_features else "STABLE",
            "details": drift_results,
        }
        
        self.drift_history.append(report)
        
        if drifted_features:
            logger.warning(f"Data drift detected in {len(drifted_features)} features: {drifted_features}")
        else:
            logger.info("No significant data drift detected")
        
        return report
    
    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index."""
        min_val = min(baseline.min(), current.min())
        max_val = max(baseline.max(), current.max())
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        baseline_hist = np.histogram(baseline, bins=bin_edges)[0] / len(baseline)
        current_hist = np.histogram(current, bins=bin_edges)[0] / len(current)
        
        # Avoid zero division
        baseline_hist = np.clip(baseline_hist, 1e-6, None)
        current_hist = np.clip(current_hist, 1e-6, None)
        
        psi = np.sum((current_hist - baseline_hist) * np.log(current_hist / baseline_hist))
        
        return psi
    
    def get_monitoring_summary(self) -> dict:
        """Get summary of all monitoring checks."""
        if not self.drift_history:
            return {"status": "NO_CHECKS_RUN", "checks": 0}
        
        return {
            "total_checks": len(self.drift_history),
            "last_check": self.drift_history[-1]["timestamp"],
            "drift_detected_count": sum(
                1 for r in self.drift_history if r["overall_status"] == "DRIFT_DETECTED"
            ),
            "latest_status": self.drift_history[-1]["overall_status"],
        }

"""
============================================================================
SGT400 Predictive Maintenance - ML Training Pipeline
============================================================================
Orchestrates end-to-end ML model training:
  1. Load data from Fabric Gold layer (or local Parquet)
  2. Train anomaly detection (Isolation Forest)
  3. Train sensor forecasting (BiLSTM)
  4. Calibrate RUL estimator
  5. Register models in MLflow / Azure AI Foundry
  6. Run drift detection baseline

Reference:
  - https://learn.microsoft.com/azure/ai-foundry/how-to/deploy-models-managed
  - https://learn.microsoft.com/azure/machine-learning/concept-mlflow
============================================================================
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from models.anomaly_detection import AnomalyDetectionModel
from models.forecasting_rul import TimeSeriesForecaster, RULEstimator
from monitoring.drift_detection import ModelMonitor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Feature columns for drift monitoring
MONITOR_FEATURES = [
    "exhaust_gas_temp_c", "compressor_discharge_temp_c", "vibration_mm_s",
    "inlet_pressure_bar", "discharge_pressure_bar", "turbine_load_mw",
    "fuel_flow_kg_s", "rotor_speed_rpm", "lube_oil_temp_c", "ambient_temp_c",
]


def load_training_data(
    data_path: str | None = None,
    use_fabric: bool = False,
) -> pd.DataFrame:
    """Load training data from Fabric or local Parquet files."""

    if use_fabric:
        logger.info("Loading data from Microsoft Fabric Gold layer...")
        try:
            from pyspark.sql import SparkSession

            spark = SparkSession.builder.appName("SGT400_Training").getOrCreate()
            df = (
                spark.table("gold_ml_training_data")
                .orderBy("timestamp")
                .toPandas()
            )
            logger.info(f"Loaded {len(df)} records from Fabric")
            return df
        except Exception as e:
            logger.warning(f"Fabric connection failed: {e}. Falling back to local data.")

    if data_path and os.path.exists(data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} records ({df.shape[1]} columns)")
        return df

    # Generate sample data if nothing else available
    logger.warning("No data source found. Generating synthetic training data...")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from generate_sample_data import generate_sensor_data

    dfs = [generate_sensor_data(num_days=90, fault_type=None, seed=42)]
    for i, fault in enumerate(["compressor_fouling", "bearing_degradation",
                                "combustion_instability", "hot_gas_path_erosion"]):
        dfs.append(generate_sensor_data(
            num_days=90, fault_type=fault, fault_start_day=30,
            turbine_id=f"SGT400-{i+2:03d}", seed=100 + i,
        ))

    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Generated {len(df)} synthetic records")
    return df


def train_anomaly_model(
    df: pd.DataFrame,
    output_dir: str = "models/artifacts",
    experiment_name: str = "sgt400-anomaly-detection",
) -> dict:
    """Train the Isolation Forest anomaly detection model."""
    logger.info("=" * 60)
    logger.info("TRAINING: Anomaly Detection (Isolation Forest)")
    logger.info("=" * 60)

    model = AnomalyDetectionModel(
        contamination=0.05,
        n_estimators=200,
        random_state=42,
    )

    labels = df.get("fault_type", None)
    metrics = model.train(df, labels=labels, experiment_name=experiment_name)

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "anomaly_detection_model.pkl")
    model.save(model_path)

    logger.info(f"Anomaly model saved to {model_path}")
    logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")

    return metrics


def train_forecasting_model(
    df: pd.DataFrame,
    output_dir: str = "models/artifacts",
    experiment_name: str = "sgt400-forecasting",
    epochs: int = 50,
) -> dict:
    """Train the BiLSTM sensor forecasting model."""
    logger.info("=" * 60)
    logger.info("TRAINING: Sensor Forecasting (BiLSTM)")
    logger.info("=" * 60)

    # Use first turbine's data for time-series training
    primary_turbine = df["turbine_id"].value_counts().index[0]
    df_single = df[df["turbine_id"] == primary_turbine].sort_values("timestamp").copy()

    logger.info(f"Training on turbine {primary_turbine}: {len(df_single)} records")

    forecaster = TimeSeriesForecaster(
        sequence_length=48,
        forecast_horizon=12,
        lstm_units=128,
        dropout_rate=0.2,
    )

    metrics = forecaster.train(
        df_single,
        epochs=epochs,
        batch_size=64,
        experiment_name=experiment_name,
    )

    logger.info(f"Forecasting metrics: {json.dumps(metrics, indent=2)}")
    return metrics


def calibrate_rul_estimator(
    df: pd.DataFrame,
    output_dir: str = "models/artifacts",
) -> dict:
    """Calibrate the RUL estimator on normal operation data."""
    logger.info("=" * 60)
    logger.info("CALIBRATING: RUL Estimator")
    logger.info("=" * 60)

    df_normal = df[df.get("fault_type", "normal") == "normal"].copy()

    if len(df_normal) < 100:
        logger.warning("Insufficient normal data for RUL calibration")
        return {"status": "skipped", "reason": "insufficient_normal_data"}

    rul = RULEstimator(
        degradation_sensors=["vibration_mm_s", "exhaust_gas_temp_c", "efficiency_pct"],
        failure_threshold=0.7,
    )
    rul.calibrate(df_normal)

    # Test on fault data
    df_fault = df[df.get("fault_type", "normal") != "normal"]
    results = {}
    if len(df_fault) > 0:
        for fault_type in df_fault["fault_type"].unique():
            df_ft = df_fault[df_fault["fault_type"] == fault_type].sort_values("timestamp")
            if len(df_ft) > 100:
                estimate = rul.estimate_rul(df_ft)
                results[fault_type] = estimate
                logger.info(f"  {fault_type}: RUL={estimate.get('rul_days', 'N/A')} days, "
                           f"Prob={estimate.get('failure_probability', 'N/A')}")

    # Save baseline stats
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    baseline_path = os.path.join(output_dir, "rul_baseline.pkl")
    with open(baseline_path, "wb") as f:
        pickle.dump(rul.baseline_stats, f)

    return {"status": "calibrated", "fault_estimates": results}


def create_drift_baseline(
    df: pd.DataFrame,
    output_dir: str = "models/artifacts",
) -> dict:
    """Create baseline statistics for drift detection."""
    logger.info("=" * 60)
    logger.info("CREATING: Drift Detection Baseline")
    logger.info("=" * 60)

    df_normal = df[df.get("fault_type", "normal") == "normal"].copy()

    monitor = ModelMonitor(
        baseline_data=df_normal,
        feature_columns=MONITOR_FEATURES,
    )

    # Save baseline statistics
    import pickle
    os.makedirs(output_dir, exist_ok=True)
    baseline_path = os.path.join(output_dir, "drift_baseline.pkl")
    baseline_stats_serializable = {}
    for col, stats in monitor.baseline_stats.items():
        baseline_stats_serializable[col] = {
            k: v for k, v in stats.items() if k != "distribution"
        }
        baseline_stats_serializable[col]["sample_size"] = len(
            stats.get("distribution", [])
        )

    with open(baseline_path, "wb") as f:
        pickle.dump(monitor.baseline_stats, f)

    logger.info(f"Drift baseline saved to {baseline_path}")
    logger.info(f"Monitored features: {MONITOR_FEATURES}")

    return {"status": "created", "features": MONITOR_FEATURES}


def run_full_pipeline(args: argparse.Namespace) -> dict:
    """Execute the complete training pipeline."""
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info("SGT400 PREDICTIVE MAINTENANCE - ML TRAINING PIPELINE")
    logger.info(f"Started at: {start_time.isoformat()}")
    logger.info("=" * 70)

    # Set MLflow tracking URI
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # Step 1: Load data
    df = load_training_data(
        data_path=args.data_path,
        use_fabric=args.use_fabric,
    )

    pipeline_results = {
        "start_time": start_time.isoformat(),
        "data_records": len(df),
        "data_columns": len(df.columns),
    }

    # Step 2: Train anomaly model
    if "anomaly" in args.models or "all" in args.models:
        anomaly_metrics = train_anomaly_model(
            df, output_dir=args.output_dir,
            experiment_name=f"{args.experiment_prefix}-anomaly",
        )
        pipeline_results["anomaly_detection"] = anomaly_metrics

    # Step 3: Train forecasting model
    if "forecast" in args.models or "all" in args.models:
        forecast_metrics = train_forecasting_model(
            df, output_dir=args.output_dir,
            experiment_name=f"{args.experiment_prefix}-forecasting",
            epochs=args.epochs,
        )
        pipeline_results["forecasting"] = forecast_metrics

    # Step 4: Calibrate RUL
    if "rul" in args.models or "all" in args.models:
        rul_results = calibrate_rul_estimator(df, output_dir=args.output_dir)
        pipeline_results["rul_estimation"] = rul_results

    # Step 5: Create drift baseline
    if "drift" in args.models or "all" in args.models:
        drift_results = create_drift_baseline(df, output_dir=args.output_dir)
        pipeline_results["drift_baseline"] = drift_results

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    pipeline_results["end_time"] = end_time.isoformat()
    pipeline_results["duration_seconds"] = round(duration, 1)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info("=" * 70)

    # Save pipeline results
    results_path = os.path.join(args.output_dir, "pipeline_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    return pipeline_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SGT400 ML Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/sample/sensor_data_training.parquet",
        help="Path to training data (Parquet)",
    )
    parser.add_argument(
        "--use-fabric",
        action="store_true",
        help="Load data from Microsoft Fabric Gold layer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/artifacts",
        help="Directory for model artifacts",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["all"],
        choices=["all", "anomaly", "forecast", "rul", "drift"],
        help="Which models to train",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for LSTM model",
    )
    parser.add_argument(
        "--experiment-prefix",
        type=str,
        default="sgt400",
        help="MLflow experiment name prefix",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (defaults to local)",
    )

    args = parser.parse_args()
    run_full_pipeline(args)

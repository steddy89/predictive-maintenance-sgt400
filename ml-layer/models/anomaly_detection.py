"""
============================================================================
SGT400 Predictive Maintenance - Anomaly Detection Model
============================================================================
Isolation Forest model for detecting anomalous turbine sensor patterns.

Reference:
  - https://learn.microsoft.com/azure/machine-learning/how-to-deploy-online-endpoints
  - https://learn.microsoft.com/azure/ai-foundry/how-to/deploy-models-managed
"""

import os
import json
import logging
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature columns for anomaly detection
FEATURE_COLUMNS = [
    "exhaust_gas_temp_c", "compressor_discharge_temp_c", "vibration_mm_s",
    "inlet_pressure_bar", "discharge_pressure_bar", "turbine_load_mw",
    "fuel_flow_kg_s", "rotor_speed_rpm", "lube_oil_temp_c", "ambient_temp_c",
    "pressure_ratio", "heat_rate", "efficiency_pct",
    "temp_differential", "specific_fuel_consumption",
    "vibration_speed_ratio",
]

# Rolling feature columns (generated in Silver layer)
ROLLING_FEATURE_COLUMNS = []
for sensor in ["exhaust_gas_temp_c", "vibration_mm_s", "discharge_pressure_bar",
               "turbine_load_mw", "fuel_flow_kg_s"]:
    for window in [12, 144]:
        ROLLING_FEATURE_COLUMNS.extend([
            f"{sensor}_rolling_mean_{window}",
            f"{sensor}_rolling_std_{window}",
        ])

ALL_FEATURES = FEATURE_COLUMNS + ROLLING_FEATURE_COLUMNS


class AnomalyDetectionModel:
    """Isolation Forest-based anomaly detection for SGT400 turbine."""
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        max_samples: str | int = "auto",
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("isolation_forest", IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                max_samples=max_samples,
                random_state=random_state,
                n_jobs=-1,
                verbose=0,
            )),
        ])
        
        self.feature_columns = ALL_FEATURES
        self.is_fitted = False
        self.training_metadata = {}
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and validate feature matrix from DataFrame."""
        available = [c for c in self.feature_columns if c in df.columns]
        if len(available) < len(FEATURE_COLUMNS):
            missing = set(FEATURE_COLUMNS) - set(available)
            raise ValueError(f"Missing required features: {missing}")
        
        X = df[available].copy()
        X = X.fillna(X.median())
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.feature_columns_used = available
        return X.values
    
    def train(
        self,
        df: pd.DataFrame,
        labels: pd.Series | None = None,
        experiment_name: str = "sgt400-anomaly-detection",
    ) -> dict:
        """
        Train the anomaly detection model.
        
        Args:
            df: Training DataFrame with sensor features
            labels: Optional ground truth labels (1=normal, -1=anomaly) for evaluation
            experiment_name: MLflow experiment name
        
        Returns:
            Dictionary with training metrics
        """
        logger.info("Preparing features for training...")
        X = self.prepare_features(df)
        
        logger.info(f"Training Isolation Forest on {X.shape[0]} samples, {X.shape[1]} features")
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_params({
                "model_type": "IsolationForest",
                "contamination": self.contamination,
                "n_estimators": self.n_estimators,
                "max_samples": str(self.max_samples),
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "feature_columns": json.dumps(self.feature_columns_used[:20]),  # Log first 20
            })
            
            # Fit model
            self.pipeline.fit(X)
            self.is_fitted = True
            
            # Predictions
            predictions = self.pipeline.predict(X)
            anomaly_scores = self.pipeline.named_steps["isolation_forest"].decision_function(
                self.pipeline.named_steps["scaler"].transform(X)
            )
            
            # Metrics
            n_anomalies = (predictions == -1).sum()
            anomaly_rate = n_anomalies / len(predictions)
            
            metrics = {
                "anomaly_rate": float(anomaly_rate),
                "n_anomalies": int(n_anomalies),
                "n_normal": int((predictions == 1).sum()),
                "mean_anomaly_score": float(np.mean(anomaly_scores)),
                "std_anomaly_score": float(np.std(anomaly_scores)),
            }
            
            # Evaluate against ground truth if available
            if labels is not None:
                true_labels = labels.map(lambda x: -1 if x != "normal" else 1).values
                precision, recall, f1, _ = precision_recall_fscore_support(
                    true_labels, predictions, average="binary", pos_label=-1
                )
                try:
                    auc = roc_auc_score(
                        (true_labels == -1).astype(int),
                        -anomaly_scores  # Negate for ROC: higher = more anomalous
                    )
                except ValueError:
                    auc = 0.0
                
                metrics.update({
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "auc_roc": float(auc),
                })
                
                logger.info(f"Evaluation - Precision: {precision:.4f}, Recall: {recall:.4f}, "
                           f"F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                self.pipeline,
                artifact_path="anomaly_detection_model",
                registered_model_name="sgt400-anomaly-detection",
            )
            
            # Save feature importance (based on isolation depth)
            self.training_metadata = {
                "trained_at": datetime.now().isoformat(),
                "metrics": metrics,
                "features_used": self.feature_columns_used,
            }
            
            mlflow.log_dict(self.training_metadata, "training_metadata.json")
            
            logger.info(f"Training complete. Anomaly rate: {anomaly_rate:.4f}")
            logger.info(f"Model registered as 'sgt400-anomaly-detection'")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> dict:
        """
        Run anomaly detection inference.
        
        Returns:
            Dictionary with predictions and anomaly scores
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() first or load a saved model.")
        
        X = self.prepare_features(df)
        
        predictions = self.pipeline.predict(X)
        scores_raw = self.pipeline.named_steps["isolation_forest"].decision_function(
            self.pipeline.named_steps["scaler"].transform(X)
        )
        
        # Normalize scores to 0-1 range (1 = most anomalous)
        scores_normalized = 1.0 - (scores_raw - scores_raw.min()) / (scores_raw.max() - scores_raw.min() + 1e-10)
        
        return {
            "is_anomaly": (predictions == -1).tolist(),
            "anomaly_score": scores_normalized.tolist(),
            "raw_score": scores_raw.tolist(),
        }
    
    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "pipeline": self.pipeline,
                "feature_columns_used": self.feature_columns_used,
                "training_metadata": self.training_metadata,
                "is_fitted": self.is_fitted,
            }, f)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "AnomalyDetectionModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        model = cls()
        model.pipeline = data["pipeline"]
        model.feature_columns_used = data["feature_columns_used"]
        model.training_metadata = data["training_metadata"]
        model.is_fitted = data["is_fitted"]
        
        logger.info(f"Model loaded from {path}")
        return model


def train_anomaly_model(data_path: str = None, use_fabric: bool = False) -> AnomalyDetectionModel:
    """
    Convenience function to train anomaly detection model.
    
    Args:
        data_path: Path to parquet training data (if not using Fabric)
        use_fabric: If True, read from Fabric Lakehouse Gold table
    """
    if use_fabric:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = spark.table("gold_ml_training_data").toPandas()
    elif data_path:
        df = pd.read_parquet(data_path)
    else:
        raise ValueError("Either data_path or use_fabric must be specified")
    
    logger.info(f"Training data shape: {df.shape}")
    
    model = AnomalyDetectionModel(
        contamination=0.05,
        n_estimators=200,
    )
    
    labels = df.get("fault_type", None)
    metrics = model.train(df, labels=labels)
    
    model.save("models/anomaly_detection_model.pkl")
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sample/sensor_data_training.parquet")
    parser.add_argument("--use-fabric", action="store_true")
    args = parser.parse_args()
    
    train_anomaly_model(data_path=args.data_path, use_fabric=args.use_fabric)

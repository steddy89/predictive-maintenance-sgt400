"""
============================================================================
SGT400 Predictive Maintenance - Time-Series Forecasting & RUL Estimation
============================================================================
LSTM-based forecasting model and Remaining Useful Life (RUL) estimation.

Reference:
  - https://learn.microsoft.com/azure/machine-learning/how-to-deploy-online-endpoints
  - https://learn.microsoft.com/azure/ai-foundry/how-to/deploy-models-managed
"""

import os
import json
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import mlflow
import mlflow.keras

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target sensors for forecasting
FORECAST_SENSORS = [
    "exhaust_gas_temp_c",
    "vibration_mm_s",
    "discharge_pressure_bar",
    "turbine_load_mw",
    "fuel_flow_kg_s",
]

# All input features for the LSTM
INPUT_FEATURES = [
    "exhaust_gas_temp_c", "compressor_discharge_temp_c", "vibration_mm_s",
    "inlet_pressure_bar", "discharge_pressure_bar", "turbine_load_mw",
    "fuel_flow_kg_s", "rotor_speed_rpm", "lube_oil_temp_c", "ambient_temp_c",
    "pressure_ratio", "efficiency_pct",
]


class TimeSeriesForecaster:
    """LSTM-based time-series forecasting for turbine sensor predictions."""
    
    def __init__(
        self,
        sequence_length: int = 48,        # 4 hours at 5-min intervals
        forecast_horizon: int = 12,        # 1 hour ahead
        lstm_units: int = 128,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
    ):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def _build_model(self, n_features: int, n_targets: int):
        """Build LSTM model architecture."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import (
                LSTM, Dense, Dropout, BatchNormalization, Bidirectional
            )
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            logger.warning("TensorFlow not available. Install with: pip install tensorflow")
            raise
        
        model = Sequential([
            Bidirectional(
                LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, n_features)),
                name="bidirectional_lstm_1"
            ),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.lstm_units // 2, return_sequences=False, name="lstm_2"),
            Dropout(self.dropout_rate),
            BatchNormalization(),
            
            Dense(64, activation="relu", name="dense_1"),
            Dropout(self.dropout_rate / 2),
            Dense(n_targets * self.forecast_horizon, activation="linear", name="output"),
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss="huber",
            metrics=["mae", "mse"],
        )
        
        return model
    
    def _create_sequences(
        self,
        data: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and target arrays for LSTM training."""
        
        X, y = [], []
        total_len = len(data) - self.sequence_length - self.forecast_horizon + 1
        
        for i in range(total_len):
            X.append(data[i : i + self.sequence_length])
            # Target: forecast_horizon steps of target sensors
            y.append(targets[i + self.sequence_length : i + self.sequence_length + self.forecast_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 64,
        validation_split: float = 0.2,
        experiment_name: str = "sgt400-forecasting",
    ) -> dict:
        """
        Train the LSTM forecasting model.
        
        Args:
            df: DataFrame with time-ordered sensor data for a single turbine
            epochs: Maximum training epochs
            batch_size: Training batch size
            validation_split: Fraction of data for validation
            experiment_name: MLflow experiment name
        
        Returns:
            Training metrics dictionary
        """
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        logger.info("Preparing data for LSTM training...")
        
        # Select and validate features
        available_features = [f for f in INPUT_FEATURES if f in df.columns]
        available_targets = [f for f in FORECAST_SENSORS if f in df.columns]
        
        data = df[available_features].values
        target_data = df[available_targets].values
        
        # Scale data
        self.scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        
        data_scaled = self.scaler.fit_transform(data)
        target_scaled = self.target_scaler.fit_transform(target_data)
        
        # Create sequences
        X, y = self._create_sequences(data_scaled, target_scaled)
        logger.info(f"Created {X.shape[0]} sequences of length {self.sequence_length}")
        
        # Train/validation split (chronological)
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Build model
        n_targets = len(available_targets)
        self.model = self._build_model(n_features=X.shape[2], n_targets=n_targets)
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
        ]
        
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"lstm_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                "model_type": "BiLSTM",
                "sequence_length": self.sequence_length,
                "forecast_horizon": self.forecast_horizon,
                "lstm_units": self.lstm_units,
                "dropout_rate": self.dropout_rate,
                "learning_rate": self.learning_rate,
                "n_features": X.shape[2],
                "n_targets": n_targets,
                "n_train_samples": X_train.shape[0],
                "n_val_samples": X_val.shape[0],
                "epochs": epochs,
                "batch_size": batch_size,
            })
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )
            
            self.is_fitted = True
            
            # Metrics
            val_loss = min(history.history["val_loss"])
            val_mae = min(history.history["val_mae"])
            train_loss = min(history.history["loss"])
            
            metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_mae": float(val_mae),
                "best_epoch": int(np.argmin(history.history["val_loss"]) + 1),
                "total_epochs": len(history.history["loss"]),
            }
            
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.keras.log_model(
                self.model,
                artifact_path="forecasting_model",
                registered_model_name="sgt400-sensor-forecasting",
            )
            
            logger.info(f"Training complete. Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}")
        
        return metrics
    
    def predict(self, recent_data: pd.DataFrame) -> dict:
        """
        Forecast future sensor values.
        
        Args:
            recent_data: Last `sequence_length` sensor readings
        
        Returns:
            Dictionary with forecast arrays per sensor
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        
        available_features = [f for f in INPUT_FEATURES if f in recent_data.columns]
        data = recent_data[available_features].values[-self.sequence_length:]
        
        data_scaled = self.scaler.transform(data)
        X = data_scaled.reshape(1, self.sequence_length, -1)
        
        prediction_scaled = self.model.predict(X, verbose=0)
        n_targets = len([f for f in FORECAST_SENSORS if f in recent_data.columns])
        prediction_reshaped = prediction_scaled.reshape(self.forecast_horizon, n_targets)
        
        prediction = self.target_scaler.inverse_transform(prediction_reshaped)
        
        available_targets = [f for f in FORECAST_SENSORS if f in recent_data.columns]
        result = {}
        for i, sensor in enumerate(available_targets):
            result[sensor] = prediction[:, i].tolist()
        
        return result


class RULEstimator:
    """
    Remaining Useful Life (RUL) estimation using exponential degradation model.
    Combines statistical degradation tracking with ML-based prediction.
    """
    
    def __init__(
        self,
        degradation_sensors: list[str] | None = None,
        failure_threshold: float = 0.7,
        window_days: int = 7,
    ):
        self.degradation_sensors = degradation_sensors or [
            "vibration_mm_s", "exhaust_gas_temp_c", "efficiency_pct"
        ]
        self.failure_threshold = failure_threshold
        self.window_days = window_days
        self.baseline_stats = {}
        self.is_calibrated = False
    
    def calibrate(self, df_normal: pd.DataFrame):
        """
        Calibrate baseline statistics from normal operation data.
        
        Args:
            df_normal: DataFrame containing only normal operation data
        """
        for sensor in self.degradation_sensors:
            if sensor in df_normal.columns:
                self.baseline_stats[sensor] = {
                    "mean": float(df_normal[sensor].mean()),
                    "std": float(df_normal[sensor].std()),
                    "p05": float(df_normal[sensor].quantile(0.05)),
                    "p95": float(df_normal[sensor].quantile(0.95)),
                }
        
        self.is_calibrated = True
        logger.info(f"RUL estimator calibrated with {len(self.baseline_stats)} sensors")
    
    def compute_degradation_index(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute a composite degradation index (0 = healthy, 1 = failing).
        
        Uses Mahalanobis-like distance from baseline normal operation.
        """
        if not self.is_calibrated:
            raise RuntimeError("Estimator not calibrated. Call calibrate() first.")
        
        scores = pd.DataFrame(index=df.index)
        
        for sensor in self.degradation_sensors:
            if sensor not in df.columns or sensor not in self.baseline_stats:
                continue
            
            baseline = self.baseline_stats[sensor]
            
            # Normalized deviation from baseline
            if sensor == "efficiency_pct":
                # Lower efficiency = worse
                deviation = (baseline["mean"] - df[sensor]) / (baseline["std"] + 1e-10)
            else:
                # Higher value = worse (for temp, vibration)
                deviation = (df[sensor] - baseline["mean"]) / (baseline["std"] + 1e-10)
            
            # Clip and normalize to 0-1
            scores[sensor] = np.clip(deviation / 5.0, 0, 1)  # 5-sigma = max
        
        # Weighted average degradation index
        weights = {
            "vibration_mm_s": 0.4,
            "exhaust_gas_temp_c": 0.35,
            "efficiency_pct": 0.25,
        }
        
        degradation = pd.Series(0.0, index=df.index)
        total_weight = 0
        for sensor in scores.columns:
            w = weights.get(sensor, 1.0 / len(scores.columns))
            degradation += scores[sensor] * w
            total_weight += w
        
        degradation /= total_weight
        
        return degradation
    
    def estimate_rul(self, df: pd.DataFrame, interval_minutes: int = 5) -> dict:
        """
        Estimate Remaining Useful Life.
        
        Args:
            df: Recent sensor data (at least window_days worth)
            interval_minutes: Data sampling interval
        
        Returns:
            Dictionary with RUL estimate and confidence
        """
        degradation = self.compute_degradation_index(df)
        
        # Calculate degradation rate (slope)
        samples_per_day = (24 * 60) // interval_minutes
        window_samples = self.window_days * samples_per_day
        
        recent_degradation = degradation.iloc[-window_samples:]
        
        if len(recent_degradation) < 2:
            return {
                "rul_days": None,
                "failure_probability": 0.0,
                "degradation_index": float(degradation.iloc[-1]),
                "confidence": "LOW",
            }
        
        # Linear regression on recent degradation trend
        x = np.arange(len(recent_degradation))
        y = recent_degradation.values
        
        # Remove NaN
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return {
                "rul_days": None,
                "failure_probability": float(degradation.iloc[-1]),
                "degradation_index": float(degradation.iloc[-1]),
                "confidence": "LOW",
            }
        
        coeffs = np.polyfit(x[mask], y[mask], 1)
        slope = coeffs[0]  # degradation per sample
        current_level = float(degradation.iloc[-1])
        
        # Estimate time to failure threshold
        if slope > 0:
            remaining_degradation = self.failure_threshold - current_level
            if remaining_degradation > 0:
                samples_to_failure = remaining_degradation / slope
                rul_days = samples_to_failure / samples_per_day
            else:
                rul_days = 0  # Already past threshold
        else:
            rul_days = 365  # Improving or stable, set to max
        
        # Failure probability (exponential mapping)
        failure_prob = min(1.0, np.exp(current_level * 3 - 2))
        
        # Confidence based on R² of trend fit
        y_pred = np.polyval(coeffs, x[mask])
        ss_res = np.sum((y[mask] - y_pred) ** 2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask])) ** 2) + 1e-10
        r_squared = 1 - ss_res / ss_tot
        
        if r_squared > 0.8:
            confidence = "HIGH"
        elif r_squared > 0.5:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
        
        return {
            "rul_days": round(float(rul_days), 1),
            "failure_probability": round(float(failure_prob), 4),
            "degradation_index": round(current_level, 4),
            "degradation_rate_per_day": round(float(slope * samples_per_day), 6),
            "confidence": confidence,
            "r_squared": round(float(r_squared), 4),
            "recommendation": self._get_recommendation(rul_days, failure_prob),
        }
    
    def _get_recommendation(self, rul_days: float, failure_prob: float) -> str:
        """Generate maintenance recommendation based on RUL and failure probability."""
        if rul_days <= 3 or failure_prob > 0.8:
            return "IMMEDIATE: Schedule emergency maintenance within 24 hours"
        elif rul_days <= 7 or failure_prob > 0.6:
            return "URGENT: Plan maintenance within the next 3-5 days"
        elif rul_days <= 14 or failure_prob > 0.4:
            return "WARNING: Schedule maintenance within 2 weeks"
        elif rul_days <= 30 or failure_prob > 0.2:
            return "ADVISORY: Monitor closely, plan maintenance in next scheduled window"
        else:
            return "NORMAL: Continue standard monitoring schedule"


def train_forecasting_model(
    data_path: str = None,
    use_fabric: bool = False,
) -> TimeSeriesForecaster:
    """Convenience function to train forecasting model."""
    
    if use_fabric:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        df = (
            spark.table("gold_ml_training_data")
            .filter("turbine_id = 'SGT400-001'")
            .orderBy("timestamp")
            .toPandas()
        )
    elif data_path:
        df = pd.read_parquet(data_path)
        df = df[df["turbine_id"] == df["turbine_id"].iloc[0]].sort_values("timestamp")
    else:
        raise ValueError("Either data_path or use_fabric must be specified")
    
    model = TimeSeriesForecaster(
        sequence_length=48,
        forecast_horizon=12,
        lstm_units=128,
    )
    
    metrics = model.train(df, epochs=50, batch_size=64)
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sample/sensor_data_training.parquet")
    parser.add_argument("--model-type", choices=["forecast", "rul", "both"], default="both")
    parser.add_argument("--use-fabric", action="store_true")
    args = parser.parse_args()
    
    if args.model_type in ("forecast", "both"):
        logger.info("Training forecasting model...")
        forecaster = train_forecasting_model(data_path=args.data_path)
    
    if args.model_type in ("rul", "both"):
        logger.info("Calibrating RUL estimator...")
        df = pd.read_parquet(args.data_path)
        df_normal = df[df["fault_type"] == "normal"]
        
        rul = RULEstimator()
        rul.calibrate(df_normal)
        
        # Test RUL estimation on fault data
        df_fault = df[df["fault_type"] != "normal"]
        if len(df_fault) > 0:
            result = rul.estimate_rul(df_fault)
            logger.info(f"RUL Estimate: {json.dumps(result, indent=2)}")

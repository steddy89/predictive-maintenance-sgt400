"""
Scoring script for SGT400 Sensor Forecasting endpoint.
Azure ML Managed Online Endpoint Entry Point for the BiLSTM model.

Reference: https://learn.microsoft.com/azure/machine-learning/how-to-deploy-online-endpoints
"""
import os
import json
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Globals set during init()
model = None
scaler = None
target_scaler = None
sequence_length = 48

INPUT_FEATURES = [
    "exhaust_gas_temp_c", "compressor_discharge_temp_c", "vibration_mm_s",
    "inlet_pressure_bar", "discharge_pressure_bar", "turbine_load_mw",
    "fuel_flow_kg_s", "rotor_speed_rpm", "lube_oil_temp_c", "ambient_temp_c",
    "pressure_ratio", "efficiency_pct",
]

FORECAST_SENSORS = [
    "exhaust_gas_temp_c", "vibration_mm_s", "discharge_pressure_bar",
    "turbine_load_mw", "fuel_flow_kg_s",
]


def init():
    """Initialize model - called once when the container starts."""
    global model, scaler, target_scaler, sequence_length

    try:
        import tensorflow as tf
    except ImportError:
        logger.error("TensorFlow not available in this environment")
        raise

    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")

    # Load Keras model
    model_path = os.path.join(model_dir, "forecasting_model")
    if os.path.isdir(model_path):
        model = tf.keras.models.load_model(model_path)
    else:
        model = tf.keras.models.load_model(
            os.path.join(model_dir, "forecasting_model.h5")
        )

    # Load scalers
    scaler_path = os.path.join(model_dir, "input_scaler.pkl")
    target_scaler_path = os.path.join(model_dir, "target_scaler.pkl")

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    if os.path.exists(target_scaler_path):
        with open(target_scaler_path, "rb") as f:
            target_scaler = pickle.load(f)

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        sequence_length = cfg.get("sequence_length", 48)

    logger.info(
        "Forecasting model loaded. Sequence length: %d, "
        "Input features: %d, Target sensors: %d",
        sequence_length,
        len(INPUT_FEATURES),
        len(FORECAST_SENSORS),
    )


def run(raw_data: str) -> str:
    """
    Run forecasting inference.

    Input format:
    {
        "data": [
            {"exhaust_gas_temp_c": 550, "vibration_mm_s": 2.8, ...},
            ...  // at least `sequence_length` records
        ]
    }
    """
    try:
        input_data = json.loads(raw_data)
        records = input_data["data"]
        df = pd.DataFrame(records)

        available_features = [f for f in INPUT_FEATURES if f in df.columns]
        data = df[available_features].values

        # Ensure we have enough data
        if len(data) < sequence_length:
            return json.dumps({
                "error": f"Need at least {sequence_length} records, got {len(data)}."
            })

        # Take last sequence_length records
        data = data[-sequence_length:]

        # Scale if scaler available
        if scaler is not None:
            data_scaled = scaler.transform(data)
        else:
            data_scaled = data

        X = data_scaled.reshape(1, sequence_length, -1)
        prediction_scaled = model.predict(X, verbose=0)

        n_targets = len(FORECAST_SENSORS)
        forecast_horizon = prediction_scaled.shape[1] // n_targets
        prediction_reshaped = prediction_scaled.reshape(forecast_horizon, n_targets)

        # Inverse transform if target scaler available
        if target_scaler is not None:
            prediction = target_scaler.inverse_transform(prediction_reshaped)
        else:
            prediction = prediction_reshaped

        # Build response
        forecasts = {}
        for i, sensor in enumerate(FORECAST_SENSORS):
            forecasts[sensor] = [round(float(v), 4) for v in prediction[:, i]]

        result = {
            "forecasts": forecasts,
            "forecast_horizon": int(forecast_horizon),
            "interval_minutes": 5,
            "sensors": FORECAST_SENSORS,
        }

        return json.dumps(result)

    except Exception as e:
        logger.error("Forecast inference error: %s", str(e))
        return json.dumps({"error": str(e)})

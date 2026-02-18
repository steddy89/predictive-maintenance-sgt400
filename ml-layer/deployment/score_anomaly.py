"""
Scoring script for SGT400 Anomaly Detection endpoint.
Azure ML Managed Online Endpoint Entry Point.

Reference: https://learn.microsoft.com/azure/machine-learning/how-to-deploy-online-endpoints
"""
import os
import json
import logging
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def init():
    """Initialize model - called once when the container starts."""
    global model, feature_columns
    
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", "."), "anomaly_detection_model.pkl")
    
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    
    model = data["pipeline"]
    feature_columns = data["feature_columns_used"]
    
    logger.info(f"Anomaly detection model loaded. Features: {len(feature_columns)}")


def run(raw_data: str) -> str:
    """
    Run inference on input data.
    
    Input format:
    {
        "data": [
            {"exhaust_gas_temp_c": 550.2, "vibration_mm_s": 2.8, ...},
            ...
        ]
    }
    """
    try:
        input_data = json.loads(raw_data)
        df = pd.DataFrame(input_data["data"])
        
        available = [c for c in feature_columns if c in df.columns]
        X = df[available].fillna(0).replace([np.inf, -np.inf], 0).values
        
        predictions = model.predict(X)
        raw_scores = model.named_steps["isolation_forest"].decision_function(
            model.named_steps["scaler"].transform(X)
        )
        
        # Normalize scores to 0-1
        scores_norm = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)
        
        results = {
            "predictions": [
                {
                    "is_anomaly": bool(pred == -1),
                    "anomaly_score": round(float(score), 4),
                    "raw_score": round(float(raw), 4),
                }
                for pred, score, raw in zip(predictions, scores_norm, raw_scores)
            ],
            "summary": {
                "total_records": len(predictions),
                "anomalies_detected": int((predictions == -1).sum()),
                "max_anomaly_score": round(float(scores_norm.max()), 4),
                "mean_anomaly_score": round(float(scores_norm.mean()), 4),
            }
        }
        
        return json.dumps(results)
    
    except Exception as e:
        logger.error(f"Inference error: {str(e)}")
        return json.dumps({"error": str(e)})

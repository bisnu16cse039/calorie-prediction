#!/usr/bin/env python
"""
Prediction script for the Calories Prediction model.
This script loads a trained model and makes predictions on new data.
Parameters are loaded from config.yaml instead of command-line args.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
CONFIG_PATH = os.getenv('CALORIES_PRED_CONFIG', 'config.yaml')
logger.info(f"Loading config from {CONFIG_PATH}")
try:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    sys.exit(1)

# Extract paths and settings
model_path = cfg.get('model_path')
data_path = cfg.get('predict_data_path')
output_path = cfg.get('prediction_output_path')
target_column = cfg.get('target', None)

# Validate required config entries
for key in ['model_path', 'predict_data_path']:
    if not cfg.get(key):
        logger.error(f"Missing required config entry: '{key}'")
        sys.exit(1)


def load_model(path: str) -> Any:
    logger.info(f"Loading model from {path}")
    return joblib.load(path)


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
    df_copy = df.copy()
    if 'id' in df_copy.columns:
        df_copy = df_copy.drop('id', axis=1)
    if target_col and target_col in df_copy.columns:
        df_copy = df_copy.drop(target_col, axis=1)
    return df_copy


def make_predictions(model: Any, X: pd.DataFrame) -> np.ndarray:
    logger.info("Making predictions...")
    preds = model.predict(X)
    return np.maximum(preds, 0)


def save_predictions(preds: np.ndarray, out_path: str, ids: Optional[pd.Series] = None) -> None:
    logger.info(f"Saving predictions to {out_path}")
    if ids is not None:
        df_out = pd.DataFrame({'id': ids, 'Calories': preds})
    else:
        df_out = pd.DataFrame({'Calories': preds})
    df_out.to_csv(out_path, index=False)


def calculate_metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    actual = np.maximum(actual, 0)
    pred = np.maximum(pred, 0)
    rmsle = np.sqrt(mean_squared_error(np.log1p(actual), np.log1p(pred)))
    metrics = {
        'MAE': mean_absolute_error(actual, pred),
        'RMSE': np.sqrt(mean_squared_error(actual, pred)),
        'RMSLE': rmsle,
        'R2': r2_score(actual, pred)
    }
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    return metrics


def main():
    # Load model and data
    model = load_model(model_path)
    df = load_data(data_path)

    # Preserve IDs and true values if available
    ids = df['id'] if 'id' in df.columns else None
    true_vals = df[target_column].values if target_column and target_column in df.columns else None

    # Preprocess and predict
    X = preprocess_data(df, target_column)
    preds = make_predictions(model, X)
    # Save
    save_predictions(preds, output_path, ids)

    # Metrics
    if true_vals is not None:
        calculate_metrics(true_vals, preds)

    logger.info("Prediction completed successfully!")

if __name__ == '__main__':
    main()

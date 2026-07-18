"""
config.py — Central configuration for PGA Predictor
"""

import os

# Base directory — all files live in the same folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model Settings ────────────────────────────────────────────────────────────
FORM_WINDOW_WEEKS    = 12
COURSE_HISTORY_YEARS = 5
MIN_COURSE_STARTS    = 2
MODEL_TARGET         = "top10"  # "top10", "top5", "win", "regression"

# ── XGBoost Hyperparameters ───────────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      500,
    "max_depth":         5,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "gamma":             0.1,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "random_state":      42,
    "eval_metric":       "logloss",
    "use_label_encoder": False,
}

# ── File Paths — all relative to BASE_DIR (your Golf Model folder) ────────────
RAW_DATA_DIR       = os.path.join(BASE_DIR, "data", "raw") + os.sep
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed") + os.sep
MODEL_OUTPUT_DIR   = os.path.join(BASE_DIR, "output", "models") + os.sep
PREDICTIONS_DIR    = os.path.join(BASE_DIR, "output", "predictions") + os.sep

# Create directories if they don't exist
for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_OUTPUT_DIR, PREDICTIONS_DIR]:
    os.makedirs(d, exist_ok=True)

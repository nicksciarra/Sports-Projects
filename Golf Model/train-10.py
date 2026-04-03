"""
models/train.py — Train XGBoost model on historical PGA season stats
---
Since we're using season-level stats (not round-by-round),
the model learns which player skill profiles correlate with
top-10 finishes using SG splits, driving, putting, etc.

Usage:
    python train.py
    python train.py --target top5
    python train.py --target win
"""

import os
import sys
import json
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import XGB_PARAMS, MODEL_TARGET, MODEL_OUTPUT_DIR, RAW_DATA_DIR

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# ── Elite tournament filter ───────────────────────────────────────────────────
ELITE_TOURNAMENTS = [
    'Masters', 'U.S. Open', 'The Open Championship', 'PGA Championship',
    'PLAYERS', 'Players Championship',
    'Memorial Tournament',
    'Arnold Palmer Invitational',
    'Genesis Invitational', 'Genesis Open',
    'Genesis Scottish Open', 'Scottish Open',
    'Waste Management Phoenix Open', 'WM Phoenix Open',
]

def is_elite(name: str) -> bool:
    if not isinstance(name, str):
        return False
    name_lower = name.lower()
    return any(t.lower() in name_lower for t in ELITE_TOURNAMENTS)


# Columns to never use as features
NON_FEATURES = [
    "player_name", "player_id", "season", "dg_rank",
    "top10", "top5", "win", "target",
]


def load_training_data(target: str = "top10") -> tuple:
    """
    Load historical stats and build training labels.

    Since we have season-level SG stats (not per-tournament results),
    we create training labels by ranking players within each season:
      - top10 : player ranked in top 10 by SG Total that season
      - top5  : player ranked in top 5
      - win   : player ranked #1 (best SG Total that season)

    This is a proxy for tournament performance — SG Total rank
    correlates strongly with actual win/top-10 rates on tour.
    """
    path = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No historical_stats.csv found at {path}\n"
            "Run fetch_data.py with fetch_historical_stats() enabled first."
        )

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} player-season rows from {path}")
    print(f"Columns: {list(df.columns)}")

    if "sg_total" not in df.columns:
        raise ValueError(
            "sg_total column not found. Check that fetch_data.py ran successfully."
        )

    # Build rank-based labels within each season
    df["sg_rank"] = df.groupby("season")["sg_total"].rank(ascending=False, method="min")
    season_size   = df.groupby("season")["sg_total"].transform("count")

    df["top10"] = (df["sg_rank"] <= 10).astype(int)
    df["top5"]  = (df["sg_rank"] <= 5).astype(int)
    df["win"]   = (df["sg_rank"] <= 1).astype(int)

    # Drop non-feature columns
    drop_cols = [c for c in NON_FEATURES if c in df.columns]
    drop_cols += ["sg_rank"]
    X = df.drop(columns=drop_cols + [target], errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target]

    print(f"\nFeature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target '{target}': {y.sum()} positive / {len(y)} total ({y.mean():.1%} rate)")
    print(f"Features: {list(X.columns)}")
    return X, y


def train_model(X: pd.DataFrame, y: pd.Series, target: str = "top10") -> tuple:
    """Train XGBoost with stratified cross-validation."""
    params = {k: v for k, v in XGB_PARAMS.items()
              if k not in ["use_label_encoder"]}

    model  = XGBClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    print(f"\nCross-validation AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    model.fit(X, y)
    return model, list(X.columns), scores.mean()


def save_model(model, feature_names: list, target: str, cv_score: float):
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgb_{target}.joblib")
    meta_path  = os.path.join(MODEL_OUTPUT_DIR, f"model_meta_{target}.json")

    joblib.dump(model, model_path)

    meta = {
        "target":        target,
        "cv_score":      round(cv_score, 4),
        "n_features":    len(feature_names),
        "feature_names": feature_names,
        "trained_on":    pd.Timestamp.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ Model saved    → {model_path}")
    print(f"✓ Metadata saved → {meta_path}")


def plot_feature_importance(model, feature_names: list, target: str, top_n: int = None):
    importances = model.feature_importances_
    top_n = min(top_n or len(feature_names), len(feature_names))
    sorted_idx  = np.argsort(importances)[::-1][:top_n]
    top_feats   = [feature_names[i] for i in sorted_idx]
    top_vals    = importances[sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), top_vals[::-1], color="#2563eb")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_feats[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features — {target} model")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(MODEL_OUTPUT_DIR, f"feature_importance_{target}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Feature importance chart → {plot_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=MODEL_TARGET,
                        choices=["top10", "top5", "win"])
    parser.add_argument("--mode", default="season",
                        choices=["season", "round"],
                        help="season: season-level stats; round: round-by-round data")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  PGA Predictor — Training [{args.target}] [{args.mode} mode]")
    print("=" * 60)

    if args.mode == "round":
        X, y = load_round_by_round_data(target=args.target)
    else:
        X, y = load_training_data(target=args.target)

    model, feature_names, cv_score = train_model(X, y, target=args.target)
    save_model(model, feature_names, args.target, cv_score)
    plot_feature_importance(model, feature_names, args.target)

    print(f"\n✅ Training complete! CV AUC: {cv_score:.4f}")
    
    print("Next: python predict.py")

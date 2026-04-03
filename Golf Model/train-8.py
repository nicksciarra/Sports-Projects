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

# Columns to never use as features
NON_FEATURES = [
    "player_name", "player_id", "season", "dg_rank",
    "top10", "top5", "win", "target",
]


def load_round_by_round_data(target: str = "top10") -> tuple:
    path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No round_by_round.csv found.\nRun: python fetch_rounds.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} player-tournament rows")

    # Drop rows missing the target
    df = df.dropna(subset=["finish_position", target]).reset_index(drop=True)
    print(f"Rows after dropna: {len(df)}")

    # Sort for time-based features
    df = df.sort_values(["player_name", "year", "tournamentName"]).reset_index(drop=True)

    # Recent form — last 5 tournaments (no leakage via shift)
    df["recent_avg_finish"] = (
        df.groupby("player_name")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["recent_top10_rate"] = (
        df.groupby("player_name")["top10"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Career stats per player using only prior years
    df["career_avg_finish"] = (
        df.groupby("player_name")["finish_position"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["career_top10_rate"] = (
        df.groupby("player_name")["top10"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df["career_starts"] = (
        df.groupby("player_name")["finish_position"]
        .transform(lambda x: x.shift(1).expanding().count())
    )

    # Bayesian shrinkage on career stats
    tour_avg_top10  = df["top10"].mean()
    tour_avg_finish = df["finish_position"].mean()
    k = 5
    w = df["career_starts"] / (df["career_starts"] + k)
    df["career_top10_rate"]  = w * df["career_top10_rate"].fillna(tour_avg_top10)  + (1 - w) * tour_avg_top10
    df["career_avg_finish"]  = w * df["career_avg_finish"].fillna(tour_avg_finish) + (1 - w) * tour_avg_finish

    # Fill SG nulls with tour average for that year (not 0)
    sg_cols = ["sg_total","sg_ott","sg_app","sg_arg","sg_putt",
               "sg_ballstriking","sg_shortgame","driving_dist",
               "driving_acc","gir_pct","scrambling_pct","putts_per_round","scoring_avg"]
    for col in sg_cols:
        if col in df.columns:
            # Fill with yearly tour average so missing != bad player
            df[col] = df.groupby("year")[col].transform(
                lambda x: x.fillna(x.mean())
            )
            # Any remaining nulls (whole year missing) fill with global mean
            df[col] = df[col].fillna(df[col].mean())

    # Add missingness indicator — was SG data available for this player-year?
    df["has_sg_data"] = (df["sg_total"] != 0).astype(int)

    # Drop non-feature columns
    drop_cols = [
        "tournament_id", "player_id", "player_name", "player_name_norm",
        "tournamentName", "courseName", "year", "season",
        "finish_position", "made_cut", "total_score", "total_par",
        "r1_score", "r2_score", "r3_score", "r4_score",
        "r1_par", "r2_par", "r3_par", "r4_par",
        "top10", "top5", "win",
    ]
    y = df[target]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns] + [target],
                errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
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

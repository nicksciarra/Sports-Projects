"""
train.py — Train XGBoost model on elite PGA Tour tournament data
---
Usage:
    python train.py
    python train.py --mode round --target top10
    python train.py --mode round --target top5
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
from xgboost import XGBClassifier

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

# ── Non-feature columns ───────────────────────────────────────────────────────
DROP_COLS = [
    "tournamentName", "courseName", "player_name", "player_name_norm",
    "year", "season", "finish_position", "made_cut", "total_score",
    "total_par", "r1_score", "r2_score", "r3_score", "r4_score",
    "r1_par", "r2_par", "r3_par", "r4_par",
    "top10", "top5", "win", "start", "end",
    "earnings", "fedex_points", "position", "score",
]


# ─────────────────────────────────────────────────────────────────────────────
# Season-level training (simple baseline)
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(target: str = "top10") -> tuple:
    path = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No historical_stats.csv found.\nRun fetch_data.py first."
        )

    df = pd.read_csv(path)
    print(f"Loaded {len(df)} player-season rows")

    if "sg_total" not in df.columns:
        raise ValueError("sg_total column not found.")

    df["sg_rank"]   = df.groupby("season")["sg_total"].rank(ascending=False, method="min")
    df["top10"]     = (df["sg_rank"] <= 10).astype(int)
    df["top5"]      = (df["sg_rank"] <= 5).astype(int)
    df["win"]       = (df["sg_rank"] <= 1).astype(int)

    non_feat = ["player_name", "season", "dg_rank", "sg_rank", "top10", "top5", "win"]
    X = df.drop(columns=[c for c in non_feat if c in df.columns] + [target], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target]

    print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Round-by-round training (elite events only)
# ─────────────────────────────────────────────────────────────────────────────

def load_round_by_round_data(target: str = "top10") -> tuple:
    path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No round_by_round.csv found.\nRun fetch_rounds.py or build_dataset.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} player-tournament rows")

    # Filter to elite tournaments only
    df = df[df["tournamentName"].apply(is_elite)].copy()
    print(f"Elite tournament rows: {len(df)}")

    df = df.dropna(subset=["finish_position", target]).reset_index(drop=True)
    print(f"Rows after dropna: {len(df)}")

    # Sort for time-based features
    df = df.sort_values(["player_name", "year", "tournamentName"]).reset_index(drop=True)

    # Recent form — last 5 tournaments (no leakage)
    df["recent_avg_finish"] = (
        df.groupby("player_name")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["recent_top10_rate"] = (
        df.groupby("player_name")["top10"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Career stats — expanding window (no leakage)
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
    df["career_top10_rate"] = (
        w * df["career_top10_rate"].fillna(tour_avg_top10) +
        (1 - w) * tour_avg_top10
    )
    df["career_avg_finish"] = (
        w * df["career_avg_finish"].fillna(tour_avg_finish) +
        (1 - w) * tour_avg_finish
    )

    # Fill SG nulls with yearly tour average (not 0)
    sg_cols = ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt",
               "sg_ballstriking", "sg_shortgame", "driving_dist",
               "driving_acc", "gir_pct", "scrambling_pct",
               "putts_per_round", "scoring_avg"]
    for col in sg_cols:
        if col in df.columns:
            df[col] = df.groupby("year")[col].transform(
                lambda x: x.fillna(x.mean())
            )
            df[col] = df[col].fillna(df[col].mean())

    df["has_sg_data"] = (df["sg_total"] != 0).astype(int)

    y = df[target]
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [target],
                errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)

    print(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"Target '{target}': {y.sum()} positive / {len(y)} total ({y.mean():.1%} rate)")
    print(f"Features: {list(X.columns)}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Train, evaluate, save
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X: pd.DataFrame, y: pd.Series, target: str = "top10") -> tuple:
    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    model  = XGBClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\nCV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
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


def plot_feature_importance(model, feature_names: list, target: str):
    importances = model.feature_importances_
    top_n       = min(15, len(feature_names))
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
    print(f"✓ Feature importance → {plot_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

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

"""
train.py — Train XGBoost model on elite PGA Tour tournament data
---
Usage:
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
    return any(t.lower() in name.lower() for t in ELITE_TOURNAMENTS)

SG_COLS = [
    "sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt",
    "sg_ballstriking", "sg_shortgame", "driving_dist",
    "driving_acc", "gir_pct", "scrambling_pct",
    "putts_per_round", "scoring_avg",
]

DROP_COLS = [
    "tournamentName", "courseName", "player_name", "player_name_norm",
    "year", "season", "finish_position", "made_cut", "total_score",
    "total_par", "r1_score", "r2_score", "r3_score", "r4_score",
    "r1_par", "r2_par", "r3_par", "r4_par",
    "top10", "top5", "win", "start", "end",
    "earnings", "fedex_points", "position", "score",
    "career_avg_finish_raw", "career_top10_raw",
]


# ─────────────────────────────────────────────────────────────────────────────
# Career stats computation
# ─────────────────────────────────────────────────────────────────────────────

def get_player_career_stats(df: pd.DataFrame, course_name: str = None) -> tuple:
    """Compute career stats per player from a dataframe of elite tournament results."""
    df = df[df["tournamentName"].apply(is_elite)].copy()
    df = df.sort_values(["player_name", "year", "tournamentName"]).reset_index(drop=True)

    tour_avg_top10  = df["top10"].mean()
    tour_avg_finish = df["finish_position"].mean()
    k = 5

    career = (
        df.groupby("player_name")
        .agg(
            career_starts=("finish_position", "count"),
            career_avg_finish_raw=("finish_position", "mean"),
            career_top10_raw=("top10", "mean"),
            recent_avg_finish=("finish_position", lambda x: x.tail(5).mean()),
            recent_top10_rate=("top10", lambda x: x.tail(5).mean()),
        )
        .reset_index()
    )

    w = career["career_starts"] / (career["career_starts"] + k)
    career["career_top10_rate"] = (
        w * career["career_top10_raw"] + (1 - w) * tour_avg_top10
    )
    career["career_avg_finish"] = (
        w * career["career_avg_finish_raw"] + (1 - w) * tour_avg_finish
    )

    # Course history
    if course_name:
        course_key  = course_name.split("(")[0].strip()
        course_mask = df["courseName"].str.contains(course_key, case=False, na=False)
        course_df   = df[course_mask]

        if len(course_df) > 0:
            ch = (
                course_df.groupby("player_name")
                .agg(
                    course_starts=("finish_position", "count"),
                    course_avg_finish_raw=("finish_position", "mean"),
                    course_top10_raw=("top10", "mean"),
                )
                .reset_index()
            )
            cw = ch["course_starts"] / (ch["course_starts"] + k)
            ch["course_top10_rate"] = cw * ch["course_top10_raw"] + (1 - cw) * tour_avg_top10
            ch["course_avg_finish"] = cw * ch["course_avg_finish_raw"] + (1 - cw) * tour_avg_finish
            ch = ch[["player_name", "course_starts", "course_top10_rate", "course_avg_finish"]]
            career = career.merge(ch, on="player_name", how="left")

    career["course_top10_rate"] = career.get("course_top10_rate", pd.Series()).fillna(tour_avg_top10)
    career["course_avg_finish"] = career.get("course_avg_finish", pd.Series()).fillna(tour_avg_finish)
    career["course_starts"]     = career.get("course_starts",     pd.Series()).fillna(0)

    # SG from most recent season per player
    sg_avail = [c for c in SG_COLS if c in df.columns]
    if sg_avail:
        sg = (
            df.sort_values("year")
            .groupby("player_name")[sg_avail]
            .last()
            .reset_index()
        )
        for col in sg_avail:
            sg[col] = sg[col].fillna(sg[col].mean())
        career = career.merge(sg, on="player_name", how="left")

    career["has_sg_data"] = (
        career["sg_total"].notna() & (career["sg_total"] != 0)
    ).astype(int)
    for col in sg_avail:
        career[col] = career[col].fillna(df[col].mean() if col in df.columns else 0)

    return career, tour_avg_top10, tour_avg_finish


# ─────────────────────────────────────────────────────────────────────────────
# Load and build training data
# ─────────────────────────────────────────────────────────────────────────────

def load_round_by_round_data(target: str = "top10") -> tuple:
    path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No round_by_round.csv. Run build_dataset.py first.")

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} rows")

    df = df[df["tournamentName"].apply(is_elite)].copy()
    print(f"Elite tournament rows: {len(df)}")
    df = df.dropna(subset=["finish_position", target]).reset_index(drop=True)

    # Compute career stats from all available data
    career, _, _ = get_player_career_stats(df)

    # Merge career stats onto each row
    merged = df.merge(career, on="player_name", how="left", suffixes=("", "_c"))

    # Fill SG with yearly tour average
    for col in SG_COLS:
        if col in merged.columns:
            merged[col] = merged.groupby("year")[col].transform(
                lambda x: x.fillna(x.mean())
            )
            merged[col] = merged[col].fillna(merged[col].mean())
    merged["has_sg_data"] = (merged["sg_total"] != 0).astype(int)

    y = merged[target].dropna()
    X = merged.drop(
        columns=[c for c in DROP_COLS if c in merged.columns] + [target],
        errors="ignore"
    ).select_dtypes(include=[np.number]).fillna(0)
    X = X.loc[y.index]

    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Target '{target}': {y.sum()} positive / {len(y)} total ({y.mean():.1%})")
    print(f"Features: {list(X.columns)}")
    return X, y


def load_training_data(target: str = "top10") -> tuple:
    """Season-level baseline model."""
    path = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("No historical_stats.csv. Run fetch_data.py first.")

    df = pd.read_csv(path)
    df["sg_rank"] = df.groupby("season")["sg_total"].rank(ascending=False, method="min")
    df["top10"]   = (df["sg_rank"] <= 10).astype(int)
    df["top5"]    = (df["sg_rank"] <= 5).astype(int)
    df["win"]     = (df["sg_rank"] <= 1).astype(int)

    non_feat = ["player_name", "season", "dg_rank", "sg_rank", "top10", "top5", "win"]
    X = df.drop(columns=[c for c in non_feat if c in df.columns] + [target], errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Train, save, plot
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X, y, target):
    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    model  = XGBClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\nCV AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
    model.fit(X, y)
    return model, list(X.columns), scores.mean()


def save_model(model, feature_names, target, cv_score):
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgb_{target}.joblib")
    meta_path  = os.path.join(MODEL_OUTPUT_DIR, f"model_meta_{target}.json")
    joblib.dump(model, model_path)
    meta = {
        "target": target, "cv_score": round(cv_score, 4),
        "n_features": len(feature_names), "feature_names": feature_names,
        "trained_on": pd.Timestamp.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n✓ Model saved    -> {model_path}")
    print(f"✓ Metadata saved -> {meta_path}")


def plot_feature_importance(model, feature_names, target):
    importances = model.feature_importances_
    top_n       = min(15, len(feature_names))
    sorted_idx  = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[sorted_idx][::-1], color="#2563eb")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Features - {target} model")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(MODEL_OUTPUT_DIR, f"feature_importance_{target}.png")
    plt.savefig(path, dpi=150)
    print(f"✓ Feature importance -> {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default=MODEL_TARGET, choices=["top10", "top5", "win"])
    parser.add_argument("--mode",   default="round",      choices=["season", "round"])
    args = parser.parse_args()

    print("=" * 60)
    print(f"  PGA Predictor - Training [{args.target}] [{args.mode} mode]")
    print("=" * 60)

    if args.mode == "round":
        X, y = load_round_by_round_data(target=args.target)
    else:
        X, y = load_training_data(target=args.target)

    model, feature_names, cv_score = train_model(X, y, args.target)
    save_model(model, feature_names, args.target, cv_score)
    plot_feature_importance(model, feature_names, args.target)

    print(f"\n✅ Training complete! CV AUC: {cv_score:.4f}")
    print("Next: python predict.py")

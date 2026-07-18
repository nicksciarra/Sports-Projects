"""
predict.py — Generate predictions for an upcoming PGA tournament
---
Usage:
    python predict.py
    python predict.py --id R2025014 --tournament "Masters Tournament" --course "Augusta National Golf Club"
    python predict.py --target top5
"""

import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_OUTPUT_DIR, PREDICTIONS_DIR, RAW_DATA_DIR
from train import get_player_career_stats, SG_COLS, DROP_COLS, is_elite
from fetch_field import fetch_and_save_field

os.makedirs(PREDICTIONS_DIR, exist_ok=True)


def load_model(target: str = "top10"):
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgb_{target}.joblib")
    meta_path  = os.path.join(MODEL_OUTPUT_DIR, f"model_meta_{target}.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No model at {model_path}. Run: python train.py --mode round first."
        )
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  ✓ Loaded [{target}] model - CV AUC: {meta['cv_score']}")
    return model, meta


def build_prediction_features(
    field_df: pd.DataFrame,
    course_name: str,
    feature_names: list,
) -> pd.DataFrame:
    """Build feature matrix for each player in the field."""

    # Load historical data to compute career stats
    rounds_path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    if not os.path.exists(rounds_path):
        raise FileNotFoundError("No round_by_round.csv. Run build_dataset.py first.")

    historical = pd.read_csv(rounds_path, low_memory=False)

    # Compute career stats from all historical data
    career, tour_avg_top10, tour_avg_finish = get_player_career_stats(
        historical, course_name
    )

    # Load current season SG stats
    rankings_path = os.path.join(RAW_DATA_DIR, "player_rankings.csv")
    if os.path.exists(rankings_path):
        rankings = pd.read_csv(rankings_path)
        rankings["player_name_norm"] = rankings["player_name"].str.lower().str.strip()
    else:
        rankings = pd.DataFrame()

    # Drop SG from career (will use current season stats instead)
    career_no_sg = career.drop(
        columns=[c for c in SG_COLS if c in career.columns], errors="ignore"
    )

    # Merge field with career stats
    field_df = field_df.copy()
    field_df["player_name_norm"] = field_df["player_name"].str.lower().str.strip()
    merged = field_df.merge(career_no_sg, on="player_name", how="left")

    # Merge current season SG stats
    if not rankings.empty:
        sg_cols_avail = [c for c in SG_COLS if c in rankings.columns]
        sg_merge = rankings[["player_name_norm"] + sg_cols_avail]
        merged = merged.merge(sg_merge, on="player_name_norm", how="left")

    # Fill missing SG with tour average
    historical_elite = historical[historical["tournamentName"].apply(is_elite)]
    for col in SG_COLS:
        if col in merged.columns:
            tour_mean = historical_elite[col].mean() if col in historical_elite.columns else 0
            merged[col] = merged[col].fillna(tour_mean)
        else:
            merged[col] = 0

    merged["has_sg_data"] = (merged["sg_total"] != 0).astype(int)

    # Fill career stats with tour averages for players with no history
    merged["career_top10_rate"] = merged.get("career_top10_rate", pd.Series()).fillna(tour_avg_top10)
    merged["career_avg_finish"] = merged.get("career_avg_finish", pd.Series()).fillna(tour_avg_finish)
    merged["career_starts"]     = merged.get("career_starts",     pd.Series()).fillna(0)
    merged["recent_avg_finish"] = merged.get("recent_avg_finish", pd.Series()).fillna(tour_avg_finish)
    merged["recent_top10_rate"] = merged.get("recent_top10_rate", pd.Series()).fillna(tour_avg_top10)
    merged["course_top10_rate"] = merged.get("course_top10_rate", pd.Series()).fillna(tour_avg_top10)
    merged["course_avg_finish"] = merged.get("course_avg_finish", pd.Series()).fillna(tour_avg_finish)
    merged["course_starts"]     = merged.get("course_starts",     pd.Series()).fillna(0)

    # Align to training feature set
    X = merged.drop(
        columns=[c for c in DROP_COLS + ["player_name", "player_name_norm", "player_id"]
                 if c in merged.columns],
        errors="ignore"
    ).select_dtypes(include=[np.number]).fillna(0)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]

    return merged, X


def predict_tournament(tournament_id, tournament_name, course_name, target="top10"):
    print(f"\nPredicting: {tournament_name} @ {course_name}")
    print("-" * 60)

    model, meta = load_model(target)
    feature_names = meta["feature_names"]

    print("Fetching tournament field...")
    field, _ = fetch_and_save_field(tournament_id)
    if field is None or field.empty:
        return pd.DataFrame(), tournament_name

    print("Building player features...")
    merged, X = build_prediction_features(field, course_name, feature_names)

    probs   = model.predict_proba(X)[:, 1]
    results = merged[["player_name"]].copy().reset_index(drop=True)
    results[f"p_{target}"] = probs

    for col in ["sg_total", "career_top10_rate", "course_starts",
                "career_avg_finish", "recent_avg_finish"]:
        if col in merged.columns:
            results[col] = merged[col].values

    results = results.sort_values(f"p_{target}", ascending=False).reset_index(drop=True)
    results.index += 1
    return results, tournament_name


def print_leaderboard(results, target, top_n=30):
    prob_col = f"p_{target}"
    print(f"\n{'─'*72}")
    print(f"  {'Rank':<5} {'Player':<25} {'P('+target+')':<10} {'SG:Total':<10} {'Course Hist'}")
    print(f"{'─'*72}")
    for rank, row in results.head(top_n).iterrows():
        sg   = f"{row['sg_total']:+.2f}" if "sg_total" in row and row["sg_total"] != 0 else "   -"
        hist = (f"{row['career_top10_rate']:.0%} top10 ({int(row.get('course_starts',0))} starts)"
                if row.get("course_starts", 0) > 0 else "No course history")
        print(f"  {rank:<5} {str(row['player_name']):<25} "
              f"{row[prob_col]:<10.1%} {sg:<10} {hist}")
    print(f"{'─'*72}\n")


def plot_predictions(results, tournament_name, target, top_n=25):
    prob_col = f"p_{target}"
    top      = results.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.3, len(top)))
    bars   = ax.barh(range(len(top)), top[prob_col].values, color=colors, height=0.7)

    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{i+1}. {n[:22]}" for i, n in enumerate(top["player_name"])],
        fontsize=9, color="white"
    )
    ax.invert_yaxis()
    ax.set_xlabel(f"P({target.upper()} Finish)", color="white", fontsize=11)
    ax.set_title(f"{tournament_name}\nPredicted {target.upper()} Probabilities",
                 color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_color("#334155")
    ax.set_xlim(0, top[prob_col].max() * 1.25)

    for bar, val in zip(bars, top[prob_col]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", color="white", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PREDICTIONS_DIR,
                        f"{tournament_name.replace(' ','_').lower()}_{target}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1923")
    print(f"  ✓ Chart saved -> {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", default="THE PLAYERS Championship")
    parser.add_argument("--id",         default="R2025011")
    parser.add_argument("--course",     default="TPC Sawgrass (THE PLAYERS Stadium Course)")
    parser.add_argument("--target",     default="top10", choices=["top10", "top5", "win"])
    args = parser.parse_args()

    results, name = predict_tournament(
        tournament_id=args.id,
        tournament_name=args.tournament,
        course_name=args.course,
        target=args.target,
    )

    if results is not None and not results.empty:
        print_leaderboard(results, args.target)
        plot_predictions(results, name, args.target)
        path = os.path.join(PREDICTIONS_DIR,
                            f"{name.replace(' ','_').lower()}_{args.target}.csv")
        results.to_csv(path)
        print(f"  ✓ Predictions saved -> {path}")
        print("\n✅ Done!")
    else:
        print("\n[INFO] Field not available yet. Try again Tuesday of tournament week.")

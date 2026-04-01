"""
backtest.py — Simulate predictions for a past tournament
---
Retrains the model using only data available before the target tournament,
then predicts that tournament and compares to actual results.

Usage:
    python backtest.py --tournament "THE PLAYERS Championship" --year 2024
    python backtest.py --tournament "Masters Tournament" --year 2024
    python backtest.py --tournament "THE PLAYERS Championship" --year 2023
"""

import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, MODEL_OUTPUT_DIR, PREDICTIONS_DIR, XGB_PARAMS

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

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



def load_and_split_data(tournament_name: str, year: int, target: str = "top10"):
    """
    Split round_by_round.csv into:
      - train: all data BEFORE the target tournament/year
      - test:  the target tournament itself
    """
    path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    df   = pd.read_csv(path, low_memory=False)

    # Find the target tournament rows
    name_mask = df["tournamentName"].str.contains(tournament_name, case=False, na=False)
    year_mask = df["year"] == year
    test_df   = df[name_mask & year_mask].copy()

    if test_df.empty:
        # Try partial match
        name_mask = df["tournamentName"].str.contains(
            tournament_name.split()[0], case=False, na=False
        )
        test_df = df[name_mask & year_mask].copy()

    if test_df.empty:
        print(f"[ERROR] Could not find '{tournament_name}' {year} in dataset.")
        print("Available tournaments that year:")
        print(df[df["year"] == year]["tournamentName"].unique()[:20])
        return None, None, None

    course_name = test_df["courseName"].iloc[0]
    print(f"Found: {test_df['tournamentName'].iloc[0]} @ {course_name}")
    print(f"Players in field: {len(test_df)}")
    print(f"Actual top-10: {test_df[test_df[target]==1]['player_name'].tolist()}")

    # Training data: elite tournaments only, before this tournament
    train_df = df[
        (
            (df["year"] < year) |
            ((df["year"] == year) & ~(name_mask))
        ) & df["tournamentName"].apply(is_elite)
    ].copy()

    print(f"\nTraining on {len(train_df)} rows from before this tournament")
    return train_df, test_df, course_name


def build_features(df: pd.DataFrame, course_name: str, is_train: bool = True) -> pd.DataFrame:
    """Build feature matrix with no data leakage."""
    df = df.sort_values(["player_name", "year", "tournamentName"]).reset_index(drop=True)

    # Recent form
    df["recent_avg_finish"] = (
        df.groupby("player_name")["finish_position"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df["recent_top10_rate"] = (
        df.groupby("player_name")["top10"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Career stats
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

    # Bayesian shrinkage
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

    # Course-specific history
    course_mask = df["courseName"].str.contains(
        course_name.split("(")[0].strip(), case=False, na=False
    )
    course_df = df[course_mask].copy()

    course_hist = (
        course_df.groupby("player_name")
        .agg(
            course_starts_raw=("finish_position", "count"),
            course_avg_finish_raw=("finish_position", "mean"),
            course_top10_raw=("top10", "mean"),
        )
        .reset_index()
    )
    course_hist["cw"] = course_hist["course_starts_raw"] / (course_hist["course_starts_raw"] + k)
    course_hist["course_top10_rate"] = (
        course_hist["cw"] * course_hist["course_top10_raw"] +
        (1 - course_hist["cw"]) * tour_avg_top10
    )
    course_hist["course_avg_finish"] = (
        course_hist["cw"] * course_hist["course_avg_finish_raw"] +
        (1 - course_hist["cw"]) * tour_avg_finish
    )
    course_hist = course_hist[["player_name", "course_starts_raw",
                                "course_top10_rate", "course_avg_finish"]]
    df = df.merge(course_hist, on="player_name", how="left")
    df["course_top10_rate"] = df["course_top10_rate"].fillna(tour_avg_top10)
    df["course_avg_finish"] = df["course_avg_finish"].fillna(tour_avg_finish)
    df["course_starts_raw"] = df["course_starts_raw"].fillna(0)

    # Fill SG nulls
    sg_cols = ["sg_total","sg_ott","sg_app","sg_arg","sg_putt",
               "sg_ballstriking","sg_shortgame","driving_dist",
               "driving_acc","gir_pct","scrambling_pct","putts_per_round","scoring_avg"]
    for col in sg_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def run_backtest(tournament_name: str, year: int, target: str = "top10"):
    print("=" * 60)
    print(f"  BACKTEST: {tournament_name} {year}")
    print("=" * 60)

    # Load and split data
    train_df, test_df, course_name = load_and_split_data(tournament_name, year, target)
    if train_df is None:
        return

    # Build features for training set
    print("\nBuilding training features...")
    train_feat = build_features(train_df, course_name, is_train=True)

    drop_cols = [
        "tournamentName", "courseName", "player_name", "player_name_norm",
        "year", "season", "finish_position", "made_cut", "total_score",
        "total_par", "r1_score", "r2_score", "r3_score", "r4_score",
        "r1_par", "r2_par", "r3_par", "r4_par",
        "top10", "top5", "win", "start", "end",
        "earnings", "fedex_points", "position", "score",
    ]

    y_train = train_feat[target].dropna()
    X_train = train_feat.drop(
        columns=[c for c in drop_cols if c in train_feat.columns] + [target],
        errors="ignore"
    ).select_dtypes(include=[np.number]).fillna(0)
    X_train = X_train.loc[y_train.index]

    print(f"Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")

    # Train model
    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    model  = XGBClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    model.fit(X_train, y_train)

    # Build features for test set
    # Combine train+test for feature computation, then extract test rows
    print("\nBuilding test features...")
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined_feat = build_features(combined, course_name, is_train=False)
    test_feat = combined_feat.iloc[-len(test_df):].copy()

    feature_names = list(X_train.columns)
    X_test = test_feat.drop(
        columns=[c for c in drop_cols if c in test_feat.columns] + [target],
        errors="ignore"
    ).select_dtypes(include=[np.number]).fillna(0)

    # Align columns
    for col in feature_names:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_names]

    # Predict
    probs = model.predict_proba(X_test)[:, 1]

    # Build results
    results = test_df[["player_name", "finish_position", target]].copy().reset_index(drop=True)
    results["predicted_prob"] = probs
    results = results.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    results.index += 1

    # Add SG for display
    if "sg_total" in test_feat.columns:
        results["sg_total"] = test_feat["sg_total"].values

    # Print leaderboard
    print(f"\n{'─'*72}")
    print(f"  {'Rank':<5} {'Player':<25} {'P('+target+')':<10} {'Actual Finish':<15} {'Hit?'}")
    print(f"{'─'*72}")
    for rank, row in results.head(30).iterrows():
        actual = int(row["finish_position"]) if row["finish_position"] < 999 else "CUT"
        hit    = "✓" if row[target] == 1 else ""
        print(f"  {rank:<5} {str(row['player_name']):<25} "
              f"{row['predicted_prob']:<10.1%} {str(actual):<15} {hit}")
    print(f"{'─'*72}")

    # Accuracy metrics
    top10_predicted = set(results.head(10)["player_name"])
    top10_actual    = set(results[results[target]==1]["player_name"])
    overlap         = top10_predicted & top10_actual
    precision       = len(overlap) / 10
    recall          = len(overlap) / max(len(top10_actual), 1)

    print(f"\nAccuracy summary:")
    print(f"  Predicted top-10 that actually finished top-10: {len(overlap)}/10")
    print(f"  Precision: {precision:.0%}  |  Recall: {recall:.0%}")
    print(f"\nActual top-10 players:")
    actual_top10 = results[results[target]==1].sort_values("finish_position")
    for _, row in actual_top10.iterrows():
        predicted_rank = results[results["player_name"]==row["player_name"]].index[0]
        print(f"  Actual #{int(row['finish_position']):<4} {row['player_name']:<25} "
              f"→ Model ranked #{predicted_rank}")

    # Save
    out_path = os.path.join(PREDICTIONS_DIR,
                            f"backtest_{tournament_name.replace(' ','_').lower()}_{year}_{target}.csv")
    results.to_csv(out_path)
    print(f"\n✓ Saved → {out_path}")

    # Plot
    plot_backtest(results, tournament_name, year, target)
    return results


def plot_backtest(results: pd.DataFrame, tournament_name: str, year: int, target: str):
    top30 = results.head(30)
    colors = ["#22c55e" if hit else "#94a3b8"
              for hit in top30[target].values]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    bars = ax.barh(range(len(top30)), top30["predicted_prob"].values[::-1],
                   color=colors[::-1], height=0.7)

    labels = [f"{i+1}. {name[:20]} (#{int(fin) if fin < 999 else 'CUT'})"
              for i, (name, fin) in enumerate(
                  zip(top30["player_name"], top30["finish_position"]))]

    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5, color="white")
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Probability", color="white")
    ax.set_title(f"🏌️  {tournament_name} {year} — Backtest\n"
                 f"Green = actual top-10 finish",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[["top","right","left","bottom"]].set_color("#334155")

    green_patch = mpatches.Patch(color="#22c55e", label="Actual top-10")
    grey_patch  = mpatches.Patch(color="#94a3b8", label="Outside top-10")
    ax.legend(handles=[green_patch, grey_patch], loc="lower right",
              facecolor="#1e293b", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(PREDICTIONS_DIR,
                        f"backtest_{tournament_name.replace(' ','_').lower()}_{year}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1923")
    print(f"✓ Chart saved → {path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", default="THE PLAYERS Championship")
    parser.add_argument("--year",       type=int, default=2024)
    parser.add_argument("--target",     default="top10",
                        choices=["top10", "top5", "win"])
    args = parser.parse_args()

    run_backtest(args.tournament, args.year, args.target)

"""
backtest.py — Simulate predictions for a past tournament
---
Usage:
    python backtest.py --tournament "Masters Tournament" --year 2025
    python backtest.py --tournament "THE PLAYERS Championship" --year 2024
    python backtest.py --tournament "U.S. Open" --year 2024 --target top5
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR, PREDICTIONS_DIR, XGB_PARAMS

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
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split_data(tournament_name: str, year: int, target: str = "top10"):
    path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    df   = pd.read_csv(path, low_memory=False)

    name_mask = df["tournamentName"].str.contains(tournament_name, case=False, na=False)
    year_mask = df["year"] == year
    test_df   = df[name_mask & year_mask].copy()

    if test_df.empty:
        name_mask = df["tournamentName"].str.contains(
            tournament_name.split()[0], case=False, na=False
        )
        test_df = df[name_mask & year_mask].copy()

    if test_df.empty:
        print(f"[ERROR] Could not find '{tournament_name}' {year}")
        print("Available tournaments:")
        print(df[df["year"] == year]["tournamentName"].unique()[:20])
        return None, None, None

    course_name = test_df["courseName"].iloc[0]
    print(f"Found: {test_df['tournamentName'].iloc[0]} @ {course_name}")
    print(f"Players in field: {len(test_df)}")
    print(f"Actual top-10: {test_df[test_df[target]==1]['player_name'].tolist()}")

    train_df = df[
        (df["year"] < year) |
        ((df["year"] == year) & ~name_mask)
    ].copy()

    print(f"Training on {len(train_df)} rows from before this tournament")
    return train_df, test_df, course_name


# ─────────────────────────────────────────────────────────────────────────────
# Career stats — computed from training data only
# ─────────────────────────────────────────────────────────────────────────────

def get_player_career_stats(train_df: pd.DataFrame, course_name: str) -> tuple:
    df = train_df[train_df["tournamentName"].apply(is_elite)].copy()
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
# Build feature matrices
# ─────────────────────────────────────────────────────────────────────────────

def build_train_features(train_df, career, target):
    train_elite  = train_df[train_df["tournamentName"].apply(is_elite)].copy()
    train_merged = train_elite.merge(career, on="player_name", how="left", suffixes=("", "_c"))

    for col in SG_COLS:
        if col in train_merged.columns:
            train_merged[col] = train_merged.groupby("year")[col].transform(
                lambda x: x.fillna(x.mean())
            )
            train_merged[col] = train_merged[col].fillna(train_merged[col].mean())
    train_merged["has_sg_data"] = (train_merged["sg_total"] != 0).astype(int)

    y = train_merged[target].dropna()
    X = train_merged.drop(
        columns=[c for c in DROP_COLS if c in train_merged.columns] + [target],
        errors="ignore"
    ).select_dtypes(include=[np.number]).fillna(0)
    X = X.loc[y.index]
    return X, y


def build_test_features(test_df, career, feature_names, target):
    career_no_sg = career.drop(
        columns=[c for c in SG_COLS if c in career.columns], errors="ignore"
    )
    test_merged = test_df.merge(career_no_sg, on="player_name", how="left")

    for col in SG_COLS:
        if col in test_merged.columns:
            test_merged[col] = test_merged[col].fillna(test_merged[col].mean())
    test_merged["has_sg_data"] = (test_merged["sg_total"] != 0).astype(int)

    X = test_merged.drop(
        columns=[c for c in DROP_COLS if c in test_merged.columns] + [target],
        errors="ignore"
    ).select_dtypes(include=[np.number]).fillna(0)

    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    return X


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(tournament_name: str, year: int, target: str = "top10"):
    print("=" * 60)
    print(f"  BACKTEST: {tournament_name} {year}")
    print("=" * 60)

    train_df, test_df, course_name = load_and_split_data(tournament_name, year, target)
    if train_df is None:
        return

    print("\nBuilding player career stats from training data...")
    career, tour_avg_top10, tour_avg_finish = get_player_career_stats(train_df, course_name)

    X_train, y_train = build_train_features(train_df, career, target)
    print(f"Training: {X_train.shape[0]} samples × {X_train.shape[1]} features")

    params = {k: v for k, v in XGB_PARAMS.items() if k != "use_label_encoder"}
    model  = XGBClassifier(**params)
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"CV AUC: {scores.mean():.4f} ± {scores.std():.4f}")
    model.fit(X_train, y_train)

    print("\nBuilding test features...")
    X_test = build_test_features(test_df, career, list(X_train.columns), target)

    probs   = model.predict_proba(X_test)[:, 1]
    results = test_df[["player_name", "finish_position", target]].copy().reset_index(drop=True)
    results["predicted_prob"] = probs
    results = results.sort_values("predicted_prob", ascending=False).reset_index(drop=True)
    results.index += 1

    # Print leaderboard
    print(f"\n{'─'*72}")
    print(f"  {'Rank':<5} {'Player':<25} {'P('+target+')':<10} {'Actual Finish':<15} {'Hit?'}")
    print(f"{'─'*72}")
    for rank, row in results.head(30).iterrows():
        actual = int(row["finish_position"]) if row["finish_position"] < 70 else "CUT"
        hit    = "✓" if row[target] == 1 else ""
        print(f"  {rank:<5} {str(row['player_name']):<25} "
              f"{row['predicted_prob']:<10.1%} {str(actual):<15} {hit}")
    print(f"{'─'*72}")

    # Accuracy
    top10_predicted = set(results.head(10)["player_name"])
    top10_actual    = set(results[results[target] == 1]["player_name"])
    overlap         = top10_predicted & top10_actual
    precision       = len(overlap) / 10
    recall          = len(overlap) / max(len(top10_actual), 1)

    print(f"\nAccuracy summary:")
    print(f"  Predicted top-10 that actually finished top-10: {len(overlap)}/10")
    print(f"  Precision: {precision:.0%}  |  Recall: {recall:.0%}")
    print(f"\nActual top-10 players:")
    for _, row in results[results[target] == 1].sort_values("finish_position").iterrows():
        rank = results[results["player_name"] == row["player_name"]].index[0]
        print(f"  Actual #{int(row['finish_position']):<4} {row['player_name']:<25} -> Model ranked #{rank}")

    # Save
    out_path = os.path.join(
        PREDICTIONS_DIR,
        f"backtest_{tournament_name.replace(' ','_').lower()}_{year}_{target}.csv"
    )
    results.to_csv(out_path)
    print(f"\n✓ Saved -> {out_path}")

    plot_backtest(results, tournament_name, year, target)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest(results: pd.DataFrame, tournament_name: str, year: int, target: str):
    top30  = results.head(30)
    colors = ["#22c55e" if hit else "#94a3b8" for hit in top30[target].values]

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    ax.barh(range(len(top30)), top30["predicted_prob"].values[::-1],
            color=colors[::-1], height=0.7)

    labels = [
        f"{i+1}. {name[:20]} (#{int(fin) if fin < 70 else 'CUT'})"
        for i, (name, fin) in enumerate(zip(top30["player_name"], top30["finish_position"]))
    ]
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(labels[::-1], fontsize=8.5, color="white")
    ax.invert_yaxis()
    ax.set_xlabel("Predicted Probability", color="white")
    ax.set_title(f"{tournament_name} {year} Backtest — Green = actual {target}",
                 color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[["top", "right", "left", "bottom"]].set_color("#334155")

    green_patch = mpatches.Patch(color="#22c55e", label=f"Actual {target}")
    grey_patch  = mpatches.Patch(color="#94a3b8", label="Did not hit")
    ax.legend(handles=[green_patch, grey_patch], loc="lower right",
              facecolor="#1e293b", labelcolor="white")

    plt.tight_layout()
    path = os.path.join(
        PREDICTIONS_DIR,
        f"backtest_{tournament_name.replace(' ','_').lower()}_{year}.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0f1923")
    print(f"✓ Chart saved -> {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", default="Masters Tournament")
    parser.add_argument("--year",       type=int, default=2025)
    parser.add_argument("--target",     default="top10",
                        choices=["top10", "top5", "win"])
    args = parser.parse_args()

    run_backtest(args.tournament, args.year, args.target)

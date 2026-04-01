"""
predict.py — Generate win probability predictions for an upcoming PGA tournament
---
Usage:
    python predict.py
    python predict.py --tournament "THE PLAYERS Championship" --course "TPC Sawgrass" --date 2026-03-12
    python predict.py --target top5
"""

import os
import sys
import json
import joblib
import argparse
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODEL_OUTPUT_DIR, PREDICTIONS_DIR, RAW_DATA_DIR
from fetch_field import fetch_and_save_field

os.makedirs(PREDICTIONS_DIR, exist_ok=True)

GRAPHQL_URL = "https://orchestrator.pgatour.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.pgatour.com/",
    "x-api-key": "da2-gsrx5bibzbb4njvhl7t37wqyl4",
}


# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(target: str = "top10"):
    model_path = os.path.join(MODEL_OUTPUT_DIR, f"xgb_{target}.joblib")
    meta_path  = os.path.join(MODEL_OUTPUT_DIR, f"model_meta_{target}.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model found at {model_path}\n"
            "Run: python train.py --mode round first."
        )

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"  ✓ Loaded [{target}] model — CV AUC: {meta['cv_score']}")
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# Fetch current tournament field
# ─────────────────────────────────────────────────────────────────────────────

def fetch_field(tournament_id: str) -> pd.DataFrame:
    """Fetch the current field for a tournament by ID."""
    query = f"""
    query {{
      field(id: "{tournament_id}") {{
        players {{
          id
          firstName
          lastName
          isAlternate
        }}
      }}
    }}
    """
    try:
        resp = requests.post(GRAPHQL_URL, json={"query": query},
                             headers=HEADERS, timeout=20)
        data = resp.json()
        players = data.get("data", {}).get("field", {}).get("players", []) or []

        rows = []
        for p in players:
            if p.get("isAlternate"):
                continue
            rows.append({
                "player_id":   str(p.get("id", "")),
                "player_name": f"{p.get('firstName','')} {p.get('lastName','')}".strip(),
            })

        df = pd.DataFrame(rows)
        print(f"  ✓ Fetched {len(df)} players in field")
        return df

    except Exception as e:
        print(f"  [WARN] Could not fetch field: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Build prediction features for each player in the field
# ─────────────────────────────────────────────────────────────────────────────

def build_prediction_features(
    field_df: pd.DataFrame,
    course_name: str,
    feature_names: list,
) -> pd.DataFrame:
    """
    For each player in the field, build their feature vector using:
      - Current season SG stats (from player_rankings.csv)
      - Historical results at this course (from round_by_round.csv)
      - Recent form (last 5 tournament finishes)
    """
    # Load current season stats
    rankings_path = os.path.join(RAW_DATA_DIR, "player_rankings.csv")
    if os.path.exists(rankings_path):
        rankings = pd.read_csv(rankings_path)
        rankings["player_name_norm"] = rankings["player_name"].str.lower().str.strip()
    else:
        print("  [WARN] player_rankings.csv not found — SG features will be 0")
        rankings = pd.DataFrame()

    # Load historical round results
    rounds_path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    if os.path.exists(rounds_path):
        rounds = pd.read_csv(rounds_path)
        rounds["player_name_norm"] = rounds["player_name"].str.lower().str.strip()
    else:
        print("  [WARN] round_by_round.csv not found — course history will be 0")
        rounds = pd.DataFrame()

    features = []

    for _, player in field_df.iterrows():
        name      = player["player_name"]
        name_norm = name.lower().strip()
        row       = {"player_name": name}

        # ── SG stats ──────────────────────────────────────────────────────────
        if not rankings.empty:
            match = rankings[rankings["player_name_norm"] == name_norm]
            if not match.empty:
                for col in ["sg_total", "sg_ott", "sg_app", "sg_arg", "sg_putt",
                            "driving_dist", "driving_acc", "gir_pct",
                            "scrambling_pct", "putts_per_round", "scoring_avg",
                            "sg_ballstriking", "sg_shortgame"]:
                    if col in match.columns:
                        row[col] = match.iloc[0][col]

        # ── Course history ────────────────────────────────────────────────────
        if not rounds.empty and course_name:
            course_mask = rounds["courseName"].str.contains(
                course_name, case=False, na=False
            )
            player_mask = rounds["player_name_norm"] == name_norm
            past = rounds[course_mask & player_mask]

            row["course_starts"]      = len(past)
            row["career_avg_finish"]  = past["finish_position"].mean() if len(past) > 0 else None
            row["career_best_finish"] = past["finish_position"].min()  if len(past) > 0 else None
            row["career_top10_rate"]  = past["top10"].mean()           if len(past) > 0 else None

        # ── Recent form — last 5 finishes ─────────────────────────────────────
        if not rounds.empty:
            player_rounds = (
                rounds[rounds["player_name_norm"] == name_norm]
                .sort_values("year", ascending=False)
                .head(5)
            )
            row["recent_avg_finish"] = player_rounds["finish_position"].mean() \
                                       if len(player_rounds) > 0 else None
            row["recent_top10_rate"] = player_rounds["top10"].mean() \
                                       if len(player_rounds) > 0 else None

        features.append(row)

    feat_df = pd.DataFrame(features)

    # Align to training feature set
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0

    X = feat_df[feature_names].fillna(0)
    return feat_df, X


# ─────────────────────────────────────────────────────────────────────────────
# Generate and display predictions
# ─────────────────────────────────────────────────────────────────────────────

def predict_tournament(
    tournament_id: str,
    tournament_name: str,
    course_name: str,
    target: str = "top10",
) -> pd.DataFrame:

    print(f"\nPredicting: {tournament_name} @ {course_name}")
    print("-" * 60)

    # Load model
    model, meta = load_model(target)
    feature_names = meta["feature_names"]

    # Fetch field
    print("Fetching tournament field...")
    field = fetch_field(tournament_id)

    if field.empty:
        print("[ERROR] Could not fetch field. Check tournament ID.")
        return pd.DataFrame()

    # Build features
    print("Building player features...")
    feat_df, X = build_prediction_features(field, course_name, feature_names)

    # Predict
    probs = model.predict_proba(X)[:, 1]

    results = feat_df[["player_name"]].copy()
    results[f"p_{target}"] = probs

    # Add key stats for display
    for col in ["sg_total", "sg_ott", "sg_app", "sg_putt",
                "career_top10_rate", "course_starts", "recent_avg_finish"]:
        if col in feat_df.columns:
            results[col] = feat_df[col].values

    results = results.sort_values(f"p_{target}", ascending=False).reset_index(drop=True)
    results.index += 1

    return results, tournament_name


def print_leaderboard(results: pd.DataFrame, target: str, top_n: int = 30):
    prob_col = f"p_{target}"
    print(f"\n{'─'*72}")
    print(f"  {'Rank':<5} {'Player':<25} {'P('+target+')':<10} {'SG:Total':<10} {'Course Hist'}")
    print(f"{'─'*72}")

    for rank, row in results.head(top_n).iterrows():
        sg    = f"{row['sg_total']:+.2f}" if "sg_total" in row and row["sg_total"] != 0 else "   —"
        hist  = (f"{row['career_top10_rate']:.0%} top10 ({int(row['course_starts'])} starts)"
                 if "course_starts" in row and row.get("course_starts", 0) > 0
                 else "No course history")
        print(f"  {rank:<5} {str(row['player_name']):<25} "
              f"{row[prob_col]:<10.1%} {sg:<10} {hist}")
    print(f"{'─'*72}\n")


def plot_predictions(results: pd.DataFrame, tournament_name: str, target: str, top_n: int = 25):
    prob_col = f"p_{target}"
    top      = results.head(top_n)

    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor("#0f1923")
    ax.set_facecolor("#0f1923")

    colors = plt.cm.RdYlGn(np.linspace(0.85, 0.3, len(top)))
    y_pos  = range(len(top))
    bars   = ax.barh(list(y_pos), top[prob_col].values, color=colors, height=0.7)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(
        [f"{i+1}. {name[:22]}" for i, name in enumerate(top["player_name"])],
        fontsize=9, color="white"
    )
    ax.invert_yaxis()
    ax.set_xlabel(f"P({target.upper()} Finish)", color="white", fontsize=11)
    ax.set_title(f"🏌️  {tournament_name}\nPredicted {target.upper()} Probabilities",
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
    print(f"  ✓ Chart saved → {path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament", default="THE PLAYERS Championship")
    parser.add_argument("--id",         default="R2025011",
                        help="PGA Tour tournament ID")
    parser.add_argument("--course",     default="TPC Sawgrass")
    parser.add_argument("--date",       default="2026-03-12")
    parser.add_argument("--target",     default="top10",
                        choices=["top10", "top5", "win"])
    args = parser.parse_args()

    results, name = predict_tournament(
        tournament_id=args.id,
        tournament_name=args.tournament,
        course_name=args.course,
        target=args.target,
    )

    if not results.empty:
        print_leaderboard(results, args.target)
        plot_predictions(results, name, args.target)

        path = os.path.join(PREDICTIONS_DIR,
                            f"{name.replace(' ','_').lower()}_{args.target}.csv")
        results.to_csv(path)
        print(f"  ✓ Predictions saved → {path}")
        print("\n✅ Done!")

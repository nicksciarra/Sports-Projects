"""
build_dataset.py — Merge Kaggle results with SG stats to build round_by_round.csv
---
Run once after downloading pga_results_2001-2025.csv to data/raw/

    python build_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# Load Kaggle results
# ─────────────────────────────────────────────────────────────────────────────

def load_kaggle_results(start_year: int = 2018) -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "pga_results_2001-2025.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find {path}")

    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df)} rows from Kaggle dataset")

    # Filter to recent seasons
    df = df[df["season"] >= start_year].copy()
    print(f"After filtering to {start_year}+: {len(df)} rows")

    # Parse finish position — strip T from ties, handle CUT/WD/DQ
    def parse_position(pos):
        pos = str(pos).strip()
        try:
            return int(pos.replace("T", "").replace("t", ""))
        except ValueError:
            return 70  # CUT/WD/DQ — treated as ~cut line finish

    df["finish_position"] = df["position"].apply(parse_position)
    df["made_cut"]        = (df["finish_position"] < 999).astype(int)
    df["top10"]           = (df["finish_position"] <= 10).astype(int)
    df["top5"]            = (df["finish_position"] <= 5).astype(int)
    df["win"]             = (df["finish_position"] == 1).astype(int)

    # Rename columns to match our schema
    df = df.rename(columns={
        "name":       "player_name",
        "tournament": "tournamentName",
        "location":   "courseName",
        "season":     "year",
        "round1":     "r1_score",
        "round2":     "r2_score",
        "round3":     "r3_score",
        "round4":     "r4_score",
        "score":      "total_par",
        "total":      "total_score",
    })

    # Normalize player names for merging
    df["player_name_norm"] = df["player_name"].str.lower().str.strip()

    # Extract just the course name from location (before the dash)
    df["courseName"] = df["courseName"].str.split(" - ").str[0].str.strip()

    keep_cols = [
        "year", "tournamentName", "courseName", "player_name", "player_name_norm",
        "finish_position", "made_cut", "top10", "top5", "win",
        "r1_score", "r2_score", "r3_score", "r4_score",
        "total_score", "total_par",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]

    print(f"Columns: {list(df.columns)}")
    print(f"\nSample:")
    print(df[["player_name","tournamentName","year","finish_position","top10"]].head(5).to_string())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Load SG stats
# ─────────────────────────────────────────────────────────────────────────────

def load_sg_stats() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
    if not os.path.exists(path):
        print("[WARN] historical_stats.csv not found — SG features will be missing")
        return pd.DataFrame()

    sg = pd.read_csv(path, low_memory=False)
    sg["player_name_norm"] = sg["player_name"].str.lower().str.strip()

    sg_cols = ["player_name_norm", "season", "sg_total", "sg_ott", "sg_app",
               "sg_arg", "sg_putt", "driving_dist", "driving_acc", "gir_pct",
               "scrambling_pct", "putts_per_round", "scoring_avg",
               "sg_ballstriking", "sg_shortgame"]
    sg = sg[[c for c in sg_cols if c in sg.columns]]
    sg = sg.rename(columns={"season": "year"})

    print(f"Loaded SG stats: {len(sg)} player-season rows")
    print(f"SG years: {sorted(sg['year'].unique())}")
    return sg


# ─────────────────────────────────────────────────────────────────────────────
# Merge and save
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(start_year: int = 2018):
    print("=" * 60)
    print("  Building round_by_round.csv from Kaggle + SG stats")
    print("=" * 60)

    results = load_kaggle_results(start_year)
    sg      = load_sg_stats()

    if not sg.empty:
        print(f"\nMerging SG stats...")

        # Check name overlap
        results_names = set(results["player_name_norm"].unique())
        sg_names      = set(sg["player_name_norm"].unique())
        overlap       = results_names & sg_names
        print(f"  Players in results:  {len(results_names)}")
        print(f"  Players in SG stats: {len(sg_names)}")
        print(f"  Name overlap:        {len(overlap)} ({len(overlap)/len(results_names):.0%})")

        merged = results.merge(sg, on=["player_name_norm", "year"], how="left")
        sg_fill_rate = merged["sg_total"].notna().mean()
        print(f"  SG fill rate after merge: {sg_fill_rate:.0%}")
    else:
        merged = results.copy()

    # Save
    out_path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(merged)} rows → {out_path}")
    print(f"  Columns: {list(merged.columns)}")

    # Quick sanity check — show THE PLAYERS winners by year
    players_df = merged[merged["tournamentName"].str.contains("PLAYERS", case=False, na=False)]
    if not players_df.empty:
        winners = (players_df[players_df["finish_position"] == 1]
                   .sort_values("year")[["year","player_name","sg_total"]]
                   .drop_duplicates("year"))
        print(f"\nTHE PLAYERS winners in dataset:")
        print(winners.to_string(index=False))

    return merged


if __name__ == "__main__":
    df = build_dataset(start_year=2018)
    print("\n✅ Done. Next: python train.py --mode round --target top10")

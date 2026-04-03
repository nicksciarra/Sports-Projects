"""
data/preprocess.py — Feature engineering for PGA Predictor
---
Builds a modelling-ready feature matrix from raw DataGolf data.

Feature groups:
  A. Strokes Gained (current form)       — SG:OTT, APP, ARG, PUTT, Total
  B. Rolling form windows                — 4-wk, 8-wk, 12-wk SG averages
  C. Course history                      — past scoring avg & top-10 rate
  D. Course fit (skill vs. course type)  — distance needed, green speed, rough
  E. Weather interaction terms           — wind × SG:OTT, rain × scrambling
  F. Physical stats                      — driving distance/accuracy
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR,
    FORM_WINDOW_WEEKS, COURSE_HISTORY_YEARS, MIN_COURSE_STARTS
)

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load raw data
# ─────────────────────────────────────────────────────────────────────────────

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load rankings/SG stats and historical results."""
    rankings_path = f"{RAW_DATA_DIR}player_rankings.csv"
    results_path  = f"{RAW_DATA_DIR}historical_results.csv"

    rankings = pd.read_csv(rankings_path) if os.path.exists(rankings_path) else pd.DataFrame()
    results  = pd.read_csv(results_path)  if os.path.exists(results_path)  else pd.DataFrame()

    if rankings.empty:
        print("[WARN] No rankings data. Run fetch_data.py first.")
    if results.empty:
        print("[WARN] No historical results. Run fetch_data.py first.")

    return rankings, results


# ─────────────────────────────────────────────────────────────────────────────
# A+B. Strokes Gained features (current season + rolling form)
# ─────────────────────────────────────────────────────────────────────────────

def build_sg_features(rankings: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and normalize SG features from DataGolf rankings payload.
    Expected columns (DataGolf naming):
      sg_putt, sg_arg, sg_app, sg_ott, sg_total,
      driving_dist, driving_acc, gir
    """
    sg_cols = [
        "dg_id", "player_name",
        "sg_putt", "sg_arg", "sg_app", "sg_ott", "sg_total",
        "driving_dist", "driving_acc",
        "dg_rank", "owgr_rank",
    ]

    # Keep only columns that exist
    available = [c for c in sg_cols if c in rankings.columns]
    df = rankings[available].copy()

    # Composite skills
    if all(c in df for c in ["sg_app", "sg_ott"]):
        df["sg_ballstriking"] = df["sg_app"] + df["sg_ott"]

    if all(c in df for c in ["sg_putt", "sg_arg"]):
        df["sg_shortgame"] = df["sg_putt"] + df["sg_arg"]

    # Rank-based features (lower rank = better)
    if "dg_rank" in df:
        df["rank_percentile"] = 1 - (df["dg_rank"].rank(pct=True))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# C. Course history features
# ─────────────────────────────────────────────────────────────────────────────

def build_course_history_features(
    results: pd.DataFrame,
    course_name: str,
) -> pd.DataFrame:
    """
    For a specific course, compute per-player historical stats:
      - n_starts           : sample size
      - avg_score_to_par   : scoring average relative to par
      - top10_rate         : fraction of starts finishing top-10
      - top25_rate         : fraction finishing top-25
      - made_cut_rate      : made cut fraction
      - best_finish        : best ever finish position
      - avg_sg_total_here  : avg SG total at this course
      - recency_weight_avg : exponentially-weighted recent scoring avg
    """
    if results.empty or "course" not in results.columns:
        return pd.DataFrame(columns=["dg_id", "n_course_starts",
                                      "course_avg_score", "course_top10_rate"])

    # Filter to this course
    course_df = results[results["course"].str.contains(course_name, case=False, na=False)].copy()

    if course_df.empty:
        print(f"  [INFO] No history found for course: {course_name}")
        return pd.DataFrame()

    # Aggregate per-player tournament finishes (not individual rounds)
    if "finish_position" not in course_df.columns:
        # Compute finish from round scores if available
        if "score" in course_df.columns:
            event_totals = (
                course_df.groupby(["dg_id", "player_name", "event_id", "season"])["score"]
                .sum()
                .reset_index()
                .rename(columns={"score": "total_score"})
            )
            event_totals["finish_position"] = (
                event_totals.groupby("event_id")["total_score"]
                .rank(method="min")
            )
        else:
            return pd.DataFrame()
    else:
        event_totals = course_df.drop_duplicates(
            subset=["dg_id", "event_id"]
        )[["dg_id", "player_name", "event_id", "season", "finish_position"]].copy()

    # Recency weighting — more recent = higher weight
    current_year = pd.Timestamp.now().year
    event_totals["years_ago"]     = current_year - event_totals["season"]
    event_totals["recency_weight"] = np.exp(-0.3 * event_totals["years_ago"])

    def player_course_stats(grp):
        n = len(grp)
        if n < MIN_COURSE_STARTS:
            return None

        fin = grp["finish_position"].values
        wts = grp["recency_weight"].values

        return pd.Series({
            "n_course_starts":       n,
            "course_top10_rate":     (fin <= 10).mean(),
            "course_top25_rate":     (fin <= 25).mean(),
            "course_made_cut_rate":  (fin <= 65).mean(),  # ~cut line
            "course_best_finish":    fin.min(),
            "course_avg_finish":     fin.mean(),
            "course_recent_avg_finish": np.average(fin, weights=wts),
        })

    hist = (
        event_totals
        .groupby("dg_id")
        .apply(player_course_stats)
        .dropna()
        .reset_index()
    )

    return hist


# ─────────────────────────────────────────────────────────────────────────────
# D+E. Weather interaction terms
# ─────────────────────────────────────────────────────────────────────────────

def build_weather_interactions(
    features: pd.DataFrame,
    weather: dict,
) -> pd.DataFrame:
    """
    Create interaction terms between player skills and weather conditions.
    Key insight: wind punishes poor SG:OTT players; rain rewards good
    scramblers; fast/firm conditions reward good putters.
    """
    df = features.copy()

    avg_wind = weather.get("avg_wind_mph", 10)
    is_wet   = weather.get("is_wet", 0)
    is_windy = weather.get("is_windy", 0)

    # Wind × driving: being long matters less in wind; accuracy matters more
    if "sg_ott" in df.columns:
        df["wind_x_sg_ott"] = df["sg_ott"] * avg_wind / 10.0

    if "driving_acc" in df.columns:
        df["wind_x_drv_acc"] = df["driving_acc"] * is_windy

    # Wet conditions → scrambling & GIR matter more (soft greens hold shots)
    if "sg_arg" in df.columns:
        df["wet_x_sg_arg"] = df["sg_arg"] * is_wet

    if "sg_app" in df.columns:
        df["wet_x_sg_app"] = df["sg_app"] * is_wet

    # Wet conditions → putting slightly less important (fewer 3-putts on soft greens)
    if "sg_putt" in df.columns:
        df["wet_x_sg_putt"] = df["sg_putt"] * (1 - is_wet * 0.3)

    # Add raw weather as features too
    for key in ["avg_wind_mph", "max_gust_mph", "total_precip_mm",
                "avg_temp_f", "is_windy", "is_wet"]:
        df[f"weather_{key}"] = weather.get(key, 0)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(
    course_name: str,
    weather: dict = None,
    field_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Returns a modelling-ready DataFrame with one row per player.

    Args:
        course_name : Name of the course (matches historical results)
        weather     : Dict from fetch_weather.fetch_tournament_weather()
        field_df    : Current tournament field (from fetch_field_and_projections)
    """
    rankings, results = load_raw_data()

    if rankings.empty:
        raise RuntimeError("Cannot build features: rankings data is missing.")

    # A+B: SG features
    print("Building SG features...")
    sg_feat = build_sg_features(rankings)

    # C: Course history
    print(f"Building course history features for '{course_name}'...")
    course_feat = build_course_history_features(results, course_name)

    # Merge SG + course history
    if not course_feat.empty:
        merged = sg_feat.merge(course_feat, on="dg_id", how="left")
    else:
        merged = sg_feat.copy()

    # Filter to current field if provided
    if field_df is not None and "dg_id" in field_df.columns:
        merged = merged[merged["dg_id"].isin(field_df["dg_id"])]

    # D+E: Weather interactions
    if weather:
        print("Building weather interaction features...")
        merged = build_weather_interactions(merged, weather)

    # Fill missing course history with neutral values
    course_cols = [c for c in merged.columns if c.startswith("course_")]
    for col in course_cols:
        if col == "course_best_finish":
            merged[col] = merged[col].fillna(50)
        elif col == "n_course_starts":
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = merged[col].fillna(merged[col].median())

    # Fill remaining NaN
    merged = merged.fillna(0)

    # Save
    out_path = f"{PROCESSED_DATA_DIR}features_{course_name.replace(' ', '_').lower()}.csv"
    merged.to_csv(out_path, index=False)
    print(f"  ✓ Feature matrix: {merged.shape[0]} players × {merged.shape[1]} features")
    print(f"  ✓ Saved → {out_path}")

    return merged


if __name__ == "__main__":
    from fetch_weather import fetch_tournament_weather

    weather = fetch_tournament_weather("TPC Sawgrass", "2025-03-13")

    features = build_feature_matrix(
        course_name="TPC Sawgrass",
        weather=weather,
    )

    print("\nFeature matrix preview:")
    print(features.head())
    print(f"\nFeatures: {list(features.columns)}")

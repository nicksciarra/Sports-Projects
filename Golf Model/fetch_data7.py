"""
fetch_data.py — Pull PGA Tour stats via their GraphQL API
---
✅ No API key required — uses the same endpoint as pgatour.com
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR

GRAPHQL_URL = "https://orchestrator.pgatour.com/graphql"

HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.pgatour.com/",
    "x-api-key": "da2-gsrx5bibzbb4njvhl7t37wqyl4",
}

STATS = {
    "02675": "sg_total",
    "02567": "sg_ott",
    "02568": "sg_app",
    "02569": "sg_arg",
    "02564": "sg_putt",
    "101":   "driving_dist",
    "102":   "driving_acc",
    "103":   "gir_pct",
    "130":   "scrambling_pct",
    "427":   "putts_per_round",
    "111":   "scoring_avg",
}

STAT_QUERY = """
query StatDetails($tourCode: TourCode!, $statId: String!, $year: Int) {
  statDetails(tourCode: $tourCode, statId: $statId, year: $year) {
    statId
    statTitle
    rows {
      ... on StatDetailsPlayer {
        playerId
        playerName
        rank
        stats {
          statName
          statValue
        }
      }
    }
  }
}
"""


def fetch_stat(stat_id: str, stat_name: str, year: int) -> pd.DataFrame:
    payload = {
        "operationName": "StatDetails",
        "query": STAT_QUERY,
        "variables": {
            "tourCode": "R",
            "statId": stat_id,
            "year": year,
        }
    }

    try:
        resp = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  [WARN] Network error for stat {stat_id}: {e}")
        return pd.DataFrame()

    # Show raw response for debugging
    raw_text = resp.text
    print(f"  [DEBUG] stat {stat_id} status={resp.status_code} response[:300]={raw_text[:300]}")

    try:
        data = resp.json()
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse JSON for stat {stat_id}")
        return pd.DataFrame()

    if data is None:
        print(f"  [WARN] data is None for stat {stat_id}")
        return pd.DataFrame()

    rows = (data.get("data") or {}).get("statDetails") or {}
    rows = rows.get("rows", [])

    if not rows:
        print(f"  [WARN] No rows for stat {stat_id}/{year}")
        return pd.DataFrame()

    records = []
    for row in rows:
        player_name = row.get("playerName")
        player_id   = row.get("playerId")
        stats_list  = row.get("stats", [])
        value = None
        for s in stats_list:
            val = s.get("statValue", "")
            try:
                value = float(str(val).replace("%", "").replace(",", "").strip())
                break
            except (ValueError, AttributeError):
                continue

        if player_name and value is not None:
            records.append({
                "player_name": player_name,
                stat_name:     value,
            })

    return pd.DataFrame(records) if records else pd.DataFrame()


def fetch_all_stats(year: int = 2025) -> pd.DataFrame:
    print(f"\nFetching PGA Tour stats for {year}...")
    merged = None

    for stat_id, stat_name in tqdm(STATS.items(), desc="  Pulling stats"):
        df = fetch_stat(stat_id, stat_name, year)
        if df.empty:
            continue
        merge_cols = ["player_name"]
        merged = df if merged is None else merged.merge(df, on=merge_cols, how="outer")
        time.sleep(0.5)

    if merged is None or merged.empty:
        print(f"  [ERROR] No stats fetched for {year}.")
        return pd.DataFrame()

    merged["season"] = year
    if "sg_ott" in merged.columns and "sg_app" in merged.columns:
        merged["sg_ballstriking"] = merged["sg_ott"].fillna(0) + merged["sg_app"].fillna(0)
    if "sg_putt" in merged.columns and "sg_arg" in merged.columns:
        merged["sg_shortgame"] = merged["sg_putt"].fillna(0) + merged["sg_arg"].fillna(0)

    print(f"  ✓ {len(merged)} players, {len(merged.columns)} features")
    return merged


def fetch_player_rankings(year: int = 2025) -> pd.DataFrame:
    df = fetch_all_stats(year)
    if df.empty:
        return df
    if "sg_total" in df.columns:
        df = df.sort_values("sg_total", ascending=False).reset_index(drop=True)
        df["dg_rank"] = df.index + 1
    path = os.path.join(RAW_DATA_DIR, "player_rankings.csv")
    df.to_csv(path, index=False)
    print(f"✓ Saved {len(df)} players → {path}")
    return df


def fetch_historical_stats(start_year: int = 2020, end_year: int = 2025) -> pd.DataFrame:
    all_seasons = []
    for year in range(start_year, end_year + 1):
        df = fetch_all_stats(year)
        if not df.empty:
            all_seasons.append(df)
        time.sleep(1.0)
    if not all_seasons:
        return pd.DataFrame()
    combined = pd.concat(all_seasons, ignore_index=True)
    path = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
    combined.to_csv(path, index=False)
    print(f"\n✓ Historical stats ({start_year}–{end_year}) → {path}")
    return combined


def fetch_field_and_projections() -> pd.DataFrame:
    return fetch_player_rankings()


if __name__ == "__main__":
    print("=" * 60)
    print("  PGA Predictor — Fetching via PGA Tour GraphQL API")
    print("=" * 60)

    current = fetch_player_rankings(year=2025)

    if not current.empty:
        cols = [c for c in ["player_name", "sg_total", "sg_ott", "sg_app", "sg_putt"]
                if c in current.columns]
        print(f"\nTop 10 by SG Total:")
        print(current[cols].head(10).to_string(index=False))
    else:
        print("\n[ERROR] Could not fetch stats. Paste the [DEBUG] lines above for help.")

    fetch_historical_stats(start_year=2020, end_year=2025)

    
    print("\n✅ Done.")

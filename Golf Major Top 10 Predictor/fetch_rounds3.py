"""
fetch_rounds.py — Fetch round-by-round PGA Tour tournament results
---
✅ No API key required — uses PGA Tour GraphQL API
✅ Fetches every tournament result for a given season
✅ Saves one row per player per tournament with:
     - finish position
     - round scores (R1-R4)
     - total score
     - tournament & course info

Run standalone:
    python fetch_rounds.py
    python fetch_rounds.py --start_year 2020 --end_year 2024
"""

import os
import sys
import time
import json
import argparse
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


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Get all tournament IDs for a season
# ─────────────────────────────────────────────────────────────────────────────

SCHEDULE_QUERY = """
query {
  schedule(tourCode: "R", year: "%s") {
    completed {
      tournaments {
        id
        tournamentName
        courseName
        startDate
      }
    }
  }
}
"""

def fetch_schedule(year: int) -> list[dict]:
    """Return list of {id, tournamentName, courseName, startDate} for a season."""
    payload = {"query": SCHEDULE_QUERY % year}
    try:
        resp = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [WARN] Schedule fetch failed for {year}: {e}")
        return []

    completed = data.get("data", {}).get("schedule", {}).get("completed", [])
    tournaments = []
    for group in completed:
        for t in group.get("tournaments", []):
            tournaments.append({
                "id":             t.get("id"),
                "tournamentName": t.get("tournamentName"),
                "courseName":     t.get("courseName"),
                "startDate":      t.get("startDate"),
                "year":           year,
            })
    return tournaments


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Fetch past results for one tournament
# ─────────────────────────────────────────────────────────────────────────────

PAST_RESULTS_QUERY = """
query {
  tournamentPastResults(id: "%s", year: %d) {
    id
    players {
      id
      position
      player {
        id
        firstName
        lastName
      }
      rounds {
        score
        parRelativeScore
      }
      total
      parRelativeScore
      additionalData
    }
  }
}
"""

def fetch_tournament_results(tournament_id: str, year: int) -> list[dict]:
    """Fetch all player results for one tournament. Returns list of row dicts."""
    payload = {"query": PAST_RESULTS_QUERY % (tournament_id, year)}
    try:
        resp = requests.post(GRAPHQL_URL, json=payload, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    [WARN] Results fetch failed for {tournament_id}: {e}")
        return []

    players = (
        data.get("data", {})
            .get("tournamentPastResults", {})
            .get("players", []) or []
    )

    rows = []
    for p in players:
        player_info = p.get("player") or {}
        rounds      = p.get("rounds") or []

        # Parse finish position — strip T from tied positions e.g. "T5" -> 5
        raw_pos = str(p.get("position", "") or "")
        try:
            finish = int(raw_pos.replace("T", "").replace("CUT", "999")
                                .replace("WD", "999").replace("DQ", "999").strip())
        except ValueError:
            finish = 999

        row = {
            "tournament_id":   tournament_id,
            "player_id":       player_info.get("id", p.get("id")),
            "player_name":     f"{player_info.get('firstName', '')} {player_info.get('lastName', '')}".strip(),
            "finish_position": finish,
            "made_cut":        int(finish < 999),
            "total_score":     p.get("total"),
            "total_par":       p.get("parRelativeScore"),
        }

        # Round scores
        for i, rnd in enumerate(rounds[:4], 1):
            row[f"r{i}_score"] = rnd.get("score")
            row[f"r{i}_par"]   = rnd.get("parRelativeScore")

        # Fill missing rounds
        for i in range(len(rounds) + 1, 5):
            row[f"r{i}_score"] = None
            row[f"r{i}_par"]   = None

        rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Build labels and merge with season SG stats
# ─────────────────────────────────────────────────────────────────────────────

def build_round_by_round_dataset(
    start_year: int = 2020,
    end_year:   int = 2024,
) -> pd.DataFrame:
    """
    For each season:
      1. Get all tournament IDs
      2. Fetch results for each tournament
      3. Merge with that season's SG stats
      4. Compute top-10/top-5/win labels per tournament
    Returns a combined DataFrame ready for train.py
    """
    sg_path = os.path.join(RAW_DATA_DIR, "historical_stats.csv")
    if not os.path.exists(sg_path):
        print("[WARN] historical_stats.csv not found — results will lack SG features.")
        print("       Run fetch_data.py with fetch_historical_stats() first.")
        sg_by_season = {}
    else:
        sg_all = pd.read_csv(sg_path)
        # Normalize player names for merging
        sg_all["player_name_norm"] = (
            sg_all["player_name"].str.lower().str.strip()
        )
        sg_by_season = {yr: grp for yr, grp in sg_all.groupby("season")}

    all_records = []

    for year in range(start_year, end_year + 1):
        print(f"\n── Season {year} ─────────────────────────────")
        tournaments = fetch_schedule(year)
        print(f"  Found {len(tournaments)} tournaments")

        sg_season = sg_by_season.get(year, pd.DataFrame())

        for t in tqdm(tournaments, desc=f"  {year} tournaments"):
            tid  = t["id"]
            name = t["tournamentName"]
            course = t["courseName"]

            results = fetch_tournament_results(tid, year)
            if not results:
                continue

            for row in results:
                row["tournamentName"] = name
                row["courseName"]     = course
                row["year"]           = year
                # Labels
                fp = row["finish_position"]
                row["top10"] = int(fp <= 10)
                row["top5"]  = int(fp <= 5)
                row["win"]   = int(fp == 1)

            all_records.extend(results)
            time.sleep(0.3)  # polite delay

    if not all_records:
        print("[ERROR] No results fetched.")
        return pd.DataFrame()

    results_df = pd.DataFrame(all_records)
    # Build player ID -> name lookup
    print("\nFetching player names by ID...")
    unique_ids = results_df["player_id"].dropna().unique()
    id_to_name = {}

    for pid in tqdm(unique_ids, desc="  Player names"):
        query = f"""
        query {{
          player(id: "{pid}") {{
            id
            firstName
            lastName
          }}
        }}
        """
        try:
            resp = requests.post(GRAPHQL_URL, json={"query": query},
                                 headers=HEADERS, timeout=10)
            p = resp.json().get("data", {}).get("player") or {}
            first = p.get("firstName", "")
            last  = p.get("lastName", "")
            if first or last:
                id_to_name[str(pid)] = f"{first} {last}".strip()
        except Exception:
            pass
        time.sleep(0.2)

    results_df["player_id"] = results_df["player_id"].astype(str)
    results_df["player_name"] = results_df["player_id"].map(id_to_name)
    print(f"  ✓ Resolved {len(id_to_name)} player names")

    results_df["player_name_norm"] = results_df["player_name"].str.lower().str.strip()

    # Merge SG stats by player + year
    if not sg_season.empty if sg_by_season else False:
        # Rename season -> year for merging
        sg_all["year"] = sg_all["season"]
        results_df = results_df.merge(
            sg_all.drop(columns=["player_name", "season"], errors="ignore"),
            on=["player_name_norm", "year"],
            how="left",
        )

    # Save
    out_path = os.path.join(RAW_DATA_DIR, "round_by_round.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(results_df)} player-tournament rows → {out_path}")
    print(f"  Columns: {list(results_df.columns)}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2020)
    parser.add_argument("--end_year",   type=int, default=2025)
    args = parser.parse_args()

    print("=" * 60)
    print("  PGA Predictor — Fetching Round-by-Round Results")
    print(f"  Seasons: {args.start_year} – {args.end_year}")
    print("=" * 60)
    print("Note: This will take several minutes (one API call per tournament)")

    df = build_round_by_round_dataset(
        start_year=args.start_year,
        end_year=args.end_year,
    )

    if not df.empty:
        print(f"\nSample data:")
        print(df[["player_name", "tournamentName", "finish_position",
                   "top10", "r1_score", "r2_score", "r3_score", "r4_score"]
                 ].head(10).to_string(index=False))
        print(f"\n✅ Done. Next: python train.py --mode round")

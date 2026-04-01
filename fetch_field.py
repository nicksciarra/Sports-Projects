"""
fetch_field.py — Fetch the field for the next upcoming PGA Tour tournament
---
✅ No API key required
✅ Run Tuesday/Wednesday of tournament week for best results
✅ Falls back to top-N ranked players if field not yet posted

Usage:
    python fetch_field.py                    # fetch next tournament field
    python fetch_field.py --id R2025014      # fetch specific tournament field
"""

import os
import sys
import json
import requests
import argparse
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_DATA_DIR

GRAPHQL_URL = "https://orchestrator.pgatour.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.pgatour.com/",
    "x-api-key": "da2-gsrx5bibzbb4njvhl7t37wqyl4",
}


def get_next_tournament() -> dict:
    """Find the next upcoming PGA Tour tournament from the schedule."""
    # Try current and next season
    for year in ["2025", "2026"]:
        query = f"""
        query {{
          schedule(tourCode: "R", year: "{year}") {{
            upcoming {{
              tournaments {{
                id
                tournamentName
                courseName
                startDate
              }}
            }}
          }}
        }}
        """
        try:
            resp = requests.post(GRAPHQL_URL, json={"query": query},
                                 headers=HEADERS, timeout=20)
            data = resp.json()
            upcoming = (data.get("data", {})
                            .get("schedule", {})
                            .get("upcoming", []))

            tournaments = []
            for group in upcoming:
                tournaments.extend(group.get("tournaments", []))

            if tournaments:
                # Sort by start date and return the soonest
                tournaments.sort(key=lambda t: t.get("startDate", 0))
                t = tournaments[0]
                print(f"  Next tournament: {t['tournamentName']}")
                print(f"  Course: {t['courseName']}")
                print(f"  ID: {t['id']}")
                return t

        except Exception as e:
            print(f"  [WARN] Schedule fetch failed for {year}: {e}")

    return {}


def fetch_field_by_id(tournament_id: str) -> pd.DataFrame:
    """Fetch official field for a tournament by ID."""
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
        players = (data.get("data") or {}).get("field") or {}
        players = players.get("players") or []

        if not players:
            return pd.DataFrame()

        rows = []
        for p in players:
            if p.get("isAlternate"):
                continue
            name = f"{p.get('firstName','')} {p.get('lastName','')}".strip()
            if name:
                rows.append({
                    "player_id":   str(p.get("id", "")),
                    "player_name": name,
                })

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"  [WARN] Field fetch failed: {e}")
        return pd.DataFrame()


def fetch_field_from_rankings(top_n: int = 150) -> pd.DataFrame:
    """Fallback — use top N players from current rankings as proxy field."""
    path = os.path.join(RAW_DATA_DIR, "player_rankings.csv")
    if not os.path.exists(path):
        print("[ERROR] player_rankings.csv not found. Run fetch_data.py first.")
        return pd.DataFrame()

    rankings = pd.read_csv(path)
    if "sg_total" in rankings.columns:
        rankings = rankings.sort_values("sg_total", ascending=False)

    field = rankings.head(top_n)[["player_name"]].copy()
    field["player_id"] = ""
    print(f"  Using top {len(field)} players from rankings as proxy field")
    return field


def fetch_and_save_field(tournament_id: str = None, top_n: int = 150) -> tuple:
    """
    Main function — fetch field for upcoming tournament.
    Returns (field_df, tournament_info_dict)
    """
    print("Fetching tournament field...")

    # Get tournament info
    if tournament_id:
        t_info = {"id": tournament_id, "tournamentName": "Unknown", "courseName": "Unknown"}
    else:
        t_info = get_next_tournament()
        if not t_info:
            print("[WARN] Could not find upcoming tournament")
            t_info = {"id": "", "tournamentName": "Unknown", "courseName": "Unknown"}
        tournament_id = t_info.get("id", "")

    # Try official field first
    field = pd.DataFrame()
    if tournament_id:
        field = fetch_field_by_id(tournament_id)

    if not field.empty:
        print(f"  ✓ Official field: {len(field)} players")
        source = "official"
    else:
        print("  Field not yet posted — using rankings proxy")
        field = fetch_field_from_rankings(top_n)
        source = "proxy"

    # Save field
    t_name = t_info.get("tournamentName", "unknown").replace(" ", "_").lower()
    path   = os.path.join(RAW_DATA_DIR, f"field_{t_name}.csv")
    field.to_csv(path, index=False)
    print(f"  ✓ Saved {len(field)} players → {path}")

    # Save tournament info
    info_path = os.path.join(RAW_DATA_DIR, "next_tournament.json")
    with open(info_path, "w") as f:
        json.dump({**t_info, "field_source": source, "field_path": path}, f, indent=2)
    print(f"  ✓ Tournament info → {info_path}")

    return field, t_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=None,
                        help="Tournament ID e.g. R2025014 (Masters)")
    parser.add_argument("--top_n", type=int, default=150,
                        help="Number of players to use as proxy if field not posted")
    args = parser.parse_args()

    print("=" * 60)
    print("  PGA Predictor — Field Fetcher")
    print("=" * 60)

    field, t_info = fetch_and_save_field(args.id, args.top_n)

    if not field.empty:
        print(f"\nField preview:")
        print(field.head(10).to_string(index=False))
        print(f"\n✅ Done. Now run:")
        print(f"   python predict.py --id {t_info.get('id','')} "
              f"--tournament \"{t_info.get('tournamentName','')}\" "
              f"--course \"{t_info.get('courseName','')}\"")

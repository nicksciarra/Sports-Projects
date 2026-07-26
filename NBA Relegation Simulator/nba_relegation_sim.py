"""
NBA Promotion/Relegation Monte Carlo Simulation
================================================
Simulates future NBA seasons under a proposed reform where:
  - The G League is trimmed to 18 independent franchises
  - Each season: bottom 4 NBA teams are relegated to G League
  - Each season: top 4 G League teams are promoted to NBA
  - Promotion triggers a talent boost (from absorbed G League player pool)
  - Relegation triggers a talent penalty

Usage:
    python nba_relegation_sim.py

    Adjust CONFIG below to change seasons, runs, variance, and output options.
"""

import random
import math
import json
import csv
from collections import defaultdict
from copy import deepcopy


# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "start_year":        2027,
    "seasons":           10,       # number of seasons to simulate
    "mc_runs":           1000,     # Monte Carlo iterations
    "variance":          0.04,     # Gaussian noise std dev on win rate per season
    "gl_boost_pct":      0.08,     # win-rate boost for promoted G League teams
    "rel_penalty_pct":   0.12,     # win-rate penalty for relegated NBA teams
    "promotions_per_season": 4,
    "relegations_per_season": 4,
    "print_sample_run":  True,     # print one sample season-by-season trace
    "save_csv":          True,     # save relegation/promotion risk to CSV
    "save_json":         True,     # save full MC results to JSON
    "csv_path":          "nba_relegation_risk.csv",
    "json_path":         "nba_relegation_results.json",
}
# ──────────────────────────────────────────────────────────────────────────────


# ─── INITIAL ROSTERS (seeded from 2025-26 actual standings) ──────────────────

NBA_TEAMS = [
    {"name": "Oklahoma City Thunder",    "win_rate": 0.780},
    {"name": "San Antonio Spurs",        "win_rate": 0.756},
    {"name": "Detroit Pistons",          "win_rate": 0.732},
    {"name": "Boston Celtics",           "win_rate": 0.683},
    {"name": "Denver Nuggets",           "win_rate": 0.659},
    {"name": "New York Knicks",          "win_rate": 0.646},
    {"name": "Los Angeles Lakers",       "win_rate": 0.646},
    {"name": "Houston Rockets",          "win_rate": 0.634},
    {"name": "Cleveland Cavaliers",      "win_rate": 0.634},
    {"name": "Minnesota Timberwolves",   "win_rate": 0.598},
    {"name": "Atlanta Hawks",            "win_rate": 0.561},
    {"name": "Toronto Raptors",          "win_rate": 0.561},
    {"name": "Philadelphia 76ers",       "win_rate": 0.549},
    {"name": "Phoenix Suns",             "win_rate": 0.549},
    {"name": "Orlando Magic",            "win_rate": 0.549},
    {"name": "Charlotte Hornets",        "win_rate": 0.537},
    {"name": "Miami Heat",               "win_rate": 0.524},
    {"name": "LA Clippers",              "win_rate": 0.512},
    {"name": "Portland Trail Blazers",   "win_rate": 0.512},
    {"name": "Golden State Warriors",    "win_rate": 0.451},
    {"name": "Milwaukee Bucks",          "win_rate": 0.390},
    {"name": "Chicago Bulls",            "win_rate": 0.378},
    {"name": "New Orleans Pelicans",     "win_rate": 0.317},
    {"name": "Dallas Mavericks",         "win_rate": 0.317},
    {"name": "Memphis Grizzlies",        "win_rate": 0.305},
    {"name": "Sacramento Kings",         "win_rate": 0.268},
    {"name": "Utah Jazz",                "win_rate": 0.268},
    {"name": "Brooklyn Nets",            "win_rate": 0.244},
    {"name": "Indiana Pacers",           "win_rate": 0.232},
    {"name": "Washington Wizards",       "win_rate": 0.207},
]

G_LEAGUE_TEAMS = [
    {"name": "G: Ignite",          "win_rate": 0.72},
    {"name": "G: Motor City",      "win_rate": 0.68},
    {"name": "G: Salt Lake City",  "win_rate": 0.65},
    {"name": "G: South Bay",       "win_rate": 0.63},
    {"name": "G: Westchester",     "win_rate": 0.60},
    {"name": "G: Fort Wayne",      "win_rate": 0.58},
    {"name": "G: Long Island",     "win_rate": 0.55},
    {"name": "G: Windy City",      "win_rate": 0.52},
    {"name": "G: Capital City",    "win_rate": 0.50},
    {"name": "G: Grand Rapids",    "win_rate": 0.48},
    {"name": "G: Memphis",         "win_rate": 0.46},
    {"name": "G: Birmingham",      "win_rate": 0.44},
    {"name": "G: Texas",           "win_rate": 0.42},
    {"name": "G: Maine",           "win_rate": 0.40},
    {"name": "G: Greensboro",      "win_rate": 0.38},
    {"name": "G: Lakeland",        "win_rate": 0.36},
    {"name": "G: Santa Cruz",      "win_rate": 0.34},
    {"name": "G: Canton",          "win_rate": 0.30},
]


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def gauss_noise(win_rate: float, std: float) -> float:
    """Apply Gaussian noise to a win rate, clamped to [0.05, 0.95]."""
    return max(0.05, min(0.95, win_rate + random.gauss(0, std)))


def simulate_season(
    teams: list[dict],
    cfg: dict,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Run one season. Returns:
      - updated team list with new win_rates and league assignments
      - list of relegated team names
      - list of promoted team names
    """
    n_rel = cfg["relegations_per_season"]
    n_prom = cfg["promotions_per_season"]
    variance = cfg["variance"]
    gl_boost = cfg["gl_boost_pct"]
    rel_penalty = cfg["rel_penalty_pct"]

    nba = [t for t in teams if t["league"] == "NBA"]
    g   = [t for t in teams if t["league"] == "G"]

    # Rank by noisy win rate to determine season finish
    nba_ranked = sorted(nba, key=lambda t: gauss_noise(t["win_rate"], variance), reverse=True)
    g_ranked   = sorted(g,   key=lambda t: gauss_noise(t["win_rate"], variance), reverse=True)

    relegated_names = {t["name"] for t in nba_ranked[-n_rel:]}
    promoted_names  = {t["name"] for t in g_ranked[:n_prom]}

    next_teams = []
    for t in teams:
        nt = deepcopy(t)
        if t["name"] in relegated_names:
            nt["league"]    = "G"
            nt["win_rate"]  = max(0.15, t["win_rate"] * (1 - rel_penalty))
            nt["status"]    = "relegated"
        elif t["name"] in promoted_names:
            nt["league"]    = "NBA"
            nt["win_rate"]  = min(0.85, t["win_rate"] * (1 + gl_boost))
            nt["status"]    = "promoted"
        else:
            # Organic drift each season
            nt["win_rate"]  = max(0.10, min(0.88, t["win_rate"] + random.gauss(0, 0.02)))
            nt["status"]    = "stable"
        next_teams.append(nt)

    return next_teams, list(relegated_names), list(promoted_names)


# ─── MONTE CARLO ENGINE ───────────────────────────────────────────────────────

def run_monte_carlo(cfg: dict) -> dict:
    """
    Run N Monte Carlo simulations and aggregate stats.

    Returns a dict with:
      - relegation_counts[team] = total times relegated across all runs & seasons
      - promotion_counts[team]  = total times promoted
      - season_league[team][season_idx] = fraction of runs team was in NBA
      - sample_run: one full season-by-season trace (for display)
    """
    mc_runs  = cfg["mc_runs"]
    seasons  = cfg["seasons"]

    all_teams = (
        [{"name": t["name"], "win_rate": t["win_rate"], "league": "NBA", "status": "stable"} for t in NBA_TEAMS] +
        [{"name": t["name"], "win_rate": t["win_rate"], "league": "G",   "status": "stable"} for t in G_LEAGUE_TEAMS]
    )
    team_names = [t["name"] for t in all_teams]

    relegation_counts = defaultdict(int)
    promotion_counts  = defaultdict(int)
    # season_nba_count[team][season] = # of runs where team was in NBA that season
    season_nba_count  = {name: [0] * seasons for name in team_names}

    sample_run = None

    print(f"\nRunning {mc_runs:,} Monte Carlo simulations × {seasons} seasons...")
    for run in range(mc_runs):
        state = deepcopy(all_teams)
        run_history = []

        for s in range(seasons):
            state, relegated, promoted = simulate_season(state, cfg)

            for name in relegated:
                relegation_counts[name] += 1
            for name in promoted:
                promotion_counts[name] += 1
            for t in state:
                if t["league"] == "NBA":
                    season_nba_count[t["name"]][s] += 1

            if run == 0:
                run_history.append({
                    "season": cfg["start_year"] + s,
                    "nba":  sorted([t for t in state if t["league"] == "NBA"], key=lambda x: -x["win_rate"]),
                    "g":    sorted([t for t in state if t["league"] == "G"],   key=lambda x: -x["win_rate"]),
                    "relegated": relegated,
                    "promoted":  promoted,
                })

        if run == 0:
            sample_run = run_history

        if (run + 1) % 100 == 0 or run == mc_runs - 1:
            print(f"  {run + 1:>5}/{mc_runs} runs complete", end="\r")

    print()  # newline after progress

    # Normalize to per-run-per-season probability
    total = mc_runs * seasons
    relegation_risk = {name: relegation_counts[name] / total for name in team_names}
    promotion_risk  = {name: promotion_counts[name]  / total for name in team_names}
    nba_fraction    = {
        name: [season_nba_count[name][s] / mc_runs for s in range(seasons)]
        for name in team_names
    }

    return {
        "relegation_risk":  relegation_risk,
        "promotion_risk":   promotion_risk,
        "nba_fraction":     nba_fraction,
        "relegation_counts": dict(relegation_counts),
        "promotion_counts":  dict(promotion_counts),
        "sample_run":        sample_run,
        "team_names":        team_names,
        "cfg":               cfg,
    }


# ─── DISPLAY ─────────────────────────────────────────────────────────────────

def print_sample_run(sample_run: list[dict]) -> None:
    SEP = "─" * 72
    print(f"\n{'═'*72}")
    print("  SAMPLE RUN — season-by-season trace (1 of N Monte Carlo runs)")
    print(f"{'═'*72}")

    for season_data in sample_run:
        yr = season_data["season"]
        print(f"\n  {yr}  {'─'*60}")
        print(f"  {'NBA':40s}  {'G LEAGUE':28s}")
        print(f"  {SEP}")

        nba_list = season_data["nba"]
        g_list   = season_data["g"]
        rows = max(len(nba_list), len(g_list))

        for i in range(rows):
            if i < len(nba_list):
                t = nba_list[i]
                tag = " ↑PROM" if t["status"] == "promoted" else (" ↓REL" if t["status"] == "relegated" else "")
                nba_col = f"  {i+1:>2}. {t['name']:<28s} {t['win_rate']*100:4.1f}%{tag}"
            else:
                nba_col = " " * 44

            if i < len(g_list):
                t = g_list[i]
                tag = " ↑PROM" if t["status"] == "promoted" else ""
                g_col = f"  {i+1:>2}. {t['name']:<20s} {t['win_rate']*100:4.1f}%{tag}"
            else:
                g_col = ""

            print(f"{nba_col:<44}{g_col}")


def print_risk_report(results: dict) -> None:
    cfg      = results["cfg"]
    runs     = cfg["mc_runs"]
    seasons  = cfg["seasons"]

    print(f"\n{'═'*72}")
    print(f"  RELEGATION RISK — NBA teams (per season, across {runs:,} runs × {seasons} seasons)")
    print(f"{'═'*72}")
    nba_names = [t["name"] for t in NBA_TEAMS]
    nba_risk  = sorted(nba_names, key=lambda n: -results["relegation_risk"][n])
    for name in nba_risk:
        pct  = results["relegation_risk"][name] * 100
        bar  = "█" * int(pct / 2)
        print(f"  {name:<32s} {pct:5.2f}%  {bar}")

    print(f"\n{'═'*72}")
    print(f"  PROMOTION RISK — G League teams (per season)")
    print(f"{'═'*72}")
    gl_names = [t["name"] for t in G_LEAGUE_TEAMS]
    gl_risk  = sorted(gl_names, key=lambda n: -results["promotion_risk"][n])
    for name in gl_risk:
        pct  = results["promotion_risk"][name] * 100
        bar  = "█" * int(pct / 2)
        print(f"  {name:<32s} {pct:5.2f}%  {bar}")

    print(f"\n{'═'*72}")
    print(f"  NBA STABILITY — fraction of seasons each NBA team stays in NBA")
    print(f"{'═'*72}")
    for name in nba_risk[::-1]:  # least risky first
        frac = 1 - results["relegation_risk"][name]
        bar  = "█" * int(frac * 30)
        print(f"  {name:<32s} {frac*100:5.1f}%  {bar}")


def print_mobility_summary(results: dict) -> None:
    cfg    = results["cfg"]
    runs   = cfg["mc_runs"]
    seasons = cfg["seasons"]

    print(f"\n{'═'*72}")
    print("  LEAGUE MOBILITY SUMMARY")
    print(f"{'═'*72}")

    all_names = results["team_names"]
    total_moves = {
        name: (results["relegation_counts"].get(name, 0) + results["promotion_counts"].get(name, 0)) / runs
        for name in all_names
    }
    top = sorted(all_names, key=lambda n: -total_moves[n])[:10]
    print(f"\n  Top 10 most volatile teams (avg league changes over {seasons} seasons):")
    for name in top:
        print(f"  {name:<32s}  {total_moves[name]:.2f} moves")

    avg_rel = sum(results["relegation_counts"].values()) / (runs * seasons)
    avg_prom = sum(results["promotion_counts"].values()) / (runs * seasons)
    print(f"\n  Avg relegations per season: {avg_rel:.2f}")
    print(f"  Avg promotions per season:  {avg_prom:.2f}")


# ─── EXPORT ──────────────────────────────────────────────────────────────────

def save_csv(results: dict, path: str) -> None:
    cfg     = results["cfg"]
    seasons = cfg["seasons"]
    year0   = cfg["start_year"]

    rows = []
    for name in results["team_names"]:
        starting_league = "NBA" if name in [t["name"] for t in NBA_TEAMS] else "G"
        row = {
            "team": name,
            "starting_league": starting_league,
            "relegation_risk_per_season": f"{results['relegation_risk'][name]*100:.3f}%",
            "promotion_risk_per_season":  f"{results['promotion_risk'][name]*100:.3f}%",
        }
        for s in range(seasons):
            row[f"{year0 + s}_nba_pct"] = f"{results['nba_fraction'][name][s]*100:.1f}%"
        rows.append(row)

    rows.sort(key=lambda r: (r["starting_league"], r["relegation_risk_per_season"]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  CSV saved → {path}")


def save_json(results: dict, path: str) -> None:
    out = {
        "config":           results["cfg"],
        "relegation_risk":  results["relegation_risk"],
        "promotion_risk":   results["promotion_risk"],
        "nba_fraction":     results["nba_fraction"],
        "relegation_counts": results["relegation_counts"],
        "promotion_counts":  results["promotion_counts"],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  JSON saved → {path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(42)  # remove for non-deterministic runs

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   NBA Promotion / Relegation — Monte Carlo Simulation           ║")
    print("║   Reform takes effect: 2027                                     ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\n  Config:")
    for k, v in CONFIG.items():
        if k not in ("csv_path", "json_path"):
            print(f"    {k:<28s} {v}")

    results = run_monte_carlo(CONFIG)

    if CONFIG["print_sample_run"]:
        print_sample_run(results["sample_run"])

    print_risk_report(results)
    print_mobility_summary(results)

    if CONFIG["save_csv"]:
        save_csv(results, CONFIG["csv_path"])
    if CONFIG["save_json"]:
        save_json(results, CONFIG["json_path"])

    print(f"\n{'═'*72}")
    print("  Done.")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()

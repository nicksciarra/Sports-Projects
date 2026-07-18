"""
MLB Player Market Value & Surplus Value Analysis
=================================================
Estimates the dollar value of WAR, calculates surplus value for each player,
identifies undervalued/overvalued players, and evaluates team roster efficiency.

Data sources (FanGraphs-standardized player IDs):
  - players.csv         : player identity lookup
  - player_seasons.csv  : seasonal WAR by player
  - salaries.csv        : seasonal salary by player
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_dir: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three source tables."""
    players = pd.read_csv("/Users/nicksciarra/Desktop/MLB Models/players.csv")
    seasons = pd.read_csv("/Users/nicksciarra/Desktop/MLB Models/player_seasons.csv")
    salaries = pd.read_csv("/Users/nicksciarra/Desktop/MLB Models/salaries.csv")
    return players, seasons, salaries


def clean_salary(salary_series: pd.Series) -> pd.Series:
    """Convert '$10,175,000.00' → 10175000.0"""
    return (
        salary_series
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.strip()
        .pipe(pd.to_numeric, errors="coerce")
    )


def build_master(players, seasons, salaries) -> pd.DataFrame:
    """
    Join the three tables on (playerid_fg, season) and return a clean
    master DataFrame with one row per player-season.
    """
    salaries = salaries.copy()
    salaries["salary"] = clean_salary(salaries["salary"])

    # Inner join: only rows that have both WAR and a salary record
    df = (
        seasons
        .merge(salaries, on=["playerid_fg", "season"], how="inner")
        .merge(players[["playerid_fg", "name"]], on="playerid_fg", how="left", suffixes=("", "_lookup"))
    )

    # Prefer the display name from the lookup table; fall back to seasons name
    df["player_name"] = df["name_lookup"].fillna(df["name"])
    df = df.drop(columns=["name", "name_lookup", "name_clean"], errors="ignore")

    # Drop rows with missing WAR or salary
    df = df.dropna(subset=["war", "salary"])

    # Sanitise extreme outliers: cap WAR at ±20, salary ≥ $0
    df = df[(df["salary"] >= 0) & (df["war"].between(-10, 20))]

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. MARKET VALUE MODEL  ($/WAR by season)
# ─────────────────────────────────────────────────────────────────────────────

# Dollar value of 1 WAR on the open market has grown roughly $5–9 M
# historically.  We derive it empirically from the data using a simple
# OLS regression of salary ~ WAR, estimated separately for each season
# so that inflation is captured automatically.

def estimate_war_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every season in the data, regress salary on WAR (players with
    positive WAR only, to exclude negative-value / replacement contracts
    from distorting the slope).

    Returns a DataFrame: season | dollars_per_war | intercept | r2 | n_players
    """
    records = []
    for season, grp in df[df["war"] > 0].groupby("season"):
        if len(grp) < 5:
            continue
        slope, intercept, r, _, _ = stats.linregress(grp["war"], grp["salary"])
        records.append({
            "season": season,
            "dollars_per_war": max(slope, 1_000_000),   # floor at $1 M to avoid negatives
            "intercept": intercept,
            "r2": r ** 2,
            "n_players": len(grp),
        })
    return pd.DataFrame(records).sort_values("season").reset_index(drop=True)


def attach_market_value(df: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """
    Merge dollars_per_war back onto the master frame, then compute:
      estimated_value  = max(WAR, 0) × dollars_per_war   [floor at 0]
      surplus_value    = estimated_value − salary
      surplus_pct      = surplus_value / salary            [salary > 0]
    """
    df = df.merge(market[["season", "dollars_per_war", "r2"]], on="season", how="left")

    df["estimated_value"] = np.maximum(df["war"], 0) * df["dollars_per_war"]
    df["surplus_value"] = df["estimated_value"] - df["salary"]
    df["surplus_pct"] = np.where(
        df["salary"] > 0,
        df["surplus_value"] / df["salary"] * 100,
        np.nan,
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. PLAYER-LEVEL CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

SURPLUS_THRESHOLDS = {
    "Highly Undervalued": 5_000_000,
    "Undervalued":        1_000_000,
    "Fair Value":        -1_000_000,
    "Overvalued":        -5_000_000,
    # below -5 M → "Highly Overvalued"
}


def classify_player(surplus: float) -> str:
    if surplus >= 5_000_000:
        return "Highly Undervalued"
    elif surplus >= 1_000_000:
        return "Undervalued"
    elif surplus >= -1_000_000:
        return "Fair Value"
    elif surplus >= -5_000_000:
        return "Overvalued"
    else:
        return "Highly Overvalued"


def add_classification(df: pd.DataFrame) -> pd.DataFrame:
    df["value_label"] = df["surplus_value"].apply(classify_player)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. TEAM EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────

def team_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate by (team, season):
      total_payroll      : sum of salaries
      total_war          : sum of WAR
      total_surplus      : sum of surplus_value
      war_per_dollar     : WAR / payroll (efficiency ratio, $/WAR inverted)
      efficiency_score   : z-score of war_per_dollar within each season
                           (positive = above-average bang for buck)
    """
    grp = (
        df.groupby(["team", "season"])
        .agg(
            total_payroll=("salary", "sum"),
            total_war=("war", "sum"),
            total_surplus=("surplus_value", "sum"),
            n_players=("playerid_fg", "nunique"),
        )
        .reset_index()
    )
    grp["war_per_million"] = grp["total_war"] / (grp["total_payroll"] / 1_000_000)

    # Within-season z-score so teams are compared only to peers in the same year
    grp["efficiency_score"] = grp.groupby("season")["war_per_million"].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    grp["efficiency_label"] = pd.cut(
        grp["efficiency_score"],
        bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
        labels=["Poor", "Below Avg", "Average", "Above Avg", "Excellent"],
    )
    return grp.sort_values(["season", "efficiency_score"], ascending=[True, False]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUMMARY REPORTS
# ─────────────────────────────────────────────────────────────────────────────

def top_undervalued(df: pd.DataFrame, n: int = 20, season: int = None) -> pd.DataFrame:
    """Top-N most undervalued player-seasons (highest surplus_value)."""
    subset = df if season is None else df[df["season"] == season]
    cols = ["player_name", "team", "season", "war", "salary", "estimated_value",
            "surplus_value", "surplus_pct", "value_label"]
    return (
        subset[cols]
        .sort_values("surplus_value", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )


def top_overvalued(df: pd.DataFrame, n: int = 20, season: int = None) -> pd.DataFrame:
    """Top-N most overvalued player-seasons (most negative surplus_value)."""
    subset = df if season is None else df[df["season"] == season]
    cols = ["player_name", "team", "season", "war", "salary", "estimated_value",
            "surplus_value", "surplus_pct", "value_label"]
    return (
        subset[cols]
        .sort_values("surplus_value", ascending=True)
        .head(n)
        .reset_index(drop=True)
    )


def career_surplus(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate surplus value across all seasons for each player."""
    return (
        df.groupby(["playerid_fg", "player_name"])
        .agg(
            seasons=("season", "nunique"),
            total_war=("war", "sum"),
            total_salary=("salary", "sum"),
            total_estimated_value=("estimated_value", "sum"),
            total_surplus=("surplus_value", "sum"),
        )
        .reset_index()
        .sort_values("total_surplus", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. FORMATTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fmt_dollars(val: float) -> str:
    if pd.isna(val):
        return "N/A"
    sign = "-" if val < 0 else ""
    return f"{sign}${abs(val):,.0f}"


def print_market_summary(market: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  MARKET VALUE OF WAR BY SEASON")
    print("=" * 60)
    print(f"{'Season':>8}  {'$/WAR':>14}  {'R²':>6}  {'N Players':>10}")
    print("-" * 60)
    for _, row in market.iterrows():
        print(f"{int(row.season):>8}  {fmt_dollars(row.dollars_per_war):>14}  "
              f"{row.r2:>6.3f}  {int(row.n_players):>10}")


def print_player_table(df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(f"{'Player':<25} {'Team':>5} {'Yr':>5} {'WAR':>6} {'Salary':>15} "
          f"{'Est. Value':>15} {'Surplus':>15} {'Label':<20}")
    print("-" * 90)
    for _, r in df.iterrows():
        print(
            f"{str(r.player_name):<25} {str(r.team):>5} {int(r.season):>5} "
            f"{r.war:>6.1f} {fmt_dollars(r.salary):>15} "
            f"{fmt_dollars(r.estimated_value):>15} "
            f"{fmt_dollars(r.surplus_value):>15} "
            f"{str(r.value_label):<20}"
        )


def print_team_table(df: pd.DataFrame, title: str, n: int = 30) -> None:
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)
    print(f"{'Team':>5} {'Season':>7} {'Payroll':>15} {'WAR':>7} "
          f"{'Surplus':>15} {'WAR/$M':>8} {'Eff. Score':>12} {'Label':<12}")
    print("-" * 90)
    for _, r in df.head(n).iterrows():
        print(
            f"{str(r.team):>5} {int(r.season):>7} "
            f"{fmt_dollars(r.total_payroll):>15} "
            f"{r.total_war:>7.1f} "
            f"{fmt_dollars(r.total_surplus):>15} "
            f"{r.war_per_million:>8.2f} "
            f"{r.efficiency_score:>12.2f} "
            f"{str(r.efficiency_label):<12}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: str = "data") -> dict:
    """
    Execute the full pipeline. Returns a dict of the key DataFrames so
    callers (notebooks, tests, dashboards) can work with the results.
    """
    print("Loading data …")
    players, seasons, salaries = load_data(data_dir)

    print("Building master table …")
    master = build_master(players, seasons, salaries)
    print(f"  → {len(master):,} player-season rows across "
          f"{master.season.nunique()} seasons and {master.team.nunique()} teams")

    print("Estimating $/WAR by season …")
    market = estimate_war_market_value(master)

    print("Attaching market value & surplus …")
    master = attach_market_value(master, market)
    master = add_classification(master)

    print("Computing team efficiency …")
    teams = team_efficiency(master)

    # ── Reports ────────────────────────────────────────────────────────────
    print_market_summary(market)

    latest_season = master["season"].max()
    print(f"\n[Showing results for most recent season: {latest_season}]")

    print_player_table(
        top_undervalued(master, n=15, season=latest_season),
        f"TOP 15 MOST UNDERVALUED PLAYERS — {latest_season}"
    )
    print_player_table(
        top_overvalued(master, n=15, season=latest_season),
        f"TOP 15 MOST OVERVALUED PLAYERS — {latest_season}"
    )

    # All-time career surplus leaders
    careers = career_surplus(master)
    print("\n" + "=" * 80)
    print("  CAREER SURPLUS VALUE LEADERS (all seasons combined)")
    print("=" * 80)
    print(f"{'Player':<25} {'Seasons':>8} {'Total WAR':>10} "
          f"{'Total Salary':>15} {'Total Surplus':>15}")
    print("-" * 80)
    for _, r in careers.head(15).iterrows():
        print(f"{str(r.player_name):<25} {int(r.seasons):>8} "
              f"{r.total_war:>10.1f} "
              f"{fmt_dollars(r.total_salary):>15} "
              f"{fmt_dollars(r.total_surplus):>15}")

    # Team efficiency — top & bottom in latest season
    latest_teams = teams[teams["season"] == latest_season]
    print_team_table(
        latest_teams.sort_values("efficiency_score", ascending=False),
        f"TEAM ROSTER EFFICIENCY — {latest_season} (Best → Worst)",
    )

    # Value distribution summary
    print("\n" + "=" * 50)
    print(f"  VALUE CLASSIFICATION DISTRIBUTION — {latest_season}")
    print("=" * 50)
    dist = (
        master[master["season"] == latest_season]["value_label"]
        .value_counts()
        .reindex(["Highly Undervalued", "Undervalued", "Fair Value",
                  "Overvalued", "Highly Overvalued"])
    )
    for label, count in dist.items():
        print(f"  {label:<22}  {count:>4} players")

    return {
        "master": master,
        "market": market,
        "teams": teams,
        "careers": careers,
    }


if __name__ == "__main__":
    results = run(data_dir="/Users/nicksciarra/Desktop/MLB Models")

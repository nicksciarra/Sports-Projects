# MLB Player Market Value & Surplus Value Analysis

A data-driven Python project that estimates the dollar value of on-field
performance (WAR), calculates each player's surplus value relative to their
salary, identifies undervalued and overvalued players, and evaluates team-level
roster efficiency.

All player identity is standardised on **FanGraphs player IDs** (`playerid_fg`).

---

## Project Structure

```
mlb_surplus_value/
├── analysis.py          # Full pipeline — run this
├── README.md
└── data/
    ├── players.csv          # Player identity lookup
    ├── player_seasons.csv   # Seasonal WAR by player
    └── salaries.csv         # Seasonal salary by player
```

---

## Methodology

### 1. Data Joining

The three relational tables are joined on `(playerid_fg, season)`:

```
players  ──┐
            ├──► master DataFrame (one row per player-season)
salaries ──┤
            │
seasons  ──┘
```

Only rows with **both** a WAR figure and a salary record are kept (inner join).

### 2. Market Value Model ($/WAR by Season)

For every season we run an **OLS regression** of `salary ~ WAR` on players
with positive WAR.  This gives us a season-specific $/WAR slope that
automatically captures baseball's salary inflation over time.

```
estimated_value = max(WAR, 0) × dollars_per_war
```

Flooring estimated value at zero prevents a negative-WAR season from
producing a "negative salary" artefact.

### 3. Surplus Value

```
surplus_value = estimated_value − salary
surplus_pct   = surplus_value / salary × 100
```

Positive surplus → player is **undervalued** relative to market.  
Negative surplus → player is **overvalued** (paid more than performance warrants).

### 4. Player Classification

| Surplus Value      | Label               |
|--------------------|---------------------|
| ≥ +$5 M            | Highly Undervalued  |
| $1 M – $5 M        | Undervalued         |
| -$1 M – $1 M       | Fair Value          |
| -$5 M – -$1 M      | Overvalued          |
| < -$5 M            | Highly Overvalued   |

### 5. Team Efficiency

Teams are evaluated on **WAR per $1 M of payroll**.  An **efficiency score**
(within-season z-score) lets you compare organisations across different payroll
eras.

| Efficiency Score | Label       |
|-----------------|-------------|
| > +1.0          | Excellent   |
| +0.5 – +1.0     | Above Avg   |
| -0.5 – +0.5     | Average     |
| -1.0 – -0.5     | Below Avg   |
| < -1.0          | Poor        |

---

## Quick Start

```bash
# Install dependencies (standard library + scipy)
pip install pandas numpy scipy

# Place your three CSV files in data/
# Then run:
python analysis.py
```

### Using Results Programmatically

```python
from analysis import run

results = run(data_dir="data")

master  = results["master"]   # Full player-season data with surplus columns
market  = results["market"]   # $/WAR estimates by season
teams   = results["teams"]    # Team efficiency metrics
careers = results["careers"]  # Career aggregated surplus per player
```

### Key Columns in `master`

| Column            | Description                                        |
|-------------------|----------------------------------------------------|
| `playerid_fg`     | FanGraphs player ID (join key)                     |
| `player_name`     | Display name                                       |
| `team`            | Team abbreviation                                  |
| `season`          | Season year                                        |
| `war`             | Wins Above Replacement                             |
| `salary`          | Actual salary (USD)                                |
| `dollars_per_war` | Season's modelled $/WAR                            |
| `estimated_value` | Market value of the player's WAR                   |
| `surplus_value`   | `estimated_value − salary`                         |
| `surplus_pct`     | Surplus as % of salary                             |
| `value_label`     | Classification string                              |

---

## Key Findings (2025 Season Sample)

- **$/WAR** has grown from ~$1.0 M (2021) to ~$2.3 M (2025), reflecting
  baseball's payroll inflation.
- **Pre-arb players** dominate the undervalued list (Skenes, Henderson,
  De La Cruz) because service-time rules cap their salaries far below market.
- **Mega-contracts** (Ohtani $70 M, Soto $51 M) show the largest negative
  surplus in absolute dollar terms — expected, since they were signed at
  premium prices that the linear WAR model cannot fully capture.
- **Baltimore Orioles** lead team efficiency in 2025 with a young, pre-arb
  core generating elite WAR at minimal cost.

---

## Limitations & Extensions

- The linear $/WAR model undervalues elite players (superstar premium) and
  overvalues replacement-level contracts.  A **non-linear or quantile regression**
  would improve accuracy.
- `"2 Tms"` rows appear when a player changed teams mid-season; salary
  attribution across stints is a known edge case.
- Defensive metrics and positional adjustments embedded in WAR are already
  accounted for, but **playing time / injury risk** are not modelled.
- Adding **free-agent contract data** would let you test whether teams overpay
  on the open market vs. retaining their own players.

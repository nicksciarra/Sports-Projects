"""
data/fetch_weather.py — Pull forecast & historical weather for PGA venues
---
Uses Open-Meteo (https://open-meteo.com)
  ✅ Completely FREE for non-commercial use
  ✅ No API key required
  ✅ No registration or credit card needed
  ✅ Forecast up to 16 days out + full historical archive

Weather features engineered:
  - avg_wind_mph      : affects driving & approach distances
  - max_gust_mph      : penalty for drivers/long irons
  - avg_temp_f        : affects ball flight distance
  - total_precip_mm   : wet conditions = softer greens (holds approach shots)
  - avg_humidity      : slightly affects ball flight
  - precip_days       : number of rainy days during tournament
  - is_windy          : flag — avg wind > 15 mph
  - is_wet            : flag — total precip > 5mm
"""

import sys
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR

import os
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Open-Meteo endpoints — no key needed
OM_FORECAST_URL  = "https://api.open-meteo.com/v1/forecast"
OM_ARCHIVE_URL   = "https://archive-api.open-meteo.com/v1/archive"

# Known PGA Tour venues with lat/lon
# Add more as needed: Google "<venue name> coordinates"
VENUE_COORDS = {
    "TPC Sawgrass":               (30.1975, -81.3953),
    "Augusta National":           (33.5020, -82.0196),
    "Pebble Beach Golf Links":    (36.5686, -121.9481),
    "TPC Scottsdale":             (33.6607, -111.8910),
    "Riviera Country Club":       (34.0480, -118.5157),
    "Muirfield Village":          (40.0787, -83.1037),
    "Quail Hollow Club":          (35.4373, -80.8482),
    "East Lake Golf Club":        (33.7270, -84.3105),
    "Bay Hill Club":              (28.4669, -81.4962),
    "Colonial Country Club":      (32.7157, -97.3622),
    "Torrey Pines":               (32.8953, -117.2455),
    "TPC Deere Run":              (41.5247, -90.5168),
    "Harbour Town Golf Links":    (32.1382, -80.8070),
    "Sedgefield Country Club":    (36.0299, -79.9170),
    "Caves Valley Golf Club":     (39.4648, -76.7502),
}

# Weather variables to request from Open-Meteo
HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "relative_humidity_2m",
]


def _ms_to_mph(ms: float) -> float:
    return ms * 2.23694


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def fetch_forecast(
    venue_name: str,
    lat: float,
    lon: float,
    tournament_start: str,  # "YYYY-MM-DD"
    rounds: int = 4,
) -> dict:
    """
    Fetch Open-Meteo forecast for tournament window.
    Works for tournaments starting within the next ~16 days.
    No API key required.
    """
    params = {
        "latitude":        lat,
        "longitude":       lon,
        "hourly":          ",".join(HOURLY_VARS),
        "wind_speed_unit": "mph",
        "temperature_unit": "fahrenheit",
        "timezone":        "auto",
        "forecast_days":   16,
    }

    try:
        resp = requests.get(OM_FORECAST_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Open-Meteo forecast failed for {venue_name}: {e}")
        return _neutral_weather(venue_name)

    return _aggregate_weather(data, venue_name, tournament_start, rounds, source="forecast")


def fetch_historical_weather(
    venue_name: str,
    lat: float,
    lon: float,
    tournament_start: str,  # "YYYY-MM-DD"
    rounds: int = 4,
) -> dict:
    """
    Fetch Open-Meteo historical weather archive.
    Use this for past tournaments when building training data.
    Archive available from 1940 onwards.
    """
    start_dt = datetime.strptime(tournament_start, "%Y-%m-%d")
    end_dt   = start_dt + timedelta(days=rounds - 1)

    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       tournament_start,
        "end_date":         end_dt.strftime("%Y-%m-%d"),
        "hourly":           ",".join(HOURLY_VARS),
        "wind_speed_unit":  "mph",
        "temperature_unit": "fahrenheit",
        "timezone":         "auto",
    }

    try:
        resp = requests.get(OM_ARCHIVE_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Open-Meteo archive failed for {venue_name}: {e}")
        return _neutral_weather(venue_name)

    return _aggregate_weather(data, venue_name, tournament_start, rounds, source="archive")


def _aggregate_weather(
    data: dict,
    venue_name: str,
    tournament_start: str,
    rounds: int,
    source: str = "forecast",
) -> dict:
    """Parse Open-Meteo hourly response into model features."""
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])

    if not times:
        return _neutral_weather(venue_name)

    start_dt = datetime.strptime(tournament_start, "%Y-%m-%d")
    end_dt   = start_dt + timedelta(days=rounds)

    wind_speeds, gusts, temps, precips, humidities = [], [], [], [], []

    for i, t in enumerate(times):
        try:
            dt = datetime.fromisoformat(t)
        except ValueError:
            continue

        # Only include daylight tournament hours (7am–7pm)
        if not (start_dt <= dt <= end_dt):
            continue
        if not (7 <= dt.hour <= 19):
            continue

        ws  = hourly.get("wind_speed_10m",      [None] * len(times))[i]
        g   = hourly.get("wind_gusts_10m",       [None] * len(times))[i]
        t_  = hourly.get("temperature_2m",       [None] * len(times))[i]
        p   = hourly.get("precipitation",        [None] * len(times))[i]
        h   = hourly.get("relative_humidity_2m", [None] * len(times))[i]

        if ws  is not None: wind_speeds.append(ws)
        if g   is not None: gusts.append(g)
        if t_  is not None: temps.append(t_)
        if p   is not None: precips.append(p)
        if h   is not None: humidities.append(h)

    if not wind_speeds:
        print(f"  [WARN] No daytime weather data found for window — using neutral.")
        return _neutral_weather(venue_name)

    features = {
        "venue":           venue_name,
        "weather_source":  source,
        "avg_wind_mph":    round(sum(wind_speeds) / len(wind_speeds), 1),
        "max_gust_mph":    round(max(gusts), 1) if gusts else 0,
        "avg_temp_f":      round(sum(temps) / len(temps), 1) if temps else 72,
        "total_precip_mm": round(sum(precips), 1),
        "avg_humidity":    round(sum(humidities) / len(humidities), 1) if humidities else 55,
        "precip_days":     sum(1 for p in precips if p > 0.5),
        "is_windy":        int((sum(wind_speeds) / len(wind_speeds)) > 15),
        "is_wet":          int(sum(precips) > 5),
    }

    print(f"  ✓ Weather [{source}] @ {venue_name}: "
          f"{features['avg_wind_mph']} mph wind, "
          f"{features['total_precip_mm']}mm rain, "
          f"{features['avg_temp_f']}°F")
    return features


def _neutral_weather(venue_name: str) -> dict:
    """Fallback neutral weather when fetch fails."""
    return {
        "venue": venue_name, "weather_source": "neutral",
        "avg_wind_mph": 10.0, "max_gust_mph": 15.0,
        "avg_temp_f": 72.0, "total_precip_mm": 0.0,
        "avg_humidity": 55.0, "precip_days": 0,
        "is_windy": 0, "is_wet": 0,
    }


def fetch_tournament_weather(
    venue_name: str,
    tournament_start: str,
    rounds: int = 4,
    use_archive: bool = False,
) -> dict:
    """
    High-level wrapper. Automatically picks forecast vs. archive
    based on tournament date, or use use_archive=True to force archive.
    """
    coords = VENUE_COORDS.get(venue_name)
    if coords is None:
        print(f"[WARN] '{venue_name}' not in VENUE_COORDS.")
        print(f"       Add it to VENUE_COORDS in fetch_weather.py")
        return _neutral_weather(venue_name)

    lat, lon = coords

    # If date is in the past, use historical archive
    t_start = datetime.strptime(tournament_start, "%Y-%m-%d")
    if use_archive or t_start < datetime.now() - timedelta(days=1):
        return fetch_historical_weather(venue_name, lat, lon, tournament_start, rounds)
    else:
        return fetch_forecast(venue_name, lat, lon, tournament_start, rounds)


def save_weather_features(weather_dict: dict, tournament_name: str):
    df = pd.DataFrame([weather_dict])
    path = f"{RAW_DATA_DIR}weather_{tournament_name.replace(' ', '_').lower()}.csv"
    df.to_csv(path, index=False)
    print(f"  ✓ Saved → {path}")
    return df


if __name__ == "__main__":
    print("Open-Meteo Weather Fetch — No API key required\n")

    # Upcoming tournament forecast
    w = fetch_tournament_weather(
        venue_name="TPC Sawgrass",
        tournament_start="2025-03-13",
        rounds=4,
    )
    print("\nWeather features:")
    for k, v in w.items():
        print(f"  {k:22s}: {v}")
    save_weather_features(w, "THE PLAYERS Championship")

    # Historical example (past tournament)
    print("\nFetching historical weather example...")
    w_hist = fetch_tournament_weather(
        venue_name="Augusta National",
        tournament_start="2024-04-11",
        rounds=4,
        use_archive=True,
    )
    print(f"  Masters 2024: {w_hist['avg_wind_mph']} mph wind, "
          f"{w_hist['total_precip_mm']}mm rain")

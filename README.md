# PGA Tour Predictor — Full Pipeline

## Overview
A complete machine learning pipeline to predict PGA Tour tournament standings using strokes gained, course history, weather, and more.

## Project Structure
```
pga_predictor/
├── README.md
├── requirements.txt
├── config.py                  # API keys, settings
├── data/
│   ├── fetch_data.py          # DataGolf API ingestion
│   ├── fetch_weather.py       # OpenWeather API ingestion
│   └── preprocess.py          # Feature engineering
├── models/
│   ├── train.py               # Model training (XGBoost)
│   ├── predict.py             # Predict upcoming tournament
│   └── evaluate.py            # Backtest & evaluation
├── utils/
│   └── helpers.py             # Shared utilities
└── output/                    # Saved models & predictions
```

## Data Sources (Free/Freemium)
1. **DataGolf API** (datagolf.com) — Best source for PGA strokes gained, rankings, course fit
   - Free tier: limited historical + current season data
   - Paid (~$30/mo): full historical SG splits, field projections
2. **OpenWeatherMap API** (free tier) — Forecast data for tournament locations
3. **PGA Tour website** — Supplemental schedule/field info (scraped)

## Quickstart
```bash
pip install -r requirements.txt

# 1. Set your API keys in config.py
# 2. Fetch data
python data/fetch_data.py

# 3. Preprocess & engineer features
python data/preprocess.py

# 4. Train model
python models/train.py

# 5. Predict next tournament
python models/predict.py --tournament "THE PLAYERS Championship" --year 2025
```

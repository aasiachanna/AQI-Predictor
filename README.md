# AQI Predictor (Serverless-ready)

An end-to-end pipeline to forecast Air Quality Index (AQI) for the next 3 days using public APIs, feature engineering, and a lightweight dashboard.

## Quickstart

1) Create and activate a Python 3.10+ virtualenv, then install deps:

```bash
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

2) Optional: Configure `.env` for live APIs

```ini
AQICN_TOKEN=YOUR_AQICN_TOKEN
OPENWEATHER_API_KEY=YOUR_OPENWEATHER_KEY
```

3) Get data
- Backfill via Open-Meteo (no key):
```bash
python data/raw/open_meteo.py
```
- (Optional) Live nowcast samples:
```bash
python -m src.ingestion.fetch_aqicn
python -m src.ingestion.fetch_openweather
```

4) Use REAL AQI from AQICN (history CSVs)
- Export hourly AQI CSVs from the AQICN Data Platform for your location.
- Place files under: `data/raw/aqicn_history/`
- Build cleaned AQICN:
```bash
python -m src.ingestion.aqicn_history_loader
```
- Build features (prefers AQICN history for target):
```bash
python -m src.features.feature_pipeline
```
- Train with time-based split:
```bash
python scripts/train_model_time_split.py
```

5) Run the dashboard
```bash
python dashboard/run_dashboard.py
```
Open http://localhost:8501

## Project Structure
- `src/ingestion/aqicn_history_loader.py`: Clean & consolidate AQICN history CSVs → `data/processed/aqicn_clean.csv`
- `src/features/feature_pipeline.py`: Prefers true AQI from `aqicn_clean.csv`, merges covariates, engineers features
- `scripts/train_model_time_split.py`: Chronological split training & evaluation
- `dashboard/`: Streamlit app and utils

## Notes
- For Streamlit Cloud, include `data/processed/processed_features.csv` and `models/` artifacts in the repo, or implement a startup fetch/build flow.

# AQI Predictor (Serverless-ready)

An end-to-end pipeline to forecast Air Quality Index (AQI) for the next 3 days using public APIs, feature engineering, and a lightweight dashboard.

## Quickstart

1) Create and activate a Python 3.10+ virtualenv, then install deps:

```bash
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

2) Configure environment variables in a `.env` file at repo root:

```ini
# AQICN (optional if you only use Open-Meteo)
AQICN_TOKEN=YOUR_AQICN_TOKEN

# OpenWeather (optional; used by src/ingestion/fetch_openweather.py)
OPENWEATHER_API_KEY=YOUR_OPENWEATHER_KEY
```

3) Collect raw data (examples):

- AQICN nowcast by geo:
```bash
python -m src.ingestion.fetch_aqicn
```
- OpenWeather current:
```bash
python -m src.ingestion.fetch_openweather
```
- Open-Meteo hourly (weather only):
```bash
python data/raw/weather_data.py
```

4) Build processed features:
```bash
python -m src.features.feature_pipeline
```
This writes `data/processed/processed_features.csv`.

5) Train a baseline model:
```bash
python scripts/train_model.py
```
This writes `models/model.joblib` and `models/metadata.json`.

6) Run the dashboard:
```bash
python dashboard/run_dashboard.py
```
Open http://localhost:8501

## Project Structure

- `src/ingestion/`: API scripts (AQICN, OpenWeather)
- `src/features/feature_pipeline.py`: Joins raw JSONs, engineers features, saves CSV
- `scripts/train_model.py`: Trains RandomForest on processed features
- `dashboard/`: Streamlit app and utils
- `data/raw/`: Raw JSON/CSV files
- `data/processed/processed_features.csv`: Engineered features
- `models/`: Saved model artifacts

## Notes
- The dashboard expects processed features and a trained model. Run steps 4–5 first.
- You can extend training to XGBoost/LightGBM, and add SHAP explanations.
- For automation (hourly/daily runs), wire these scripts into Airflow or GitHub Actions.

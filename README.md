# Pearls AQI Predictor

A comprehensive machine learning project to forecast Air Quality Index (AQI) for the next 3 days using time-series analysis, feature engineering, and multiple regression models.

## Overview

This project implements an end-to-end machine learning pipeline that:
- Processes raw air quality and weather data
- Engineers time-based and lag features
- Trains and compares multiple regression models
- Provides an interactive Streamlit dashboard for predictions
- Includes SHAP-based explainability

## Features

- **Data Processing**: Automated pipeline to process raw CSV files into features
- **Feature Engineering**: Time-based features including lags, rolling statistics, and seasonal patterns
- **Multiple Models**: Trains and compares Linear Regression, Ridge, Random Forest, XGBoost, and LightGBM
- **Model Selection**: Automatically selects the best-performing model based on RMSE
- **Interactive Dashboard**: Streamlit app for visualization and predictions
- **Model Explainability**: SHAP values for feature importance analysis
- **Reproducibility**: Fixed random seeds and consistent data splits

## Project Structure

```
AQI-Predictor/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   ├── raw/                     # Raw CSV data files
│   └── processed/               # Processed features and history
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   └── utils/
│       ├── feature_store.py     # Feature loading utilities
│       └── model_loader.py      # Model loading utilities
├── models/                      # Trained models (generated)
├── notebooks/
│   └── exploration.ipynb        # Data exploration notebook
├── scripts/
│   └── train_models.py          # Model training script
├── src/
│   └── pearls_aqi/
│       ├── __init__.py
│       ├── config.py            # Configuration loader
│       ├── data_processing.py   # Data processing utilities
│       ├── forecast.py          # Forecast utilities
│       └── pipeline.py          # Training pipeline
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── run_dashboard.py             # Dashboard runner script
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository or navigate to the project directory:
```bash
cd AQI-Predictor
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

The project expects raw CSV files in the `data/raw/` directory with the following columns:
- `timestamp`: DateTime column
- `pm2_5`: PM2.5 concentration
- Other optional columns: `pm10`, `ozone`, `temperature_2m`, `relative_humidity_2m`, etc.

Example data files are already included in `data/raw/`.

## Usage

### Step 1: Process Data

Run the feature pipeline to process raw data into features:

```bash
python -m src.features.feature_pipeline
```

This will:
- Load all CSV files from `data/raw/`
- Calculate AQI from PM2.5 values
- Aggregate to daily level
- Create lag features, rolling statistics, and time-based features
- Save processed data to `data/processed/`

### Step 2: Train Models

Train and compare multiple models:

```bash
python scripts/train_models.py
```

This will:
- Train 5 models: Linear Regression, Ridge, Random Forest, XGBoost, LightGBM
- Evaluate each model using RMSE, MAE, and R²
- Select the best model based on test RMSE
- Save all models and metadata to `models/` directory
- Generate a comparison report

**Output:**
- `models/linearregression_model.joblib` - Linear Regression model
- `models/ridge_model.joblib` - Ridge Regression model
- `models/randomforest_model.joblib` - Random Forest model
- `models/xgboost_model.joblib` - XGBoost model
- `models/lightgbm_model.joblib` - LightGBM model
- `models/model_comparison.json` - Comparison report with metrics
- Individual metadata files for each model

### Step 3: Run Dashboard

Launch the Streamlit dashboard:

```bash
python run_dashboard.py
```

Or directly:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

**Dashboard Features:**
- Select different trained models from the sidebar
- View model performance metrics (RMSE, MAE, R²)
- See AQI predictions for the next 3 days
- Visualize predictions with interactive plots
- View feature importance
- Explore SHAP values for model explainability

## Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  processed_features_file: "data/processed/processed_features.csv"
  daily_history_file: "data/processed/daily_history.csv"

model:
  target: "aqi"
  forecast_days: 3
  random_seed: 42
  test_fraction: 0.2
  min_training_rows: 90
  max_lag: 7
  rolling_windows: [3, 7]
  select_k_best: 10
```

## Model Details

### Feature Engineering

The pipeline creates the following features:
- **Lag Features**: AQI values for the previous 1-7 days
- **Rolling Statistics**: Mean and standard deviation over 3 and 7-day windows
- **Difference Features**: Day-to-day and week-to-week differences
- **Time Features**: Day of week, month, day of year, weekend indicator
- **Seasonal Features**: Sine and cosine transformations of day of year

### Models

1. **Linear Regression**: Baseline linear model
2. **Ridge Regression**: Regularized linear model (L2 penalty)
3. **Random Forest**: Ensemble of decision trees
4. **XGBoost**: Gradient boosting framework
5. **LightGBM**: Lightweight gradient boosting framework

### Evaluation Metrics

- **RMSE** (Root Mean Squared Error): Measures prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **R²** (Coefficient of Determination): Proportion of variance explained

## Data Sources

The project uses air quality and weather data. Example data files are provided in `data/raw/`. The data includes:
- PM2.5 and PM10 concentrations
- Ozone, carbon monoxide, nitrogen dioxide, sulphur dioxide
- Temperature, humidity, pressure, wind speed, precipitation

## Troubleshooting

### Common Issues

1. **FileNotFoundError: Features file not found**
   - Solution: Run the feature pipeline first: `python -m src.features.feature_pipeline`

2. **FileNotFoundError: Model file not found**
   - Solution: Train models first: `python scripts/train_models.py`

3. **Import errors**
   - Solution: Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Ensure you're in the project root directory

4. **Dashboard shows no data**
   - Solution: Check that `data/processed/processed_features.csv` exists
   - Verify the date range in the dashboard includes available data

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

The project uses:
- `black` for code formatting
- `flake8` for linting

### Adding New Models

To add a new model, edit `scripts/train_models.py` and add it to the `models_to_train` dictionary.

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

## Contact

For questions or issues, please open an issue on the repository.

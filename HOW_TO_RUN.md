# Complete Guide: How to Run the AQI Predictor Project

## 📁 Project Structure Analysis

### Directory Overview

```
AQI-Predictor/
├── config/                          # Configuration files
│   └── config.yaml                  # Main configuration (data paths, model params)
│
├── data/                            # Data directories
│   ├── raw/                         # Raw CSV data files (input)
│   │   ├── air_quality_weather_2024-01-01_to_2024-03-30.csv
│   │   ├── air_quality_weather_2024-03-31_to_2024-06-28.csv
│   │   ├── air_quality_weather_2024-06-29_to_2024-09-26.csv
│   │   ├── air_quality_weather_2024-09-27_to_2024-12-25.csv
│   │   └── air_quality_weather_2024-12-26_to_2024-12-31.csv
│   └── processed/                   # Processed features (generated)
│       ├── processed_features.csv   # Features for model training
│       └── daily_history.csv        # Daily aggregated AQI history
│
├── dashboard/                       # Streamlit dashboard
│   ├── app.py                       # Main dashboard application
│   └── utils/
│       ├── feature_store.py         # Feature loading utilities
│       └── model_loader.py          # Model loading utilities
│
├── models/                          # Trained models (generated)
│   ├── linearregression_model.joblib
│   ├── ridge_model.joblib
│   ├── randomforest_model.joblib
│   ├── xgboost_model.joblib
│   ├── lightgbm_model.joblib
│   ├── *_metadata.json              # Model metadata files
│   └── model_comparison.json        # Model comparison report
│
├── notebooks/                       # Jupyter notebooks
│   └── exploration.ipynb            # Data exploration notebook
│
├── scripts/                         # Training scripts
│   └── train_models.py              # Model training script
│
├── src/                             # Source code
│   ├── features/
│   │   └── feature_pipeline.py      # Feature engineering pipeline
│   └── pearls_aqi/                  # Core package
│       ├── __init__.py
│       ├── config.py                # Configuration loader
│       ├── data_processing.py       # Data processing utilities
│       ├── forecast.py              # Forecast utilities
│       └── pipeline.py              # Complete training pipeline
│
├── tests/                           # Unit tests
│   ├── test_data.py
│   └── test_models.py
│
├── venv/                            # Virtual environment (Python packages)
│
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package setup file
├── run_dashboard.py                 # Dashboard runner script
├── verify_setup.py                  # Setup verification script
├── README.md                        # Main documentation
├── QUICKSTART.md                    # Quick start guide
└── PROJECT_ENHANCEMENTS.md          # Project enhancements summary
```

## 🔍 Key Files Analysis

### Configuration Files
- **`config/config.yaml`**: Contains all project settings:
  - Data paths (raw_dir, processed_dir)
  - Model parameters (target, forecast_days, random_seed, etc.)
  - Feature engineering parameters (max_lag, rolling_windows)

### Core Source Files
- **`src/pearls_aqi/config.py`**: Loads and parses configuration from YAML
- **`src/pearls_aqi/data_processing.py`**: Processes raw data, calculates AQI, creates features
- **`src/features/feature_pipeline.py`**: Main feature engineering pipeline entry point
- **`src/pearls_aqi/forecast.py`**: Generates forecasts for future dates
- **`src/pearls_aqi/pipeline.py`**: Wraps complete training workflow

### Training Scripts
- **`scripts/train_models.py`**: Trains 5 models (Linear, Ridge, RandomForest, XGBoost, LightGBM)
  - Loads processed features
  - Splits data (time-based)
  - Trains and evaluates all models
  - Saves models and metadata
  - Generates comparison report

### Dashboard Files
- **`dashboard/app.py`**: Streamlit dashboard with:
  - Model selection
  - AQI predictions visualization
  - Feature importance
  - SHAP explainability
- **`dashboard/utils/feature_store.py`**: Loads and generates features for predictions
- **`dashboard/utils/model_loader.py`**: Loads trained models and metadata
- **`run_dashboard.py`**: Convenience script to launch dashboard

### Utility Files
- **`verify_setup.py`**: Checks if project is set up correctly
- **`requirements.txt`**: Python package dependencies
- **`setup.py`**: Package installation configuration

## 🚀 Complete Running Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Windows PowerShell (you're on Windows)

### Step 1: Verify Setup

First, check if everything is set up correctly:

```powershell
python verify_setup.py
```

This will check:
- ✓ All required packages are installed
- ✓ Data files exist in `data/raw/`
- ✓ Configuration file is valid
- ✓ All required modules exist

### Step 2: Install Dependencies (if needed)

If verification shows missing packages, install them:

```powershell
pip install -r requirements.txt
```

**Note:** You already have a `venv` directory, so you might want to activate it first:

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 3: Process Data (Feature Engineering)

This step processes raw CSV files into features for model training:

```powershell
python -m src.features.feature_pipeline
```

**What this does:**
- Loads all CSV files from `data/raw/`
- Calculates AQI from PM2.5 values
- Aggregates to daily level
- Creates lag features (previous 1-7 days)
- Creates rolling statistics (3-day and 7-day windows)
- Creates time-based features (day of week, month, etc.)
- Saves to `data/processed/processed_features.csv` and `data/processed/daily_history.csv`

**Expected output:**
```
Loading configuration...
Loading raw data from data/raw...
Saving processed features to data/processed/processed_features.csv...
Saving daily history to data/processed/daily_history.csv...
Feature pipeline completed successfully!
  - Daily history: XXX rows
  - Feature table: XXX rows
  - Features: XXX columns
```

### Step 4: Train Models

This step trains and compares multiple machine learning models:

```powershell
python scripts/train_models.py
```

**What this does:**
- Loads processed features
- Splits data into training (80%) and test (20%) sets
- Trains 5 models:
  1. Linear Regression
  2. Ridge Regression
  3. Random Forest
  4. XGBoost
  5. LightGBM
- Evaluates each model (RMSE, MAE, R²)
- Selects best model (lowest RMSE)
- Saves all models to `models/` directory
- Generates comparison report

**Expected output:**
```
============================================================
AQI Predictor - Model Training
============================================================
Loading data...
Preparing features and target...
Dataset shape: (XXX, XXX)
Number of features: XXX
Training set: XXX samples
Test set: XXX samples

Training models...
  Training LinearRegression...
    Test RMSE: X.XXXX
    Test MAE: X.XXXX
    Test R²: X.XXXX
  Training Ridge...
  Training RandomForest...
  Training XGBoost...
  Training LightGBM...

Best model: XGBoost
  Test RMSE: X.XXXX
  Test MAE: X.XXXX
  Test R²: X.XXXX

Saving models...
  Saved LinearRegression to models/linearregression_model.joblib
  Saved Ridge to models/ridge_model.joblib
  ...

Model comparison saved to models/model_comparison.json

============================================================
Training completed successfully!
============================================================
```

**Training time:** ~1-5 minutes depending on your system

### Step 5: Run Dashboard

Launch the interactive Streamlit dashboard:

```powershell
python run_dashboard.py
```

Or directly:

```powershell
streamlit run dashboard/app.py
```

**What happens:**
- Streamlit server starts
- Dashboard opens in your browser at `http://localhost:8501`
- You can:
  - Select different models from sidebar
  - View model performance metrics
  - See AQI predictions for next 3 days
  - Visualize predictions with interactive plots
  - View feature importance
  - Explore SHAP explainability

**To stop the dashboard:** Press `Ctrl+C` in the terminal

## 📋 Complete Workflow Summary

Here's the complete workflow in order:

```powershell
# 1. Verify setup
python verify_setup.py

# 2. Install dependencies (if needed)
pip install -r requirements.txt

# 3. Process data (feature engineering)
python -m src.features.feature_pipeline

# 4. Train models
python scripts/train_models.py

# 5. Run dashboard
python run_dashboard.py
```

## 🔧 Configuration

You can customize the project by editing `config/config.yaml`:

```yaml
data:
  raw_dir: "data/raw"                    # Raw data directory
  processed_dir: "data/processed"        # Processed data directory
  processed_features_file: "data/processed/processed_features.csv"
  daily_history_file: "data/processed/daily_history.csv"

model:
  target: "aqi"                          # Target variable
  forecast_days: 3                        # Number of days to forecast
  random_seed: 42                         # Random seed for reproducibility
  test_fraction: 0.2                      # Test set size (20%)
  min_training_rows: 90                   # Minimum training rows required
  max_lag: 7                              # Maximum lag features (1-7 days)
  rolling_windows: [3, 7]                 # Rolling window sizes
  select_k_best: 10                       # Number of best features to select
```

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: Features file not found"
**Solution:** Run the feature pipeline first:
```powershell
python -m src.features.feature_pipeline
```

### Issue: "FileNotFoundError: Model file not found"
**Solution:** Train models first:
```powershell
python scripts/train_models.py
```

### Issue: Import errors
**Solution:**
1. Make sure you're in the project root directory (`D:\AQI-Predictor`)
2. Install dependencies: `pip install -r requirements.txt`
3. Verify setup: `python verify_setup.py`

### Issue: Dashboard shows no data
**Solution:**
1. Check that `data/processed/processed_features.csv` exists
2. Verify the date range in the dashboard includes available data
3. Re-run the feature pipeline if needed

### Issue: Virtual environment not activated
**Solution:** Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

## 📊 What Each Step Produces

### After Step 3 (Feature Pipeline):
- `data/processed/processed_features.csv` - Features for training
- `data/processed/daily_history.csv` - Daily AQI history

### After Step 4 (Model Training):
- `models/linearregression_model.joblib` - Linear Regression model
- `models/ridge_model.joblib` - Ridge Regression model
- `models/randomforest_model.joblib` - Random Forest model
- `models/xgboost_model.joblib` - XGBoost model
- `models/lightgbm_model.joblib` - LightGBM model
- `models/*_metadata.json` - Model metadata (metrics, feature names)
- `models/model_comparison.json` - Comparison of all models

### After Step 5 (Dashboard):
- Interactive web dashboard at `http://localhost:8501`
- Real-time AQI predictions
- Model performance visualization
- Feature importance analysis
- SHAP explainability plots

## 🎯 Quick Reference

| Task | Command |
|------|---------|
| Verify setup | `python verify_setup.py` |
| Install dependencies | `pip install -r requirements.txt` |
| Process data | `python -m src.features.feature_pipeline` |
| Train models | `python scripts/train_models.py` |
| Run dashboard | `python run_dashboard.py` |
| Activate venv | `.\venv\Scripts\Activate.ps1` |

## 📝 Notes

- All paths are relative and cross-platform compatible
- The project uses fixed random seeds (42) for reproducibility
- Models are saved with metadata for easy loading
- The dashboard supports multiple models
- Feature engineering is consistent between training and prediction
- Time-based train/test split preserves temporal order

---

**Project Status:** ✅ Fully Functional and Ready to Run

For more details, see:
- `README.md` - Comprehensive documentation
- `QUICKSTART.md` - Quick start guide
- `PROJECT_ENHANCEMENTS.md` - Project enhancements summary


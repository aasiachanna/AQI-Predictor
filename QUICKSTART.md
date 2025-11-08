# Quick Start Guide

This guide will help you get the Pearls AQI Predictor up and running quickly.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Setup (5 minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Setup

```bash
python verify_setup.py
```

This will check that all packages are installed and files are in place.

## Running the Pipeline (10 minutes)

### Step 1: Process Data

```bash
python -m src.features.feature_pipeline
```

This processes raw data from `data/raw/` and creates features in `data/processed/`.

**Expected output:**
- `data/processed/processed_features.csv` - Features for model training
- `data/processed/daily_history.csv` - Daily aggregated AQI history

### Step 2: Train Models

```bash
python scripts/train_models.py
```

This trains 5 models and saves them to `models/` directory.

**Expected output:**
- Multiple model files (`.joblib`)
- Model metadata files (`.json`)
- `models/model_comparison.json` - Comparison of all models

**Training time:** ~1-5 minutes depending on your system

### Step 3: Run Dashboard

```bash
python run_dashboard.py
```

Or:

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`.

## Troubleshooting

### Issue: "FileNotFoundError: Features file not found"

**Solution:** Run the feature pipeline first:
```bash
python -m src.features.feature_pipeline
```

### Issue: "FileNotFoundError: Model file not found"

**Solution:** Train models first:
```bash
python scripts/train_models.py
```

### Issue: Import errors

**Solution:** 
1. Make sure you're in the project root directory
2. Install dependencies: `pip install -r requirements.txt`
3. Verify setup: `python verify_setup.py`

### Issue: Dashboard shows no data

**Solution:**
1. Check that `data/processed/processed_features.csv` exists
2. Verify the date range in the dashboard includes available data
3. Re-run the feature pipeline if needed

## Next Steps

- Explore the dashboard and try different models
- Check the SHAP explainability features
- Review model performance metrics
- Customize the configuration in `config/config.yaml`

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review the code comments for implementation details
- Open an issue if you encounter problems


"""Forecast utilities for AQI prediction."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import joblib

from .config import load_config
from .data_processing import build_forecast_feature_row


@dataclass
class ForecastResult:
    """Container for forecast results."""
    
    dates: Sequence[datetime]
    predictions: Sequence[float]
    model_name: str


def forecast_next_days(
    history_df: pd.DataFrame,
    forecast_days: int = 3,
    model_path: Path | None = None,
    model_name: str = "RandomForest"
) -> ForecastResult:
    """Generate AQI forecasts for the next N days.
    
    Args:
        history_df: DataFrame with historical AQI values (must have 'date' and 'aqi' columns)
        forecast_days: Number of days to forecast
        model_path: Path to the trained model file
        model_name: Name of the model to use (RandomForest, XGBoost, LightGBM)
    
    Returns:
        ForecastResult with dates and predictions
    """
    config = load_config()
    
    # Load model
    if model_path is None:
        models_dir = Path("models")
        model_filename = f"{model_name.lower()}_model.joblib"
        model_path = models_dir / model_filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    model = joblib.load(model_path)
    
    # Get historical AQI values
    if 'aqi' not in history_df.columns:
        raise ValueError("History DataFrame must contain 'aqi' column")
    
    history_df = history_df.sort_values('date')
    aqi_history = history_df['aqi'].values
    
    # Generate forecasts
    predictions = []
    dates = []
    current_history = list(aqi_history)
    
    for day in range(forecast_days):
        # Calculate next date
        last_date = pd.to_datetime(history_df['date'].max())
        next_date = last_date + timedelta(days=day + 1)
        dates.append(next_date)
        
        # Build feature row
        feature_row = build_forecast_feature_row(
            history_values=current_history,
            next_date=next_date,
            max_lag=config.model.max_lag,
            rolling_windows=config.model.rolling_windows
        )
        
        # Convert to DataFrame
        feature_df = pd.DataFrame([feature_row])
        
        # Select features (exclude date and target)
        exclude_cols = ['date', 'target_next_day']
        feature_cols = [c for c in feature_df.columns if c not in exclude_cols]
        X = feature_df[feature_cols].select_dtypes(include=[np.number])
        
        # Fill NaN with 0 (shouldn't happen, but safety)
        X = X.fillna(0)
        
        # Ensure feature order matches training
        if hasattr(model, 'feature_names_in_'):
            X = X[[c for c in model.feature_names_in_ if c in X.columns]]
        
        # Predict
        pred = model.predict(X)[0]
        predictions.append(float(pred))
        
        # Update history with prediction for next iteration
        current_history.append(pred)
    
    return ForecastResult(
        dates=dates,
        predictions=predictions,
        model_name=model_name
    )


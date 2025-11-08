"""Pearls AQI Predictor core package.

This package exposes high-level utilities to build the processed dataset,
train and evaluate forecasting models, and generate short-term AQI forecasts.
"""

from .config import ModelConfig, PipelineConfig, load_config
from .pipeline import run_training_pipeline
from .forecast import ForecastResult, forecast_next_days

__all__ = [
	"ModelConfig",
	"PipelineConfig",
	"load_config",
	"run_training_pipeline",
	"ForecastResult",
	"forecast_next_days",
]


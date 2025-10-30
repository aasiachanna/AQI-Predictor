import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class AQIPredictor:
	"""Loads a trained model and serves predictions for the dashboard."""

	def __init__(self, models_dir: str | Path | None = None):
		base_dir = Path(__file__).resolve().parents[2]
		self.models_dir = Path(models_dir) if models_dir else base_dir / "models"
		self.model_path = self.models_dir / "model.joblib"
		self.metadata_path = self.models_dir / "metadata.json"
		self.model = None
		self.feature_names_: list[str] | None = None
		self._load()

	def _load(self):
		if not self.model_path.exists():
			raise FileNotFoundError(
				f"Model file not found at {self.model_path}. Please run the training script to create it."
			)
		self.model = joblib.load(self.model_path)

		# Try to read feature names from model or metadata
		if hasattr(self.model, "feature_names_in_"):
			self.feature_names_ = list(self.model.feature_names_in_)
		elif self.metadata_path.exists():
			import json
			with open(self.metadata_path, "r", encoding="utf-8") as f:
				meta = json.load(f)
			self.feature_names_ = meta.get("feature_names")

	def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
		if self.feature_names_:
			missing = [c for c in self.feature_names_ if c not in df.columns]
			if missing:
				raise ValueError(f"Missing required feature columns: {missing}")
			return df[self.feature_names_]

		# Fallback: auto-select numeric columns excluding likely target columns
		numeric_df = df.select_dtypes(include=[np.number]).copy()
		for target_like in ["aqi", "pm25", "pm_25", "target"]:
			if target_like in numeric_df.columns:
				numeric_df = numeric_df.drop(columns=[target_like])
		return numeric_df

	def predict(self, features_df: pd.DataFrame) -> np.ndarray:
		if self.model is None:
			raise RuntimeError("Model is not loaded. Call _load() or reinitialize the predictor.")
		X = self._select_features(features_df)
		return self.model.predict(X)

import os
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


class AQIPredictor:
	"""Loads a trained model and serves predictions for the dashboard.
	
	Supports loading:
	- Default model: 'model.joblib' (backward compatibility)
	- Specific models: 'randomforest_model.joblib', 'xgboost_model.joblib', 'lightgbm_model.joblib'
	"""

	def __init__(self, models_dir: str | Path | None = None, model_name: str | None = None):
		base_dir = Path(__file__).resolve().parents[2]
		self.models_dir = Path(models_dir) if models_dir else base_dir / "models"
		self.model_name = model_name
		self.model = None
		self.feature_names_: list[str] | None = None
		self.model_metadata: dict | None = None
		self._load()

	def _load(self):
		"""Load model and metadata. Tries specific model first, then falls back to default."""
		# Determine model path
		if self.model_name:
			# Load specific model (e.g., 'ensemble_fast', 'xgboost_fast', 'lightgbm_fast')
			model_filename = f"{self.model_name.lower()}_model.joblib"
			metadata_filename = f"{self.model_name.lower()}_metadata.json"
			self.model_path = self.models_dir / model_filename
			self.metadata_path = self.models_dir / metadata_filename
			
			# Also try to load scaler and selector if they exist
			scaler_path = self.models_dir / f"{self.model_name.lower()}_scaler.joblib"
			selector_path = self.models_dir / f"{self.model_name.lower()}_selector.joblib"
		else:
			# Try default model first
			self.model_path = self.models_dir / "model.joblib"
			self.metadata_path = self.models_dir / "metadata.json"
			
			# If default doesn't exist, try to find any available model
			if not self.model_path.exists():
				available_models = list(self.models_dir.glob("*_model.joblib"))
				if available_models:
					# Use the first available model (prefer randomforest if exists)
					preferred = self.models_dir / "randomforest_model.joblib"
					if preferred.exists():
						self.model_path = preferred
						self.metadata_path = self.models_dir / "randomforest_metadata.json"
					else:
						self.model_path = available_models[0]
						# Infer metadata path
						model_stem = self.model_path.stem.replace("_model", "")
						self.metadata_path = self.models_dir / f"{model_stem}_metadata.json"
		
		if not self.model_path.exists():
			available = list(self.models_dir.glob("*_model.joblib"))
			available_str = ", ".join([p.stem.replace("_model", "") for p in available]) if available else "none"
			raise FileNotFoundError(
				f"Model file not found at {self.model_path}. "
				f"Available models: {available_str}. "
				"Please run the training script to create it."
			)
		
		self.model = joblib.load(self.model_path)
		
		# Try to load scaler and selector if they exist
		self.scaler = None
		self.selector = None
		scaler_path = self.models_dir / f"{self.model_name.lower()}_scaler.joblib" if self.model_name else None
		selector_path = self.models_dir / f"{self.model_name.lower()}_selector.joblib" if self.model_name else None
		
		if scaler_path and scaler_path.exists():
			try:
				self.scaler = joblib.load(scaler_path)
			except:
				pass
		if selector_path and selector_path.exists():
			try:
				self.selector = joblib.load(selector_path)
			except:
				pass

		# Try to read feature names from model or metadata
		self.feature_names_ = None
		self.model_metadata = None
		
		# First try to read from metadata file (most reliable)
		if self.metadata_path.exists():
			try:
				with open(self.metadata_path, "r", encoding="utf-8") as f:
					meta = json.load(f)
				self.feature_names_ = meta.get("feature_names")
				self.model_metadata = meta
			except Exception as e:
				print(f"Warning: Could not load metadata from {self.metadata_path}: {e}")
		
		# Fallback: try to get from model attribute (sklearn models)
		if self.feature_names_ is None and hasattr(self.model, "feature_names_in_"):
			self.feature_names_ = list(self.model.feature_names_in_)

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

	def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
		if self.model is None:
			raise RuntimeError("Model is not loaded. Call _load() or reinitialize the predictor.")
		
		# If we have a selector, we need ALL features that were used during training
		# The selector will then select the right features
		# So we need to get all features from processed_features, not just selected ones
		if self.selector is not None:
			# Get all numeric features (excluding target columns)
			X = features_df.select_dtypes(include=[np.number]).copy()
			# Remove target columns
			for target_like in ["target_next_day", "aqi", "pm25", "pm_25"]:
				if target_like in X.columns:
					X = X.drop(columns=[target_like])
		else:
			# If no selector, use feature_names_ if available
			X = self._select_features(features_df)
		
		# Apply scaler if available
		if self.scaler is not None:
			# Ensure all columns expected by scaler are present
			if hasattr(self.scaler, 'feature_names_in_'):
				expected_cols = list(self.scaler.feature_names_in_)
				missing = [c for c in expected_cols if c not in X.columns]
				if missing:
					# Add missing columns with default values based on feature type
					for col in missing:
						if 'aqi' in col and 'lag' not in col and 'roll' not in col and 'diff' not in col:
							X[col] = 50.0
						else:
							X[col] = 0.0
				# Reorder columns to match scaler expectations (critical!)
				X = X[expected_cols]
			else:
				# If no feature_names_in_, try to infer from training data shape
				# This shouldn't happen, but handle it gracefully
				pass
			
			X = pd.DataFrame(
				self.scaler.transform(X),
				columns=X.columns,
				index=X.index
			)
		
		# Apply feature selector if available
		if self.selector is not None:
			# Ensure all columns expected by selector are present
			if hasattr(self.selector, 'feature_names_in_'):
				expected_cols = list(self.selector.feature_names_in_)
				missing = [c for c in expected_cols if c not in X.columns]
				if missing:
					# Add missing columns with default values based on feature type
					for col in missing:
						if 'aqi' in col and 'lag' not in col and 'roll' not in col and 'diff' not in col:
							X[col] = 50.0
						else:
							X[col] = 0.0
				# Reorder columns to match selector expectations (critical!)
				X = X[expected_cols]
			
			X = pd.DataFrame(
				self.selector.transform(X),
				columns=[X.columns[i] for i in self.selector.get_support(indices=True)],
				index=X.index
			)
		
		return self.model.predict(X)

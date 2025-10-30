import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_CSV = BASE_DIR / "data" / "processed" / "processed_features.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def load_dataset() -> pd.DataFrame:
	if not FEATURES_CSV.exists():
		raise FileNotFoundError(
			f"Processed features not found at {FEATURES_CSV}. Run src/features/feature_pipeline.py first."
		)
	df = pd.read_csv(FEATURES_CSV)
	# Ensure consistent types
	if 'timestamp' in df.columns:
		df['timestamp'] = pd.to_datetime(df['timestamp'])
	return df


def make_splits(df: pd.DataFrame):
	# Target column
	if 'aqi' not in df.columns:
		raise ValueError("Expected target column 'aqi' in processed_features.csv")

	# Feature selection: numeric columns excluding target
	numeric = df.select_dtypes(include=[np.number]).copy()
	if 'aqi' in numeric.columns:
		numeric = numeric.drop(columns=['aqi'])

	# Impute missing values with column medians (robust for tiny datasets)
	if not numeric.empty:
		numeric = numeric.fillna(numeric.median(numeric_only=True))

	X = numeric
	y = df['aqi']

	# Drop rows with missing target only
	mask = y.notnull()
	X = X.loc[mask]
	y = y.loc[mask]

	n_samples = len(X)
	if n_samples < 2:
		raise ValueError(
			"Not enough samples to train (need at least 2 rows after preprocessing). "
			"Collect more raw data or run the pipeline over a larger period."
		)

	# Adapt test size to very small datasets
	if n_samples >= 10:
		test_size = 0.2
	elif n_samples >= 4:
		test_size = 0.5
	else:  # n_samples is 2 or 3
		test_size = 1  # leave-one-out style split

	return train_test_split(X, y, test_size=test_size, random_state=42)


def train_model(X_train, y_train) -> RandomForestRegressor:
	model = RandomForestRegressor(
		n_estimators=300,
		max_depth=None,
		random_state=42,
		n_jobs=-1,
	)
	model.fit(X_train, y_train)
	return model


def evaluate(model, X_test, y_test) -> dict:
	pred = model.predict(X_test)
	mae = mean_absolute_error(y_test, pred)
	# Compute RMSE without the 'squared' keyword for compatibility
	rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
	r2 = r2_score(y_test, pred) if len(y_test) > 1 else float('nan')
	return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_artifacts(model, feature_names: list[str], metrics: dict):
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, MODEL_PATH)
	metadata = {
		"feature_names": feature_names,
		"metrics": metrics,
	}
	with open(METADATA_PATH, "w", encoding="utf-8") as f:
		json.dump(metadata, f, indent=2)
	print(f"✅ Saved model to {MODEL_PATH}")
	print(f"📝 Saved metadata to {METADATA_PATH}")


def main():
	print("📦 Loading dataset...")
	df = load_dataset()

	print("🔹 Making train/test split...")
	X_train, X_test, y_train, y_test = make_splits(df)

	print(f"🧩 Using {X_train.shape[1]} features and {len(X_train)} training rows")

	print("🌲 Training RandomForestRegressor...")
	model = train_model(X_train, y_train)

	print("📈 Evaluating model...")
	metrics = evaluate(model, X_test, y_test)
	print("Metrics:")
	for k, v in metrics.items():
		print(f" - {k}: {v:.4f}" if isinstance(v, (int, float)) and v == v else f" - {k}: {v}")

	print("💾 Saving artifacts...")
	save_artifacts(model, feature_names=list(getattr(model, 'feature_names_in_', X_train.columns)), metrics=metrics)

	print("✅ Training complete.")


if __name__ == "__main__":
	main()

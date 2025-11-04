import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURES_CSV = BASE_DIR / "data" / "processed" / "processed_features.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"


def load_dataset() -> pd.DataFrame:
	if not FEATURES_CSV.exists():
		raise FileNotFoundError(f"Processed features not found at {FEATURES_CSV}")
	df = pd.read_csv(FEATURES_CSV)
	if 'timestamp' in df.columns:
		df['timestamp'] = pd.to_datetime(df['timestamp'])
	elif 'date' in df.columns:
		df['timestamp'] = pd.to_datetime(df['date'])
	else:
		raise ValueError("Expected a 'timestamp' or 'date' column in processed features.")
	return df


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2):
	df = df.sort_values('timestamp')
	n = len(df)
	if n < 10:
		raise ValueError("Need at least 10 rows for a time-based split.")
	cut = int((1 - test_frac) * n)
	train = df.iloc[:cut]
	test = df.iloc[cut:]
	return train, test


def prepare_xy(df: pd.DataFrame):
	if 'aqi' not in df.columns:
		raise ValueError("'aqi' column not found. Ensure AQICN history is integrated.")
	numeric = df.select_dtypes(include=[np.number]).copy()
	if 'aqi' in numeric.columns:
		numeric = numeric.drop(columns=['aqi'])
	numeric = numeric.fillna(numeric.median(numeric_only=True))
	X = numeric
	y = df['aqi']
	mask = y.notnull()
	return X.loc[mask], y.loc[mask]


def train_rf(X_train, y_train):
	model = RandomForestRegressor(
		n_estimators=400,
		max_depth=None,
		random_state=42,
		n_jobs=-1,
	)
	model.fit(X_train, y_train)
	return model


def evaluate(model, X_test, y_test):
	pred = model.predict(X_test)
	mae = mean_absolute_error(y_test, pred)
	rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
	r2 = r2_score(y_test, pred) if len(y_test) > 1 else float('nan')
	return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_artifacts(model, feature_names: list[str], metrics: dict):
	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	joblib.dump(model, MODEL_PATH)
	with open(METADATA_PATH, "w", encoding="utf-8") as f:
		json.dump({"feature_names": feature_names, "metrics": metrics}, f, indent=2)
	print(f"✅ Saved model to {MODEL_PATH}")
	print(f"📝 Saved metadata to {METADATA_PATH}")


def main():
	print("📦 Loading dataset...")
	df = load_dataset()
	print(f"Total rows: {len(df)} | Time span: {df['timestamp'].min()} → {df['timestamp'].max()}")

	print("⏱️ Time-based splitting...")
	train_df, test_df = chronological_split(df, test_frac=0.2)
	X_train, y_train = prepare_xy(train_df)
	X_test, y_test = prepare_xy(test_df)
	print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Features: {X_train.shape[1]}")

	print("🌲 Training RandomForest...")
	model = train_rf(X_train, y_train)

	print("📈 Evaluating...")
	metrics = evaluate(model, X_test, y_test)
	for k, v in metrics.items():
		print(f" - {k}: {v:.4f}" if isinstance(v, (int, float)) and v == v else f" - {k}: {v}")

	print("💾 Saving artifacts...")
	save_artifacts(model, feature_names=list(getattr(model, 'feature_names_in_', X_train.columns)), metrics=metrics)

	print("✅ Done.")


if __name__ == "__main__":
	main()

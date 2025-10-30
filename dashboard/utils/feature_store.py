# dashboard/utils/feature_store.py
import pandas as pd
import os
from pathlib import Path


def load_features(start_date, end_date):
	"""Load features from the local processed features store.

	This normalizes the timestamp column to a 'date' column for the dashboard
	and filters rows between the provided dates (inclusive).
	"""
	base_dir = Path(__file__).resolve().parents[2]
	candidates = [
		base_dir / "data" / "processed" / "processed_features.csv",
		Path.cwd() / "data" / "processed" / "processed_features.csv",
	]
	features_path = next((p for p in candidates if p.exists()), None)
	if features_path is None:
		raise FileNotFoundError(
			"Features file not found at data/processed/processed_features.csv. Please run the feature pipeline first."
		)

	df = pd.read_csv(features_path)
	# Normalize datetime column
	if 'timestamp' in df.columns:
		df['date'] = pd.to_datetime(df['timestamp'])
	elif 'date' in df.columns:
		df['date'] = pd.to_datetime(df['date'])
	else:
		raise ValueError("Expected a 'timestamp' or 'date' column in processed features.")

	# Ensure start/end are Timestamps for comparison
	start_ts = pd.to_datetime(start_date)
	end_ts = pd.to_datetime(end_date)

	mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
	return df.loc[mask].reset_index(drop=True)
"""Data loading and cleansing utilities for the Pearls AQI project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PM25_BREAKPOINTS = (
	(0.0, 12.0, 0, 50),
	(12.1, 35.4, 51, 100),
	(35.5, 55.4, 101, 150),
	(55.5, 150.4, 151, 200),
	(150.5, 250.4, 201, 300),
	(250.5, 350.4, 301, 400),
	(350.5, 500.4, 401, 500),
)


@dataclass
class RawDataset:
	"""Wrapper for the concatenated raw measurements."""

	dataframe: pd.DataFrame


@dataclass
class PreparedData:
	"""Container holding intermediate processed datasets."""

	daily_history: pd.DataFrame
	feature_table: pd.DataFrame


def _list_csv_files(raw_dir: Path) -> list[Path]:
	"""Return all CSV files within ``raw_dir`` sorted by name."""

	return sorted(path for path in raw_dir.glob("*.csv") if path.is_file())


def load_raw_dataset(raw_dir: Path) -> RawDataset:
	"""Load and concatenate all raw CSV files."""

	csv_files = _list_csv_files(raw_dir)
	if not csv_files:
		raise FileNotFoundError(
			f"No CSV files were found in '{raw_dir}'. Place raw files before running the pipeline."
		)

	frames: list[pd.DataFrame] = []
	for file_path in csv_files:
		frame = pd.read_csv(file_path)
		if "timestamp" not in frame.columns:
			raise ValueError(f"File '{file_path}' must contain a 'timestamp' column.")
		frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False, errors="coerce")
		frames.append(frame)

	combined = pd.concat(frames, ignore_index=True)
	combined = combined.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
	return RawDataset(dataframe=combined)


def pm25_to_aqi(pm25_value: float) -> float:
	"""Convert PM2.5 concentration to AQI using US EPA breakpoints."""

	if pm25_value is None or np.isnan(pm25_value):
		return np.nan

	for c_low, c_high, i_low, i_high in PM25_BREAKPOINTS:
		if c_low <= pm25_value <= c_high:
			slope = (i_high - i_low) / (c_high - c_low)
			return slope * (pm25_value - c_low) + i_low
	return PM25_BREAKPOINTS[-1][3]


def enrich_with_aqi(raw_dataset: RawDataset) -> pd.DataFrame:
	"""Add an ``aqi`` column derived from PM2.5 readings."""

	df = raw_dataset.dataframe.copy()
	if "pm2_5" not in df.columns:
		raise ValueError("Expected column 'pm2_5' in raw dataset to compute AQI target.")

	df["aqi"] = df["pm2_5"].astype(float).apply(pm25_to_aqi)
	return df


def aggregate_daily_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Aggregate the hourly dataset to daily level statistics."""

	df = df.copy()
	df["date"] = df["timestamp"].dt.floor("D")

	aggregations: dict[str, Iterable[str]] = {
		"aqi": ["mean"],
	}

	# Only aggregate numeric pollutant/weather columns if they exist.
	numeric_columns = df.select_dtypes(include=["number"]).columns
	optional_columns = [
		"pm2_5",
		"pm10",
		"ozone",
		"carbon_monoxide",
		"nitrogen_dioxide",
		"sulphur_dioxide",
		"temperature_2m",
		"relative_humidity_2m",
		"pressure_msl",
		"wind_speed_10m",
		"precipitation",
	]

	for column in optional_columns:
		if column in numeric_columns:
			aggregations[column] = ["mean"]

	daily = df.groupby("date").agg(aggregations)
	daily.columns = ["_".join(filter(None, col)).strip("_") for col in daily.columns]
	daily = daily.reset_index().sort_values("date")
	return daily


def create_supervised_features(
	daily_df: pd.DataFrame,
	max_lag: int = 7,
	rolling_windows: Sequence[int] | None = None,
) -> pd.DataFrame:
	"""Create a supervised learning table with lagged AQI features.

	The function produces an observation for each day ``t`` describing the previous
	``max_lag`` days and targets the AQI for day ``t + 1``.
	"""

	df = daily_df.copy()
	if "aqi_mean" in df.columns:
		df = df.rename(columns={"aqi_mean": "aqi"})
	elif "aqi" not in df.columns:
		raise ValueError("Daily dataframe must include an 'aqi' column.")

	rolling_windows = rolling_windows or (3, 7)

	for lag in range(1, max_lag + 1):
		lagged = df["aqi"].shift(lag)
		df[f"aqi_lag_{lag}"] = lagged

	# Rolling statistics exclude the current day by shifting first
	for window in rolling_windows:
		shifted = df["aqi"].shift(1)
		df[f"aqi_roll_mean_{window}"] = shifted.rolling(window=window, min_periods=1).mean()
		df[f"aqi_roll_std_{window}"] = shifted.rolling(window=window, min_periods=1).std().fillna(0.0)

	df["aqi_diff_1"] = df["aqi"].diff(1)
	df["aqi_diff_7"] = df["aqi"].diff(7)

	df["day_of_week"] = df["date"].dt.dayofweek
	df["month"] = df["date"].dt.month
	df["day_of_year"] = df["date"].dt.dayofyear
	df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

	# Harmonic features capture seasonal patterns
	period = 365.25
	df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / period)
	df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / period)

	# Add weather feature lags (temperature, humidity, pressure, wind, etc.)
	weather_features = [
		"temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean",
		"wind_speed_10m_mean", "precipitation_mean", "pm10_mean", "ozone_mean"
	]
	
	for weather_feat in weather_features:
		if weather_feat in df.columns:
			# Add lag features for weather (1-3 days)
			for lag in [1, 2, 3]:
				df[f"{weather_feat}_lag_{lag}"] = df[weather_feat].shift(lag)
			# Add rolling mean for weather
			df[f"{weather_feat}_roll_mean_3"] = df[weather_feat].shift(1).rolling(window=3, min_periods=1).mean()
			# Include current value
			df[f"{weather_feat}_current"] = df[weather_feat]

	# Target is next day's AQI
	df["target_next_day"] = df["aqi"].shift(-1)

	feature_cols = [
		c
		for c in df.columns
		if c
		in {
			"date",
			"aqi",
			"target_next_day",
		}
		or c.startswith("aqi_lag_")
		or c.startswith("aqi_roll_mean_")
		or c.startswith("aqi_roll_std_")
		or c.startswith("aqi_diff_")
		or c.startswith("temperature_2m_mean")
		or c.startswith("relative_humidity_2m_mean")
		or c.startswith("pressure_msl_mean")
		or c.startswith("wind_speed_10m_mean")
		or c.startswith("precipitation_mean")
		or c.startswith("pm10_mean")
		or c.startswith("ozone_mean")
		or c in {
			"day_of_week",
			"month",
			"day_of_year",
			"is_weekend",
			"day_of_year_sin",
			"day_of_year_cos",
		}
	]

	feature_df = df[feature_cols].copy()
	feature_df = feature_df.dropna().reset_index(drop=True)
	return feature_df


def build_prepared_data(
	raw_dir: Path,
	max_lag: int = 7,
	rolling_windows: Sequence[int] | None = None,
) -> PreparedData:
	"""Execute the core data preparation workflow."""

	raw_dataset = load_raw_dataset(raw_dir)
	enriched = enrich_with_aqi(raw_dataset)
	daily_history = aggregate_daily_features(enriched)
	feature_table = create_supervised_features(daily_history, max_lag=max_lag, rolling_windows=rolling_windows)
	return PreparedData(daily_history=daily_history, feature_table=feature_table)


def save_prepared_data(prepared: PreparedData, processed_dir: Path, features_path: Path, history_path: Path) -> None:
	"""Persist processed datasets to disk."""

	processed_dir.mkdir(parents=True, exist_ok=True)
	prepared.daily_history.to_csv(history_path, index=False)
	prepared.feature_table.to_csv(features_path, index=False)


def build_forecast_feature_row(
	history_values: Sequence[float],
	next_date: pd.Timestamp,
	max_lag: int,
	rolling_windows: Sequence[int] | None = None,
) -> dict[str, float | int | pd.Timestamp]:
	"""Construct a single feature row for forecasting the next day's AQI."""

	if len(history_values) < max_lag:
		raise ValueError(
			f"Expected at least {max_lag} historical values to compute lags; received {len(history_values)}."
		)

	rolling_windows = tuple(rolling_windows) if rolling_windows else (3, 7)
	values = list(history_values)

	feature_row: dict[str, float | int | pd.Timestamp] = {"date": next_date}

	for idx in range(1, max_lag + 1):
		feature_row[f"aqi_lag_{idx}"] = values[-idx]

	feature_row["aqi"] = values[-1]

	if len(values) >= 2:
		feature_row["aqi_diff_1"] = values[-1] - values[-2]
	else:
		feature_row["aqi_diff_1"] = 0.0

	if len(values) >= 7:
		feature_row["aqi_diff_7"] = values[-1] - values[-7]
	else:
		feature_row["aqi_diff_7"] = feature_row["aqi_diff_1"]

	for window in rolling_windows:
		window_values = values[-window:]
		feature_row[f"aqi_roll_mean_{window}"] = float(np.mean(window_values))
		feature_row[f"aqi_roll_std_{window}"] = float(np.std(window_values, ddof=0))

	feature_row["day_of_week"] = next_date.dayofweek
	feature_row["month"] = next_date.month
	feature_row["day_of_year"] = next_date.timetuple().tm_yday
	feature_row["is_weekend"] = int(next_date.dayofweek >= 5)
	period = 365.25
	feature_row["day_of_year_sin"] = np.sin(2 * np.pi * feature_row["day_of_year"] / period)
	feature_row["day_of_year_cos"] = np.cos(2 * np.pi * feature_row["day_of_year"] / period)
	feature_row["target_next_day"] = np.nan
	return feature_row


# dashboard/utils/feature_store.py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.pearls_aqi.config import load_config
from src.pearls_aqi.data_processing import build_forecast_feature_row


def build_comprehensive_forecast_features(aqi_history, weather_data, next_date, max_lag, rolling_windows):
	"""Build comprehensive forecast features including weather and interaction terms."""
	import numpy as np
	
	rolling_windows = tuple(rolling_windows) if rolling_windows else (3, 7)
	values = list(aqi_history)
	
	feature_row = {"date": next_date}
	
	# AQI features
	for idx in range(1, max_lag + 1):
		feature_row[f"aqi_lag_{idx}"] = values[-idx] if len(values) >= idx else values[0] if values else 50.0
	
	feature_row["aqi"] = values[-1] if values else 50.0
	
	# AQI differences
	if len(values) >= 2:
		feature_row["aqi_diff_1"] = values[-1] - values[-2]
	else:
		feature_row["aqi_diff_1"] = 0.0
	
	if len(values) >= 7:
		feature_row["aqi_diff_7"] = values[-1] - values[-7]
	else:
		feature_row["aqi_diff_7"] = feature_row["aqi_diff_1"]
	
	# AQI rolling statistics - ensure both mean and std for all windows
	for window in rolling_windows:
		window_values = values[-window:] if len(values) >= window else values
		feature_row[f"aqi_roll_mean_{window}"] = float(np.mean(window_values)) if window_values else 50.0
		feature_row[f"aqi_roll_std_{window}"] = float(np.std(window_values, ddof=0)) if len(window_values) > 1 else 0.0
	
	# Ensure all required rolling std features exist (even if not in rolling_windows)
	if 'aqi_roll_std_3' not in feature_row:
		window_values = values[-3:] if len(values) >= 3 else values
		feature_row['aqi_roll_std_3'] = float(np.std(window_values, ddof=0)) if len(window_values) > 1 else 0.0
	if 'aqi_roll_std_7' not in feature_row:
		window_values = values[-7:] if len(values) >= 7 else values
		feature_row['aqi_roll_std_7'] = float(np.std(window_values, ddof=0)) if len(window_values) > 1 else 0.0
	
	# Time features - ensure all are present
	feature_row["day_of_week"] = int(next_date.dayofweek)
	feature_row["month"] = int(next_date.month)
	feature_row["day_of_year"] = int(next_date.timetuple().tm_yday)
	feature_row["is_weekend"] = int(next_date.dayofweek >= 5)
	period = 365.25
	feature_row["day_of_year_sin"] = float(np.sin(2 * np.pi * feature_row["day_of_year"] / period))
	feature_row["day_of_year_cos"] = float(np.cos(2 * np.pi * feature_row["day_of_year"] / period))
	
	# Weather features - include base columns AND derived features
	weather_features = [
		"temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean",
		"wind_speed_10m_mean", "precipitation_mean", "pm10_mean", "ozone_mean"
	]
	
	if weather_data is not None and len(weather_data) > 0:
		for weather_feat in weather_features:
			if weather_feat in weather_data.columns:
				weather_values = weather_data[weather_feat].dropna().values
				if len(weather_values) > 0:
					# Base column (required by model)
					feature_row[weather_feat] = float(weather_values[-1])
					
					# Current value (derived feature)
					feature_row[f"{weather_feat}_current"] = float(weather_values[-1])
					
					# Lag features (1-3 days)
					for lag in [1, 2, 3]:
						if len(weather_values) >= lag:
							feature_row[f"{weather_feat}_lag_{lag}"] = float(weather_values[-lag])
						else:
							feature_row[f"{weather_feat}_lag_{lag}"] = float(weather_values[0]) if len(weather_values) > 0 else 0.0
					
					# Rolling mean
					window_values = weather_values[-3:] if len(weather_values) >= 3 else weather_values
					feature_row[f"{weather_feat}_roll_mean_3"] = float(np.mean(window_values)) if len(window_values) > 0 else 0.0
				else:
					# Default values if no weather data
					feature_row[weather_feat] = 0.0
					feature_row[f"{weather_feat}_current"] = 0.0
					for lag in [1, 2, 3]:
						feature_row[f"{weather_feat}_lag_{lag}"] = 0.0
					feature_row[f"{weather_feat}_roll_mean_3"] = 0.0
			else:
				# Default values if feature doesn't exist
				feature_row[weather_feat] = 0.0
				feature_row[f"{weather_feat}_current"] = 0.0
				for lag in [1, 2, 3]:
					feature_row[f"{weather_feat}_lag_{lag}"] = 0.0
				feature_row[f"{weather_feat}_roll_mean_3"] = 0.0
	else:
		# Default values if no weather data available
		for weather_feat in weather_features:
			feature_row[weather_feat] = 0.0
			feature_row[f"{weather_feat}_current"] = 0.0
			for lag in [1, 2, 3]:
				feature_row[f"{weather_feat}_lag_{lag}"] = 0.0
			feature_row[f"{weather_feat}_roll_mean_3"] = 0.0
	
	# Interaction features (AQI x AQI lags)
	aqi_features = ['aqi', 'aqi_lag_1', 'aqi_lag_2', 'aqi_lag_3', 'aqi_lag_4']
	aqi_feat_values = {feat: feature_row.get(feat, 50.0) for feat in aqi_features if feat in feature_row}
	
	# Generate interaction terms
	interactions = [
		('aqi', 'aqi_lag_1'),
		('aqi', 'aqi_lag_2'),
		('aqi_lag_1', 'aqi_lag_2'),
		('aqi_lag_1', 'aqi_lag_3'),
		('aqi_lag_2', 'aqi_lag_3'),
		('aqi_lag_2', 'aqi_lag_4')
	]
	
	for feat1, feat2 in interactions:
		if feat1 in aqi_feat_values and feat2 in aqi_feat_values:
			feature_row[f"{feat1}_x_{feat2}"] = aqi_feat_values[feat1] * aqi_feat_values[feat2]
		else:
			feature_row[f"{feat1}_x_{feat2}"] = 0.0
	
	return feature_row


def create_minimal_features(current_history, forecast_date, max_lag):
	"""Create minimal feature set as fallback."""
	import numpy as np
	
	feature_row = {
		'date': forecast_date,
		'aqi': current_history[-1] if current_history else 50.0,
	}
	
	# Add minimal lag features
	for lag in range(1, min(max_lag + 1, len(current_history) + 1)):
		feature_row[f'aqi_lag_{lag}'] = current_history[-lag] if len(current_history) >= lag else 50.0
	
	# Fill missing lags
	for lag in range(len(current_history) + 1, max_lag + 1):
		feature_row[f'aqi_lag_{lag}'] = 50.0
	
	# Add default values for other required features
	feature_row['aqi_diff_1'] = 0.0
	feature_row['aqi_diff_7'] = 0.0
	feature_row['aqi_roll_mean_3'] = 50.0
	feature_row['aqi_roll_std_3'] = 0.0
	feature_row['aqi_roll_mean_7'] = 50.0
	feature_row['aqi_roll_std_7'] = 0.0
	
	# Time features - ensure all are present with correct types
	feature_row['day_of_week'] = int(forecast_date.dayofweek)
	feature_row['month'] = int(forecast_date.month)
	feature_row['day_of_year'] = int(forecast_date.timetuple().tm_yday)
	feature_row['is_weekend'] = int(forecast_date.dayofweek >= 5)
	period = 365.25
	feature_row['day_of_year_sin'] = float(np.sin(2 * np.pi * feature_row['day_of_year'] / period))
	feature_row['day_of_year_cos'] = float(np.cos(2 * np.pi * feature_row['day_of_year'] / period))
	
	# Weather features (defaults) - include base columns
	weather_features = [
		"temperature_2m_mean", "relative_humidity_2m_mean", "pressure_msl_mean",
		"wind_speed_10m_mean", "precipitation_mean", "pm10_mean", "ozone_mean"
	]
	for weather_feat in weather_features:
		feature_row[weather_feat] = 0.0  # Base column (required)
		feature_row[f"{weather_feat}_current"] = 0.0
		for lag in [1, 2, 3]:
			feature_row[f"{weather_feat}_lag_{lag}"] = 0.0
		feature_row[f"{weather_feat}_roll_mean_3"] = 0.0
	
	# Interaction features (defaults)
	interactions = [
		('aqi', 'aqi_lag_1'),
		('aqi', 'aqi_lag_2'),
		('aqi_lag_1', 'aqi_lag_2'),
		('aqi_lag_1', 'aqi_lag_3'),
		('aqi_lag_2', 'aqi_lag_3'),
		('aqi_lag_2', 'aqi_lag_4')
	]
	for feat1, feat2 in interactions:
		feature_row[f"{feat1}_x_{feat2}"] = 0.0
	
	return feature_row


def load_features(start_date, end_date):
	"""Load features from the local processed features store.
	
	If the requested date range extends beyond available data, 
	generates forecast features for future dates.

	This normalizes the timestamp column to a 'date' column for the dashboard
	and filters/extends rows between the provided dates (inclusive).
	"""
	base_dir = Path(__file__).resolve().parents[2]
	candidates = [
		base_dir / "data" / "processed" / "processed_features.csv",
		Path.cwd() / "data" / "processed" / "processed_features.csv",
	]
	features_path = next((p for p in candidates if p.exists()), None)
	
	# Also try daily_history as fallback
	if features_path is None:
		history_candidates = [
			base_dir / "data" / "processed" / "daily_history.csv",
			Path.cwd() / "data" / "processed" / "daily_history.csv",
		]
		history_path = next((p for p in history_candidates if p.exists()), None)
		if history_path:
			df = pd.read_csv(history_path)
			if 'date' in df.columns:
				df['date'] = pd.to_datetime(df['date'])
				if 'aqi_mean' in df.columns:
					df['aqi'] = df['aqi_mean']
				# Use daily history as base
				features_path = history_path
			else:
				raise FileNotFoundError(
					"Features file not found. Please run the feature pipeline first: "
					"python -m src.features.feature_pipeline"
				)
		else:
			raise FileNotFoundError(
				"Features file not found. Please run the feature pipeline first: "
				"python -m src.features.feature_pipeline"
			)
	
	if features_path:
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
	
	# Load daily_history early for potential merging
	history_candidates = [
		base_dir / "data" / "processed" / "daily_history.csv",
		Path.cwd() / "data" / "processed" / "daily_history.csv",
	]
	history_path = next((p for p in history_candidates if p.exists()), None)
	daily_history = None
	if history_path:
		daily_history = pd.read_csv(history_path)
		if 'date' in daily_history.columns:
			daily_history['date'] = pd.to_datetime(daily_history['date'])
			if 'aqi_mean' in daily_history.columns:
				daily_history['aqi'] = daily_history['aqi_mean']
	
	# Filter existing data - keep ALL columns from processed_features
	if len(df) > 0:
		mask = (df['date'] >= start_ts) & (df['date'] <= end_ts)
		df_filtered = df.loc[mask].copy()
		latest_date = df['date'].max()
		
		# Ensure ALL required features exist (68 features expected by scaler/selector)
		# Load the expected features from selector if available
		try:
			import joblib
			selector_path = base_dir / "models" / "ensemble_fast_selector.joblib"
			if not selector_path.exists():
				selector_path = Path.cwd() / "models" / "ensemble_fast_selector.joblib"
			
			if selector_path.exists():
				selector = joblib.load(selector_path)
				if hasattr(selector, 'feature_names_in_'):
					required_features = list(selector.feature_names_in_)
					# Add missing features with default values
					for feat in required_features:
						if feat not in df_filtered.columns:
							# Try to get from daily_history if it's a base weather feature
							if daily_history is not None and feat in daily_history.columns:
								df_filtered = df_filtered.merge(
									daily_history[['date', feat]], 
									on='date', 
									how='left'
								)
								df_filtered[feat] = df_filtered[feat].fillna(0.0)
							else:
								# Set default value
								df_filtered[feat] = 0.0
		except Exception as e:
			# If we can't load selector, ensure at least base features exist
			required_base_cols = ['pm10_mean', 'temperature_2m_mean', 'pressure_msl_mean', 
			                      'wind_speed_10m_mean', 'relative_humidity_2m_mean', 
			                      'precipitation_mean', 'ozone_mean', 'aqi_diff_1', 
			                      'aqi_roll_std_3', 'aqi_roll_std_7', 'day_of_week', 
			                      'day_of_year', 'month', 'is_weekend', 'day_of_year_sin', 
			                      'day_of_year_cos']
			for col in required_base_cols:
				if col not in df_filtered.columns:
					if daily_history is not None and col in daily_history.columns:
						df_filtered = df_filtered.merge(
							daily_history[['date', col]], 
							on='date', 
							how='left'
						)
						df_filtered[col] = df_filtered[col].fillna(0.0)
					else:
						df_filtered[col] = 0.0
	else:
		df_filtered = pd.DataFrame()
		latest_date = None
	
	# If we need future dates, generate forecast features
	if latest_date is None or end_ts > latest_date:
		# Load config
		config = load_config()
		
		# daily_history already loaded above
		
		# Get historical AQI values for forecasting
		if 'aqi' not in df.columns and 'aqi_mean' in df.columns:
			aqi_col = 'aqi_mean'
		elif 'aqi' in df.columns:
			aqi_col = 'aqi'
		else:
			aqi_col = None
		
		if aqi_col and len(df) > 0:
			# Get recent data (use processed_features if available, otherwise daily_history)
			if len(df) > 0 and 'aqi' in df.columns:
				recent_data = df.tail(config.model.max_lag * 2).copy()
			elif daily_history is not None:
				recent_data = daily_history.tail(config.model.max_lag * 2).copy()
				if 'aqi_mean' in recent_data.columns:
					recent_data['aqi'] = recent_data['aqi_mean']
			else:
				recent_data = pd.DataFrame()
			
			# Get recent AQI values
			if len(recent_data) > 0 and aqi_col in recent_data.columns:
				aqi_history = recent_data[aqi_col].dropna().tail(config.model.max_lag * 2).values.tolist()
			else:
				aqi_history = []
			
			if len(aqi_history) < config.model.max_lag:
				# Not enough history, pad with mean
				mean_aqi = df[aqi_col].mean() if len(df) > 0 and aqi_col in df.columns else 50.0
				aqi_history = [mean_aqi] * config.model.max_lag + aqi_history
			
			# Generate features for future dates
			forecast_start = latest_date + timedelta(days=1) if latest_date else start_ts
			forecast_dates = pd.date_range(
				start=forecast_start,
				end=end_ts,
				freq='D'
			)
			
			forecast_rows = []
			current_history = aqi_history.copy()
			
			# Get recent weather data for feature generation
			weather_data = None
			if daily_history is not None and len(daily_history) > 0:
				weather_data = daily_history.tail(config.model.max_lag * 2).copy()
			
			for forecast_date in forecast_dates:
				try:
					# Build comprehensive feature row with weather features
					feature_row = build_comprehensive_forecast_features(
						aqi_history=current_history,
						weather_data=weather_data,
						next_date=forecast_date,
						max_lag=config.model.max_lag,
						rolling_windows=config.model.rolling_windows
					)
					forecast_rows.append(feature_row)
					# Update history (use last known AQI or average)
					if current_history:
						current_history.append(current_history[-1])
					else:
						current_history.append(50.0)
				except Exception as e:
					# If forecast feature generation fails, create minimal features
					feature_row = create_minimal_features(
						current_history=current_history,
						forecast_date=forecast_date,
						max_lag=config.model.max_lag
					)
					forecast_rows.append(feature_row)
			
			if forecast_rows:
				forecast_df = pd.DataFrame(forecast_rows)
				# Combine with existing data
				if len(df_filtered) > 0:
					df_filtered = pd.concat([df_filtered, forecast_df], ignore_index=True)
				else:
					df_filtered = forecast_df
	
	# Sort by date
	if len(df_filtered) > 0:
		df_filtered = df_filtered.sort_values('date').reset_index(drop=True)
		# Ensure date column is present
		if 'date' not in df_filtered.columns:
			df_filtered['date'] = pd.to_datetime(start_date)
		
		# Ensure ALL 68 required features are present (load from selector)
		try:
			import joblib
			selector_path = base_dir / "models" / "ensemble_fast_selector.joblib"
			if not selector_path.exists():
				selector_path = Path.cwd() / "models" / "ensemble_fast_selector.joblib"
			
			if selector_path.exists():
				selector = joblib.load(selector_path)
				if hasattr(selector, 'feature_names_in_'):
					required_features = list(selector.feature_names_in_)
					# Add any missing features with default values
					for feat in required_features:
						if feat not in df_filtered.columns:
							# Try to get from daily_history if it's a base weather feature
							if daily_history is not None and feat in daily_history.columns:
								df_filtered = df_filtered.merge(
									daily_history[['date', feat]], 
									on='date', 
									how='left'
								)
								df_filtered[feat] = df_filtered[feat].fillna(0.0)
							else:
								# Set default value based on feature type
								if 'lag' in feat or 'roll' in feat or 'diff' in feat:
									df_filtered[feat] = 0.0
								elif 'aqi' in feat and 'lag' not in feat and 'roll' not in feat:
									df_filtered[feat] = 50.0
								else:
									df_filtered[feat] = 0.0
		except Exception as e:
			# If we can't load selector, at least ensure critical features exist
			pass
	
	return df_filtered
# pyright: reportMissingImports=false
import json, glob, os
import pandas as pd
from datetime import datetime
from pathlib import Path

# --- Base directory resolution ---
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_CSV = BASE_DIR / "data" / "processed" / "processed_features.csv"
AQICN_CLEAN = BASE_DIR / "data" / "processed" / "aqicn_clean.csv"

print(f"\n🔍 Using RAW_DIR: {RAW_DIR}")

print(f"📂 Files found in RAW_DIR: {[p.name for p in RAW_DIR.glob('*')]}")

# --- AQI from PM2.5 helper (US EPA breakpoints) ---
def pm25_to_aqi(pm25: float) -> float:
	if pm25 is None or pd.isna(pm25):
		return float('nan')
	# Breakpoints: (Clow, Chigh, Ilow, Ihigh)
	bps = [
		(0.0, 12.0, 0, 50),
		(12.1, 35.4, 51, 100),
		(35.5, 55.4, 101, 150),
		(55.5, 150.4, 151, 200),
		(150.5, 250.4, 201, 300),
		(250.5, 350.4, 301, 400),
		(350.5, 500.4, 401, 500),
	]
	for Clow, Chigh, Ilow, Ihigh in bps:
		if Clow <= pm25 <= Chigh:
			return (Ihigh - Ilow) / (Chigh - Clow) * (pm25 - Clow) + Ilow
	return 500.0

# --- AQICN parser ---
def parse_aqicn_file(filepath):
	with open(filepath, 'r', encoding='utf-8') as f:
		j = json.load(f)
	if j.get("status") != "ok":
		return None
	data = j["data"]
	ts = data.get("time", {}).get("utc") or data.get("time", {}).get("s")
	record = {"timestamp": ts, "aqi": data.get("aqi")}
	iaqi = data.get("iaqi", {})
	for k, v in iaqi.items():
		record[k] = v.get("v")
	return record

# --- OpenWeather parser ---
def parse_openweather_file(filepath):
	with open(filepath, 'r', encoding='utf-8') as f:
		j = json.load(f)

	# Support both "current" and top-level OpenWeather formats
	if "current" in j:
		cur = j["current"]
	else:
		cur = j

	ts = datetime.utcfromtimestamp(cur.get("dt")).isoformat()
	main = cur.get("main", {})
	wind = cur.get("wind", {})

	rec = {
		"timestamp": ts,
		"temp": main.get("temp"),
		"humidity": main.get("humidity"),
		"pressure": main.get("pressure"),
		"wind_speed": wind.get("speed"),
	}
	return rec

# --- Open-Meteo air quality CSV loader (backfill) ---
def load_openmeteo_air_quality():
	csv_files = sorted(glob.glob(str(RAW_DIR / "*_aq_data_*.csv")))
	if not csv_files:
		return None
	dfs = []
	for f in csv_files:
		df = pd.read_csv(f)
		if 'date' not in df.columns:
			continue
		df['timestamp'] = pd.to_datetime(df['date'])
		# Compute AQI from PM2.5 if available
		pm_col = None
		for c in ['air_quality_pm2_5', 'pm2_5', 'pm2_5_concentration']:
			if c in df.columns:
				pm_col = c
				break
		if pm_col:
			df['aqi'] = df[pm_col].apply(pm25_to_aqi)
		# Keep key columns and any numeric features
		numeric = df.select_dtypes(include=['number'])
		numeric['timestamp'] = df['timestamp']
		dfs.append(numeric)
	if not dfs:
		return None
	merged = pd.concat(dfs, axis=0, ignore_index=True)
	merged = merged.sort_values('timestamp').drop_duplicates('timestamp')
	return merged

# --- Load cleaned AQICN history if available ---
def load_aqicn_clean():
	if AQICN_CLEAN.exists():
		df = pd.read_csv(AQICN_CLEAN)
		df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
		df = df.dropna(subset=['timestamp', 'aqi'])
		return df.sort_values('timestamp').drop_duplicates('timestamp')
	return None

# --- Feature engineering: time, lag, rolling stats ---
def create_time_and_lag_features(df):
	df = df.copy()
	df['hour'] = df['timestamp'].dt.hour
	df['dayofweek'] = df['timestamp'].dt.dayofweek
	df['month'] = df['timestamp'].dt.month
	df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

	df = df.sort_values('timestamp')

	# Lag features (previous AQI values)
	df['aqi_lag1'] = df['aqi'].shift(1)
	df['aqi_lag6'] = df['aqi'].shift(6)
	df['aqi_lag24'] = df['aqi'].shift(24)

	# Rolling statistics
	df['aqi_roll_mean_6'] = df['aqi'].rolling(window=6, min_periods=1).mean()
	df['aqi_roll_std_24'] = df['aqi'].rolling(window=24, min_periods=1).std().fillna(0)

	# Change rate (difference from last hour)
	df['aqi_change_rate'] = (df['aqi'] - df['aqi_lag1']) / (df['aqi_lag1'] + 1e-9)

	return df

# --- Build pipeline ---
def build_features():
	# Option A: Prefer cleaned AQICN + covariates (Open‑Meteo; optionally OpenWeather)
	aqicn_hist = load_aqicn_clean()
	if aqicn_hist is not None:
		print(f"📊 Loaded AQICN history: {len(aqicn_hist)} rows")
	covariates = None

	# Build covariates from Open‑Meteo
	df_om = load_openmeteo_air_quality()
	if df_om is not None and not df_om.empty:
		covariates = df_om.copy()
		print(f"🌦️ Loaded Open-Meteo covariates: {len(covariates)} rows")
	else:
		print("⚠️ No Open-Meteo covariates found. Using AQICN history only.")

	# Optionally enrich covariates with OpenWeather
	aq_files = sorted(glob.glob(str(RAW_DIR / "aqicn_*.json")))
	ow_files = sorted(glob.glob(str(RAW_DIR / "openweather_*.json")))
	print(f"📊 Found {len(aq_files)} AQICN files")
	print(f"🌦️  Found {len(ow_files)} OpenWeather files")

	if ow_files and covariates is not None:
		rows = [parse_openweather_file(f) for f in ow_files if parse_openweather_file(f)]
		df_ow = pd.DataFrame(rows)
		df_ow['timestamp'] = pd.to_datetime(df_ow['timestamp'])
		df_ow = df_ow.sort_values('timestamp').drop_duplicates('timestamp')
		covariates = pd.merge_asof(
			covariates.sort_values('timestamp'),
			df_ow.sort_values('timestamp'),
			on='timestamp',
			tolerance=pd.Timedelta('3h'),
			direction='nearest'
		)

	if aqicn_hist is not None:
		if covariates is not None:
			# Align by nearest hour to combine true AQI with covariates
			df = pd.merge_asof(
				aqicn_hist.sort_values('timestamp'),
				covariates.sort_values('timestamp'),
				on='timestamp',
				tolerance=pd.Timedelta('1h'),
				direction='nearest'
			)
			# Ensure we have AQI as target
			if 'aqi' not in df.columns:
				raise ValueError("AQICN history loaded but 'aqi' column missing after merge.")
		else:
			# Use AQICN history alone (minimal features)
			df = aqicn_hist.copy()
			print("ℹ️ Using AQICN history without weather covariates.")
	else:
		# Fallback: use previous logic combining AQICN nowcast + OpenWeather + Open‑Meteo with proxy
		df_list = []
		if aq_files and ow_files:
			rows = [parse_aqicn_file(f) for f in aq_files if parse_aqicn_file(f)]
			df_aq = pd.DataFrame(rows)
			df_aq['timestamp'] = pd.to_datetime(df_aq['timestamp'])
			df_aq = df_aq.sort_values('timestamp').drop_duplicates('timestamp')

			rows = [parse_openweather_file(f) for f in ow_files if parse_openweather_file(f)]
			df_ow = pd.DataFrame(rows)
			df_ow['timestamp'] = pd.to_datetime(df_ow['timestamp'])
			df_ow = df_ow.sort_values('timestamp').drop_duplicates('timestamp')

			df_main = pd.merge_asof(
				df_aq.sort_values('timestamp'),
				df_ow.sort_values('timestamp'),
				on='timestamp',
				tolerance=pd.Timedelta('3h'),
				direction='nearest'
			)
			if not df_main.empty:
				df_list.append(df_main)
		if df_om is not None and not df_om.empty:
			print(f"🕘 Added Open-Meteo backfill rows: {len(df_om)}")
			df_list.append(df_om)
		if not df_list:
			print("⚠️ No data found to build features!")
			return
		df = pd.concat(df_list, axis=0, ignore_index=True)
		df = df.sort_values('timestamp').drop_duplicates('timestamp')
		if 'aqi' not in df.columns:
			pm_like = None
			for c in ['air_quality_pm2_5', 'pm2_5', 'pm25']:
				if c in df.columns:
					pm_like = c
					break
			if pm_like:
				df['aqi'] = df[pm_like].apply(pm25_to_aqi)
			else:
				raise ValueError("Could not determine target 'aqi' from available data.")

	# Add time-based and lag features
	df = create_time_and_lag_features(df)

	# Save
	os.makedirs(OUT_CSV.parent, exist_ok=True)
	df.to_csv(OUT_CSV, index=False)
	print(f"✅ Saved processed features: {OUT_CSV}")
	print(f"📈 Rows: {len(df)}, Columns: {len(df.columns)}")

	return df


if __name__ == "__main__":
	build_features()

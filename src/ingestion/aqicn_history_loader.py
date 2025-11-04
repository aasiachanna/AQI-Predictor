# pyright: reportMissingImports=false
import pandas as pd
from pathlib import Path
from typing import List

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "aqicn_history"
OUT_CSV = Path(__file__).resolve().parents[2] / "data" / "processed" / "aqicn_clean.csv"


def _find_time_column(df: pd.DataFrame) -> str:
	for c in ["timestamp", "time", "date", "datetime", "utc", "published_at"]:
		if c in df.columns:
			return c
	raise ValueError("Could not find a time column in AQICN CSV. Expected one of: timestamp,time,date,datetime,utc,published_at")


def _find_aqi_column(df: pd.DataFrame) -> str | None:
	for c in ["aqi", "AQI", "aqi_value", "aqi_index"]:
		if c in df.columns:
			return c
	return None


def _find_pm25_column(df: pd.DataFrame) -> str | None:
	candidates = ["pm25", "PM25", "pm2_5", "PM2_5", "pm2.5", "PM2.5", "air_quality_pm2_5"]
	for c in candidates:
		if c in df.columns:
			return c
	return None


def _pm25_to_aqi(pm25: float) -> float:
	if pm25 is None or pd.isna(pm25):
		return float('nan')
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


def load_and_clean(csv_paths: List[Path]) -> pd.DataFrame:
	frames = []
	for p in csv_paths:
		df = pd.read_csv(p)
		# detect time and aqi columns
		tcol = _find_time_column(df)
		acol = _find_aqi_column(df)

		df = df.rename(columns={tcol: "timestamp"})
		df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

		if acol is not None:
			df = df.rename(columns={acol: "aqi"})
		else:
			# Fallback: compute AQI from PM2.5 if present
			pm_col = _find_pm25_column(df)
			if pm_col is None:
				print(f"⚠️ Warning: {p.name} has no AQI or PM2.5 column. Skipping.")
				continue
			df["aqi"] = df[pm_col].apply(_pm25_to_aqi)
			print(f"ℹ️ Computed AQI from {pm_col} for {p.name}")

		df = df.dropna(subset=["timestamp", "aqi"])  # keep only valid rows
		# Keep optional pollutants if they exist
		keep_extra = [c for c in ["pm25", "pm10", "o3", "no2", "so2", "co"] if c in df.columns]
		frames.append(df[["timestamp", "aqi"] + keep_extra])

	if not frames:
		raise FileNotFoundError("No valid AQICN history CSVs loaded.")

	full = pd.concat(frames, ignore_index=True)
	# Resample/aggregate to hourly: for multiple readings per hour, take mean
	full["timestamp"] = full["timestamp"].dt.floor("h")
	agg = {"aqi": "mean"}
	for c in [col for col in full.columns if col not in ["timestamp", "aqi"]]:
		agg[c] = "mean"
	full = full.groupby("timestamp", as_index=False).agg(agg).sort_values("timestamp")
	return full


def write_clean_csv(df: pd.DataFrame) -> Path:
	OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(OUT_CSV, index=False)
	return OUT_CSV


def main():
	csvs = sorted(RAW_DIR.glob("*.csv"))
	if not csvs:
		raise FileNotFoundError(f"No AQICN history CSVs found in {RAW_DIR}")
	df = load_and_clean(csvs)
	path = write_clean_csv(df)
	print(f"✅ Wrote cleaned AQICN to {path} with {len(df)} rows")


if __name__ == "__main__":
	main()

# scripts/fetch_historical_data.py
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_air_quality(latitude, longitude, start_date, end_date):
    """Fetch air quality data from Open-Meteo API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': ['pm2_5', 'pm10', 'ozone', 'carbon_monoxide', 'nitrogen_dioxide', 'sulphur_dioxide'],
        'start_date': start_date,
        'end_date': end_date,
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['hourly'])
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.drop('time', axis=1)
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return None

def fetch_weather(latitude, longitude, start_date, end_date):
    """Fetch weather data from Open-Meteo API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': ['temperature_2m', 'relative_humidity_2m', 'pressure_msl', 
                  'wind_speed_10m', 'wind_direction_10m', 'precipitation'],
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data['hourly'])
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.drop('time', axis=1)
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching weather data: {e}")
        return None

def collect_historical_data(latitude, longitude, start_date, end_date, output_dir='data/raw'):
    """Collect historical air quality and weather data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_date:
        batch_end = min(current_date + timedelta(days=89), end_date)  # 90 days per batch
        
        logger.info(f"Fetching data from {current_date.date()} to {batch_end.date()}")
        
        # Fetch data
        aq_data = fetch_air_quality(
            latitude, longitude,
            current_date.strftime('%Y-%m-%d'),
            batch_end.strftime('%Y-%m-%d')
        )
        
        weather_data = fetch_weather(
            latitude, longitude,
            current_date.strftime('%Y-%m-%d'),
            batch_end.strftime('%Y-%m-%d')
        )
        
        # Merge and save
        if aq_data is not None and weather_data is not None:
            merged_data = pd.merge(aq_data, weather_data, on='timestamp', how='outer')
            
            # Save batch
            filename = f"air_quality_weather_{current_date.date()}_to_{batch_end.date()}.csv"
            filepath = output_dir / filename
            merged_data.to_csv(filepath, index=False)
            logger.info(f"Saved data to {filepath}")
        
        # Move to next batch
        current_date = batch_end + timedelta(days=1)
        time.sleep(1)  # Be nice to the API

if __name__ == "__main__":
    # Example: Collect data for Sukkur, Pakistan
    collect_historical_data(
        latitude=27.7136,
        longitude=68.8489,
        start_date='2024-01-01',
        end_date='2024-12-31'
    )
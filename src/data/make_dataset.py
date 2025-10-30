import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file."""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

def load_raw_data(data_dir):
    """Load and combine all raw data files."""
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.csv"))
    
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file, parse_dates=['timestamp'])
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {file.name}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        raise ValueError("No valid data files could be loaded")
    
    return pd.concat(dfs, ignore_index=True)

def clean_data(df):
    """Clean the raw data."""
    # Remove duplicates
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp')
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(method='bfill')
    
    return df

def save_processed_data(df, output_path):
    """Save processed data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed data to {output_path}")

def process_data():
    """Main function to process the data."""
    try:
        config = load_config()
        data_config = config['data']
        
        logger.info("Loading raw data...")
        df = load_raw_data(data_config['raw_data_path'])
        
        logger.info("Cleaning data...")
        df = clean_data(df)
        
        logger.info("Saving processed data...")
        save_processed_data(df, Path(data_config['processed_path']) / "processed_data.csv")
        
        logger.info("Data processing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error in data processing: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    process_data()
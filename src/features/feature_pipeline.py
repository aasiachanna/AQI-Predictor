"""Feature pipeline to process raw data into features for model training."""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pearls_aqi.config import load_config
from src.pearls_aqi.data_processing import (
    build_prepared_data,
    save_prepared_data
)


def main():
    """Run the feature pipeline."""
    print("Loading configuration...")
    config = load_config()
    
    print(f"Loading raw data from {config.data.raw_dir}...")
    prepared = build_prepared_data(
        raw_dir=config.data.raw_dir,
        max_lag=config.model.max_lag,
        rolling_windows=config.model.rolling_windows
    )
    
    print(f"Saving processed features to {config.data.processed_features_file}...")
    print(f"Saving daily history to {config.data.daily_history_file}...")
    save_prepared_data(
        prepared=prepared,
        processed_dir=config.data.processed_dir,
        features_path=config.data.processed_features_file,
        history_path=config.data.daily_history_file
    )
    
    print(f"Feature pipeline completed successfully!")
    print(f"  - Daily history: {len(prepared.daily_history)} rows")
    print(f"  - Feature table: {len(prepared.feature_table)} rows")
    print(f"  - Features: {len(prepared.feature_table.columns)} columns")


if __name__ == "__main__":
    main()


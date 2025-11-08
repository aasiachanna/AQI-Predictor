"""Training pipeline for AQI prediction models."""

from pathlib import Path
import sys
import importlib.util

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import load_config
from .data_processing import build_prepared_data, save_prepared_data


def run_training_pipeline(config_path: Path | None = None):
    """Run the complete training pipeline.
    
    This function:
    1. Loads configuration
    2. Processes raw data into features
    3. Trains and evaluates models
    4. Saves the best model
    
    Args:
        config_path: Optional path to config file
    
    Note: This function expects to be run from the project root directory.
    For standalone execution, use the scripts directly:
    - python -m src.features.feature_pipeline
    - python scripts/train_models.py
    """
    # Import train_all_models dynamically to handle path issues
    train_script_path = project_root / "scripts" / "train_models.py"
    if not train_script_path.exists():
        raise FileNotFoundError(
            f"Training script not found at {train_script_path}. "
            "Please run from the project root directory."
        )
    
    spec = importlib.util.spec_from_file_location("train_models", train_script_path)
    train_models_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_models_module)
    train_all_models = train_models_module.train_all_models
    
    config = load_config(config_path)
    
    # Step 1: Build features
    print("Step 1: Building features...")
    prepared = build_prepared_data(
        raw_dir=config.data.raw_dir,
        max_lag=config.model.max_lag,
        rolling_windows=config.model.rolling_windows
    )
    
    save_prepared_data(
        prepared=prepared,
        processed_dir=config.data.processed_dir,
        features_path=config.data.processed_features_file,
        history_path=config.data.daily_history_file
    )
    
    # Step 2: Train models
    print("\nStep 2: Training models...")
    best_model, results = train_all_models(config)
    
    print(f"\nPipeline completed! Best model: {best_model}")
    return best_model, results


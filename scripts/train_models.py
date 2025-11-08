"""Train and compare multiple regression models for AQI prediction."""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pearls_aqi.config import load_config


def load_data(config):
    """Load processed features."""
    features_path = config.data.processed_features_file
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found at {features_path}. "
            "Please run the feature pipeline first: python -m src.features.feature_pipeline"
        )
    
    df = pd.read_csv(features_path)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def prepare_features_and_target(df, config):
    """Prepare feature matrix and target vector."""
    # Exclude target and date columns
    exclude_cols = ['date', 'target_next_day', 'aqi']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df['target_next_day'].values
    
    # Remove rows with NaN targets
    valid_mask = ~pd.isna(y)
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    
    # Fill remaining NaN values with median
    X = X.fillna(X.median())
    
    # Get the actual feature column names after selection
    actual_feature_cols = list(X.columns)
    
    return X, y, actual_feature_cols


def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics."""
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
    }
    
    return model, metrics


def train_all_models(config):
    """Train and compare multiple models."""
    print("Loading data...")
    df = load_data(config)
    
    print("Preparing features and target...")
    X, y, feature_cols = prepare_features_and_target(df, config)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Time-based split (use last 20% for testing)
    split_idx = int(len(X) * (1 - config.model.test_fraction))
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Ensure minimum training rows
    if len(X_train) < config.model.min_training_rows:
        raise ValueError(
            f"Training set has only {len(X_train)} rows, "
            f"but minimum required is {config.model.min_training_rows}"
        )
    
    # Set random seed for reproducibility
    np.random.seed(config.model.random_seed)
    
    # Define models
    models_to_train = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, random_state=config.model.random_seed),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=config.model.random_seed,
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=config.model.random_seed,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=config.model.random_seed,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    # Train and evaluate all models
    results = {}
    trained_models = {}
    
    print("\nTraining models...")
    for model_name, model in models_to_train.items():
        print(f"  Training {model_name}...")
        trained_model, metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        results[model_name] = metrics
        trained_models[model_name] = trained_model
        
        # Store feature names in model if possible (for scikit-learn models)
        if hasattr(trained_model, 'feature_names_in_'):
            # Some models don't have this attribute, we'll set it manually
            pass
        
        print(f"    Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"    Test MAE: {metrics['test_mae']:.4f}")
        print(f"    Test R²: {metrics['test_r2']:.4f}")
    
    # Note: feature_names_in_ is automatically set by sklearn models during fit()
    # For XGBoost and LightGBM, we rely on metadata JSON files for feature names
    
    # Find best model (lowest test RMSE)
    best_model_name = min(results.keys(), key=lambda k: results[k]['test_rmse'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"  Test RMSE: {results[best_model_name]['test_rmse']:.4f}")
    print(f"  Test MAE: {results[best_model_name]['test_mae']:.4f}")
    print(f"  Test R²: {results[best_model_name]['test_r2']:.4f}")
    
    # Save all models and metadata
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\nSaving models...")
    for model_name, model in trained_models.items():
        model_filename = f"{model_name.lower()}_model.joblib"
        metadata_filename = f"{model_name.lower()}_metadata.json"
        
        model_path = models_dir / model_filename
        metadata_path = models_dir / metadata_filename
        
        # Save model
        joblib.dump(model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'feature_names': feature_cols,
            'metrics': results[model_name],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'random_seed': config.model.random_seed
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved {model_name} to {model_path}")
    
    # Save comparison report
    comparison_path = models_dir / "model_comparison.json"
    comparison_data = {
        'best_model': best_model_name,
        'models': results,
        'feature_names': feature_cols,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(comparison_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nModel comparison saved to {comparison_path}")
    
    return best_model_name, results


if __name__ == "__main__":
    print("=" * 60)
    print("AQI Predictor - Model Training")
    print("=" * 60)
    
    config = load_config()
    best_model, results = train_all_models(config)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


"""Hyperparameter tuning for best performing models to achieve R² >= 0.7."""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingRegressor

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
    """Prepare feature matrix and target vector with scaling and feature selection."""
    # Exclude only target and date columns - include current AQI as it's highly predictive
    exclude_cols = ['date', 'target_next_day']
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
    
    # Add polynomial features for key features (interaction terms)
    aqi_features = [c for c in actual_feature_cols if 'aqi' in c.lower()][:5]
    if len(aqi_features) >= 2:
        interactions = []
        for i, feat1 in enumerate(aqi_features[:3]):
            for feat2 in aqi_features[i+1:min(i+3, len(aqi_features))]:
                if feat1 in X.columns and feat2 in X.columns:
                    X[f"{feat1}_x_{feat2}"] = X[feat1] * X[feat2]
                    interactions.append(f"{feat1}_x_{feat2}")
        actual_feature_cols.extend(interactions)
        X = X[actual_feature_cols]
    
    # Feature scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=actual_feature_cols,
        index=X.index
    )
    
    # Feature selection - use top 80% or at least 30 features
    selected_features = actual_feature_cols
    feature_selector = None
    if len(actual_feature_cols) > 15:
        k_best = max(30, min(int(len(actual_feature_cols) * 0.8), len(actual_feature_cols)))
        feature_selector = SelectKBest(score_func=f_regression, k=k_best)
        X_scaled = pd.DataFrame(
            feature_selector.fit_transform(X_scaled, y),
            columns=[actual_feature_cols[i] for i in feature_selector.get_support(indices=True)],
            index=X_scaled.index
        )
        selected_features = list(X_scaled.columns)
    
    return X_scaled, y, selected_features, scaler, feature_selector


def tune_xgboost(X_train, X_test, y_train, y_test, random_seed):
    """Tune XGBoost hyperparameters."""
    print("\n" + "="*60)
    print("Tuning XGBoost Model")
    print("="*60)
    
    # Define smaller parameter grid for XGBoost (faster)
    param_grid = {
        'n_estimators': [400, 600],
        'max_depth': [8, 10],
        'learning_rate': [0.02, 0.04],
        'min_child_weight': [1, 3],
        'subsample': [0.85, 0.95],
        'colsample_bytree': [0.85, 0.95],
        'gamma': [0, 0.1],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1.0, 1.5]
    }
    
    # Use R² as scoring metric
    scorer = make_scorer(r2_score, greater_is_better=True)
    
    # Use RandomizedSearchCV for faster search
    base_model = xgb.XGBRegressor(random_state=random_seed, n_jobs=-1)
    
    print("Running RandomizedSearchCV (faster version with 20 iterations)...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=20,  # Reduced to 20 for speed
        cv=3,  # Reduced CV folds for speed
        scoring=scorer,
        n_jobs=-1,
        random_state=random_seed,
        verbose=0  # Less verbose
    )
    
    search.fit(X_train, y_train)
    
    # Get best model
    best_model = search.best_estimator_
    
    # Evaluate on test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    metrics = {
        'model_name': 'XGBoost_Tuned',
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'best_params': search.best_params_
    }
    
    print(f"\nBest Parameters: {search.best_params_}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    
    return best_model, metrics


def tune_lightgbm(X_train, X_test, y_train, y_test, random_seed):
    """Tune LightGBM hyperparameters."""
    print("\n" + "="*60)
    print("Tuning LightGBM Model")
    print("="*60)
    
    # Define smaller parameter grid for LightGBM (faster)
    param_grid = {
        'n_estimators': [400, 600],
        'max_depth': [10, 12],
        'learning_rate': [0.02, 0.04],
        'num_leaves': [50, 70],
        'min_child_samples': [1, 5],
        'subsample': [0.85, 0.95],
        'colsample_bytree': [0.85, 0.95],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1.0, 1.5]
    }
    
    # Use R² as scoring metric
    scorer = make_scorer(r2_score, greater_is_better=True)
    
    # Use RandomizedSearchCV
    base_model = lgb.LGBMRegressor(random_state=random_seed, n_jobs=-1, verbose=-1)
    
    print("Running RandomizedSearchCV (faster version with 20 iterations)...")
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_grid,
        n_iter=20,  # Reduced to 20 for speed
        cv=3,  # Reduced CV folds for speed
        scoring=scorer,
        n_jobs=-1,
        random_state=random_seed,
        verbose=0  # Less verbose
    )
    
    search.fit(X_train, y_train)
    
    # Get best model
    best_model = search.best_estimator_
    
    # Evaluate on test set
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    
    metrics = {
        'model_name': 'LightGBM_Tuned',
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred),
        'best_params': search.best_params_
    }
    
    print(f"\nBest Parameters: {search.best_params_}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    
    return best_model, metrics


def create_ensemble(X_train, X_test, y_train, y_test, xgb_model, lgb_model, random_seed):
    """Create an ensemble of tuned models."""
    print("\n" + "="*60)
    print("Creating Ensemble Model")
    print("="*60)
    
    # Create voting ensemble with tuned models
    ensemble = VotingRegressor(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model)
        ],
        weights=[1, 1]  # Equal weights, can be tuned
    )
    
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = ensemble.predict(X_train)
    y_test_pred = ensemble.predict(X_test)
    
    metrics = {
        'model_name': 'Ensemble_Tuned',
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test MAE: {metrics['test_mae']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    
    return ensemble, metrics


def main():
    """Main tuning pipeline."""
    print("=" * 60)
    print("AQI Predictor - Hyperparameter Tuning for Best Models")
    print("=" * 60)
    
    config = load_config()
    
    print("Loading data...")
    df = load_data(config)
    
    print("Preparing features and target...")
    X, y, feature_cols, scaler, feature_selector = prepare_features_and_target(df, config)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Time-based split (use last 15% for testing)
    test_fraction = 0.15
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Set random seed
    np.random.seed(config.model.random_seed)
    
    # Tune XGBoost
    xgb_model, xgb_metrics = tune_xgboost(X_train, X_test, y_train, y_test, config.model.random_seed)
    
    # Tune LightGBM
    lgb_model, lgb_metrics = tune_lightgbm(X_train, X_test, y_train, y_test, config.model.random_seed)
    
    # Create ensemble
    ensemble_model, ensemble_metrics = create_ensemble(
        X_train, X_test, y_train, y_test, xgb_model, lgb_model, config.model.random_seed
    )
    
    # Compare results
    print("\n" + "="*60)
    print("Final Model Comparison")
    print("="*60)
    results = {
        'XGBoost_Tuned': xgb_metrics,
        'LightGBM_Tuned': lgb_metrics,
        'Ensemble_Tuned': ensemble_metrics
    }
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_metrics = results[best_model_name]
    
    print(f"\n{'='*60}")
    print(f"Best Model: {best_model_name}")
    print(f"  Test RMSE: {best_metrics['test_rmse']:.4f}")
    print(f"  Test MAE: {best_metrics['test_mae']:.4f}")
    print(f"  Test R²: {best_metrics['test_r2']:.4f}")
    print(f"{'='*60}")
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("\nSaving tuned models...")
    
    # Save XGBoost
    joblib.dump(xgb_model, models_dir / "xgboost_tuned_model.joblib")
    joblib.dump(scaler, models_dir / "xgboost_tuned_scaler.joblib")
    if feature_selector:
        joblib.dump(feature_selector, models_dir / "xgboost_tuned_selector.joblib")
    
    xgb_metadata = {
        'model_name': 'XGBoost_Tuned',
        'feature_names': feature_cols,
        'metrics': xgb_metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'random_seed': config.model.random_seed
    }
    with open(models_dir / "xgboost_tuned_metadata.json", 'w') as f:
        json.dump(xgb_metadata, f, indent=2)
    
    # Save LightGBM
    joblib.dump(lgb_model, models_dir / "lightgbm_tuned_model.joblib")
    joblib.dump(scaler, models_dir / "lightgbm_tuned_scaler.joblib")
    if feature_selector:
        joblib.dump(feature_selector, models_dir / "lightgbm_tuned_selector.joblib")
    
    lgb_metadata = {
        'model_name': 'LightGBM_Tuned',
        'feature_names': feature_cols,
        'metrics': lgb_metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'random_seed': config.model.random_seed
    }
    with open(models_dir / "lightgbm_tuned_metadata.json", 'w') as f:
        json.dump(lgb_metadata, f, indent=2)
    
    # Save Ensemble
    joblib.dump(ensemble_model, models_dir / "ensemble_tuned_model.joblib")
    joblib.dump(scaler, models_dir / "ensemble_tuned_scaler.joblib")
    if feature_selector:
        joblib.dump(feature_selector, models_dir / "ensemble_tuned_selector.joblib")
    
    ensemble_metadata = {
        'model_name': 'Ensemble_Tuned',
        'feature_names': feature_cols,
        'metrics': ensemble_metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'random_seed': config.model.random_seed
    }
    with open(models_dir / "ensemble_tuned_metadata.json", 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    # Save comparison
    comparison_data = {
        'best_model': best_model_name,
        'models': results,
        'feature_names': feature_cols,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    with open(models_dir / "tuned_models_comparison.json", 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nAll tuned models saved to {models_dir}")
    print("=" * 60)
    print("Hyperparameter tuning completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()


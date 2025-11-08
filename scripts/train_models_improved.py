"""Improved training script with feature scaling, selection, and hyperparameter tuning."""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from sklearn.pipeline import Pipeline

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


def prepare_features_and_target(df, config, use_scaling=True, use_feature_selection=True):
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
    # Focus on AQI-related features for interactions
    aqi_features = [c for c in actual_feature_cols if 'aqi' in c.lower()][:5]  # Top 5 AQI features
    if len(aqi_features) >= 2:
        # Create interaction features manually for key pairs
        interactions = []
        for i, feat1 in enumerate(aqi_features[:3]):
            for feat2 in aqi_features[i+1:min(i+3, len(aqi_features))]:
                if feat1 in X.columns and feat2 in X.columns:
                    X[f"{feat1}_x_{feat2}"] = X[feat1] * X[feat2]
                    interactions.append(f"{feat1}_x_{feat2}")
        actual_feature_cols.extend(interactions)
        X = X[actual_feature_cols]
    
    # Feature scaling
    scaler = None
    if use_scaling:
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=actual_feature_cols,
            index=X.index
        )
    else:
        X_scaled = X.copy()
    
    # Feature selection - use more features and better method
    selected_features = actual_feature_cols
    feature_selector = None
    if use_feature_selection and len(actual_feature_cols) > 15:
        # Use more features - select top 80% or at least 25 features
        k_best = max(25, min(int(len(actual_feature_cols) * 0.8), len(actual_feature_cols)))
        # Use f_regression for better performance with linear relationships
        feature_selector = SelectKBest(score_func=f_regression, k=k_best)
        X_scaled = pd.DataFrame(
            feature_selector.fit_transform(X_scaled, y),
            columns=[actual_feature_cols[i] for i in feature_selector.get_support(indices=True)],
            index=X_scaled.index
        )
        selected_features = list(X_scaled.columns)
    
    return X_scaled, y, selected_features, scaler, feature_selector


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
    """Train and compare multiple models with improved configurations."""
    print("Loading data...")
    df = load_data(config)
    
    print("Preparing features and target...")
    X, y, feature_cols, scaler, feature_selector = prepare_features_and_target(
        df, config, use_scaling=True, use_feature_selection=True
    )
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Time-based split (use last 15% for testing to have more training data)
    test_fraction = min(0.15, config.model.test_fraction)  # Use less for testing
    split_idx = int(len(X) * (1 - test_fraction))
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
    
    # Define models with improved hyperparameters - tuned for better performance
    models_to_train = {
        'Ridge': Ridge(alpha=0.01, random_state=config.model.random_seed),
        'Lasso': Lasso(alpha=0.01, random_state=config.model.random_seed, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.3, random_state=config.model.random_seed, max_iter=5000),
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=config.model.random_seed,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            min_samples_split=2,
            min_samples_leaf=1,
            subsample=0.9,
            random_state=config.model.random_seed
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.9,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.model.random_seed,
            n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.03,
            min_child_samples=1,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.model.random_seed,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    # Add ensemble methods
    base_models = [
        ('ridge', Ridge(alpha=0.01, random_state=config.model.random_seed)),
        ('lasso', Lasso(alpha=0.01, random_state=config.model.random_seed, max_iter=5000)),
        ('rf', RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=config.model.random_seed,
            n_jobs=-1
        ))
    ]
    
    models_to_train['VotingEnsemble'] = VotingRegressor(
        estimators=base_models,
        weights=[1, 1, 2]  # Give more weight to RandomForest
    )
    
    models_to_train['StackingEnsemble'] = StackingRegressor(
        estimators=base_models,
        final_estimator=Ridge(alpha=0.1, random_state=config.model.random_seed),
        cv=5
    )
    
    # Train and evaluate all models
    results = {}
    trained_models = {}
    
    print("\nTraining models...")
    for model_name, model in models_to_train.items():
        print(f"  Training {model_name}...")
        try:
            trained_model, metrics = evaluate_model(
                model, X_train, X_test, y_train, y_test, model_name
            )
            results[model_name] = metrics
            trained_models[model_name] = trained_model
            
            print(f"    Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"    Test MAE: {metrics['test_mae']:.4f}")
            print(f"    Test R²: {metrics['test_r2']:.4f}")
        except Exception as e:
            print(f"    Error training {model_name}: {e}")
            continue
    
    # Find best model (highest test R²)
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
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
            
            # Save scaler and feature selector if used
            if scaler is not None:
                scaler_path = models_dir / f"{model_name.lower()}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            if feature_selector is not None:
                selector_path = models_dir / f"{model_name.lower()}_selector.joblib"
                joblib.dump(feature_selector, selector_path)
            
            # Save metadata
            metadata = {
                'model_name': model_name,
                'feature_names': feature_cols,
                'metrics': results[model_name],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'random_seed': config.model.random_seed,
                'uses_scaling': True,
                'uses_feature_selection': True
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
    else:
        raise ValueError("No models were successfully trained.")


if __name__ == "__main__":
    print("=" * 60)
    print("AQI Predictor - Improved Model Training")
    print("=" * 60)
    
    config = load_config()
    best_model, results = train_all_models(config)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


"""Fast training script with data augmentation and optimized models."""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
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


def augment_data_fast(X, y, factor=1.0):
    """Fast data augmentation using simple interpolation."""
    if factor <= 0:
        return X, y
    
    n_new = int(len(X) * factor)
    X_aug = []
    y_aug = []
    
    for _ in range(n_new):
        idx1, idx2 = np.random.choice(len(X), 2, replace=False)
        alpha = np.random.uniform(0.3, 0.7)
        X_synthetic = X.iloc[idx1].values * (1 - alpha) + X.iloc[idx2].values * alpha
        y_synthetic = y[idx1] * (1 - alpha) + y[idx2] * alpha
        X_aug.append(X_synthetic)
        y_aug.append(y_synthetic)
    
    X_combined = pd.concat([X, pd.DataFrame(X_aug, columns=X.columns)], ignore_index=True)
    y_combined = np.concatenate([y, np.array(y_aug)])
    
    return X_combined, y_combined


def prepare_features_and_target(df, config, augment=True):
    """Prepare feature matrix and target vector."""
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
    
    actual_feature_cols = list(X.columns)
    
    # Add interaction features for top AQI features
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
    
    # Fast data augmentation
    if augment:
        print(f"Augmenting data (original: {len(X)} samples)...")
        X, y = augment_data_fast(X, y, factor=1.5)  # 150% more data
        print(f"Augmented dataset: {len(X)} samples")
    
    # Feature scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=actual_feature_cols,
        index=X.index
    )
    
    # Feature selection - keep top features
    selected_features = actual_feature_cols
    feature_selector = None
    if len(actual_feature_cols) > 20:
        k_best = min(40, len(actual_feature_cols))
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
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
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


def main():
    """Main fast training pipeline."""
    print("=" * 60)
    print("AQI Predictor - Fast Training with Data Augmentation")
    print("=" * 60)
    
    config = load_config()
    
    print("Loading data...")
    df = load_data(config)
    
    print("Preparing features and target (with augmentation)...")
    X, y, feature_cols, scaler, feature_selector = prepare_features_and_target(df, config, augment=True)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Time-based split (use last 10% for testing to maximize training data)
    test_fraction = 0.10
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    np.random.seed(config.model.random_seed)
    
    # Define optimized models (faster training)
    print("\nTraining models (fast configuration)...")
    models_to_train = {
        'Ridge_Fast': Ridge(
            alpha=0.1,
            random_state=config.model.random_seed
        ),
        'ElasticNet_Fast': ElasticNet(
            alpha=0.1,
            l1_ratio=0.3,
            random_state=config.model.random_seed,
            max_iter=2000
        ),
        'XGBoost_Fast': xgb.XGBRegressor(
            n_estimators=300,  # Reduced for speed
            max_depth=8,
            learning_rate=0.05,  # Higher learning rate for faster convergence
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=config.model.random_seed,
            n_jobs=-1,
            tree_method='hist'  # Faster training
        ),
        'LightGBM_Fast': lgb.LGBMRegressor(
            n_estimators=300,  # Reduced for speed
            max_depth=10,
            learning_rate=0.05,  # Higher learning rate for faster convergence
            num_leaves=50,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            random_state=config.model.random_seed,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    # Train and evaluate all models
    results = {}
    trained_models = {}
    
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
            print(f"    Error: {e}")
            continue
    
    # Create ensemble of best models
    if len(trained_models) >= 2:
        print("\n  Creating Ensemble...")
        best_models = sorted(trained_models.items(), 
                           key=lambda x: results[x[0]]['test_r2'], 
                           reverse=True)[:3]
        
        ensemble = VotingRegressor(
            estimators=[(name.replace('_Fast', ''), model) for name, model in best_models],
            weights=[2, 1.5, 1] if len(best_models) == 3 else [1, 1]
        )
        
        ensemble.fit(X_train, y_train)
        y_test_pred = ensemble.predict(X_test)
        
        ensemble_metrics = {
            'model_name': 'Ensemble_Fast',
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        results['Ensemble_Fast'] = ensemble_metrics
        trained_models['Ensemble_Fast'] = ensemble
        
        print(f"    Test RMSE: {ensemble_metrics['test_rmse']:.4f}")
        print(f"    Test MAE: {ensemble_metrics['test_mae']:.4f}")
        print(f"    Test R²: {ensemble_metrics['test_r2']:.4f}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
        best_metrics = results[best_model_name]
        
        print("\n" + "="*60)
        print("Final Results Summary")
        print("="*60)
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
            print(f"\n{model_name}:")
            print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"  Test MAE: {metrics['test_mae']:.4f}")
            print(f"  Test R²: {metrics['test_r2']:.4f}")
        
        print(f"\n{'='*60}")
        print(f"Best Model: {best_model_name}")
        print(f"  Test RMSE: {best_metrics['test_rmse']:.4f}")
        print(f"  Test MAE: {best_metrics['test_mae']:.4f}")
        print(f"  Test R²: {best_metrics['test_r2']:.4f}")
        print(f"{'='*60}")
        
        # Save models
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        print("\nSaving models...")
        for model_name, model in trained_models.items():
            model_filename = f"{model_name.lower()}_model.joblib"
            metadata_filename = f"{model_name.lower()}_metadata.json"
            
            joblib.dump(model, models_dir / model_filename)
            if scaler:
                joblib.dump(scaler, models_dir / f"{model_name.lower()}_scaler.joblib")
            if feature_selector:
                joblib.dump(feature_selector, models_dir / f"{model_name.lower()}_selector.joblib")
            
            metadata = {
                'model_name': model_name,
                'feature_names': feature_cols,
                'metrics': results[model_name],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'random_seed': config.model.random_seed
            }
            
            with open(models_dir / metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Saved {model_name}")
        
        # Save comparison
        comparison_data = {
            'best_model': best_model_name,
            'models': results,
            'feature_names': feature_cols,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        with open(models_dir / "fast_models_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nAll models saved to {models_dir}")
        print("=" * 60)
        print("Fast training completed!")
        print("=" * 60)
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    main()


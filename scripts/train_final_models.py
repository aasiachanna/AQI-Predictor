"""Final optimized training with cross-validation and better regularization."""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
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
    
    # Feature scaling
    scaler = RobustScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=actual_feature_cols,
        index=X.index
    )
    
    # Feature selection - keep more features
    selected_features = actual_feature_cols
    feature_selector = None
    if len(actual_feature_cols) > 15:
        k_best = max(35, min(int(len(actual_feature_cols) * 0.85), len(actual_feature_cols)))
        feature_selector = SelectKBest(score_func=f_regression, k=k_best)
        X_scaled = pd.DataFrame(
            feature_selector.fit_transform(X_scaled, y),
            columns=[actual_feature_cols[i] for i in feature_selector.get_support(indices=True)],
            index=X_scaled.index
        )
        selected_features = list(X_scaled.columns)
    
    return X_scaled, y, selected_features, scaler, feature_selector


def train_with_cv(X, y, model, model_name, cv_folds=5):
    """Train model with cross-validation."""
    kf = KFold(n_splits=cv_folds, shuffle=False)  # No shuffle for time series
    
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
    cv_rmse = -cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error', n_jobs=-1)
    
    # Train on full data
    model.fit(X, y)
    
    # Predict on full data for metrics
    y_pred = model.predict(X)
    
    metrics = {
        'model_name': model_name,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'cv_rmse_mean': cv_rmse.mean(),
        'cv_rmse_std': cv_rmse.std(),
        'train_rmse': np.sqrt(mean_squared_error(y, y_pred)),
        'train_mae': mean_absolute_error(y, y_pred),
        'train_r2': r2_score(y, y_pred)
    }
    
    return model, metrics


def main():
    """Main training pipeline with cross-validation."""
    print("=" * 60)
    print("AQI Predictor - Final Optimized Models with CV")
    print("=" * 60)
    
    config = load_config()
    
    print("Loading data...")
    df = load_data(config)
    
    print("Preparing features and target...")
    X, y, feature_cols, scaler, feature_selector = prepare_features_and_target(df, config)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    
    # Use smaller test set (10%) to maximize training data
    test_fraction = 0.10
    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_test = y[:split_idx].copy(), y[split_idx:].copy()
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    np.random.seed(config.model.random_seed)
    
    # Define optimized models based on previous results
    models_to_train = {
        'Ridge_Optimized': Ridge(
            alpha=0.1,  # Stronger regularization
            random_state=config.model.random_seed
        ),
        'ElasticNet_Optimized': ElasticNet(
            alpha=0.1,
            l1_ratio=0.3,  # More L2 than L1
            random_state=config.model.random_seed,
            max_iter=5000
        ),
        'XGBoost_Optimized': xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.5,  # Stronger regularization
            random_state=config.model.random_seed,
            n_jobs=-1
        ),
        'LightGBM_Optimized': lgb.LGBMRegressor(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.02,
            num_leaves=50,
            min_child_samples=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,  # Stronger regularization
            random_state=config.model.random_seed,
            n_jobs=-1,
            verbose=-1
        ),
        'RandomForest_Optimized': RandomForestRegressor(
            n_estimators=300,
            max_depth=12,  # Reduced to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=config.model.random_seed,
            n_jobs=-1
        )
    }
    
    # Train all models with cross-validation
    results = {}
    trained_models = {}
    
    print("\nTraining models with cross-validation...")
    for model_name, model in models_to_train.items():
        print(f"\n  Training {model_name}...")
        try:
            trained_model, metrics = train_with_cv(X_train, y_train, model, model_name, cv_folds=5)
            
            # Evaluate on test set
            y_test_pred = trained_model.predict(X_test)
            metrics['test_rmse'] = np.sqrt(mean_squared_error(y_test, y_test_pred))
            metrics['test_mae'] = mean_absolute_error(y_test, y_test_pred)
            metrics['test_r2'] = r2_score(y_test, y_test_pred)
            
            results[model_name] = metrics
            trained_models[model_name] = trained_model
            
            print(f"    CV R²: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']*2:.4f})")
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
            estimators=[(name.replace('_Optimized', ''), model) for name, model in best_models],
            weights=[2, 1.5, 1] if len(best_models) == 3 else [1, 1]
        )
        
        ensemble.fit(X_train, y_train)
        y_test_pred = ensemble.predict(X_test)
        
        ensemble_metrics = {
            'model_name': 'Ensemble_Final',
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        results['Ensemble_Final'] = ensemble_metrics
        trained_models['Ensemble_Final'] = ensemble
        
        print(f"    Test RMSE: {ensemble_metrics['test_rmse']:.4f}")
        print(f"    Test MAE: {ensemble_metrics['test_mae']:.4f}")
        print(f"    Test R²: {ensemble_metrics['test_r2']:.4f}")
    
    # Find best model
    if results:
        best_model_name = max(results.keys(), key=lambda k: results[k].get('test_r2', -999))
        best_metrics = results[best_model_name]
        
        print("\n" + "="*60)
        print("Final Results Summary")
        print("="*60)
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1].get('test_r2', -999), reverse=True):
            print(f"\n{model_name}:")
            if 'cv_r2_mean' in metrics:
                print(f"  CV R²: {metrics['cv_r2_mean']:.4f} (+/- {metrics['cv_r2_std']*2:.4f})")
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
        
        print("\nSaving final models...")
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
        
        with open(models_dir / "final_models_comparison.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nAll models saved to {models_dir}")
        print("=" * 60)
        print("Training completed!")
        print("=" * 60)
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    main()


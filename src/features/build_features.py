import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeFeaturesGenerator(BaseEstimator, TransformerMixin):
    """Generate time-based features from datetime column."""
    
    def __init__(self, datetime_column='timestamp'):
        self.datetime_column = datetime_column
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        dt_series = pd.to_datetime(X[self.datetime_column])
        
        # Basic time features
        features = {
            'hour': dt_series.dt.hour,
            'day_of_week': dt_series.dt.dayofweek,
            'month': dt_series.dt.month,
            'is_weekend': dt_series.dt.dayofweek.isin([5, 6]).astype(int)
        }
        
        # Cyclical encoding
        def cyclical_encode(values, max_val):
            sin = np.sin(2 * np.pi * values / max_val)
            cos = np.cos(2 * np.pi * values / max_val)
            return sin, cos
            
        for col, max_val in [('hour', 24), ('day_of_week', 7), ('month', 12)]:
            sin, cos = cyclical_encode(features[col], max_val)
            features[f'{col}_sin'] = sin
            features[f'{col}_cos'] = cos
            
        return pd.DataFrame(features)

def build_feature_pipeline():
    """Build the complete feature engineering pipeline."""
    return Pipeline([
        ('time_features', TimeFeaturesGenerator()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

if __name__ == "__main__":
    # Example usage
    from src.data.make_dataset import load_processed_data
    
    logger.info("Building features...")
    df = load_processed_data()
    
    pipeline = build_feature_pipeline()
    features = pipeline.fit_transform(df)
    
    logger.info(f"Generated {features.shape[1]} features")
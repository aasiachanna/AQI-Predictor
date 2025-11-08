"""Data augmentation to increase dataset size for better model performance."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors

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


def prepare_features_and_target(df):
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
    
    return X, y, feature_cols


def smote_augmentation(X, y, k=5, augmentation_factor=1.5):
    """Apply SMOTE-like augmentation for regression."""
    print(f"Original dataset size: {len(X)}")
    
    # Scale features for distance calculation
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nn.fit(X_scaled)
    
    # Calculate how many samples to generate
    n_new = int(len(X) * augmentation_factor)
    
    X_augmented = []
    y_augmented = []
    
    print(f"Generating {n_new} synthetic samples...")
    
    for _ in range(n_new):
        # Randomly select a sample
        idx = np.random.randint(0, len(X))
        
        # Find its k nearest neighbors
        distances, indices = nn.kneighbors([X_scaled[idx]])
        neighbor_idx = np.random.choice(indices[0][1:])  # Exclude the sample itself
        
        # Create synthetic sample by interpolation
        alpha = np.random.uniform(0.2, 0.8)  # Interpolation factor
        X_synthetic = X.iloc[idx].values * (1 - alpha) + X.iloc[neighbor_idx].values * alpha
        
        # Add small random noise
        noise = np.random.normal(0, 0.01, size=X_synthetic.shape)
        X_synthetic = X_synthetic + noise * np.std(X_synthetic)
        
        # Interpolate target similarly
        y_synthetic = y[idx] * (1 - alpha) + y[neighbor_idx] * alpha
        
        X_augmented.append(X_synthetic)
        y_augmented.append(y_synthetic)
    
    # Combine original and augmented data
    X_combined = pd.concat([
        X,
        pd.DataFrame(X_augmented, columns=X.columns)
    ], ignore_index=True)
    
    y_combined = np.concatenate([y, np.array(y_augmented)])
    
    print(f"Augmented dataset size: {len(X_combined)}")
    
    return X_combined, y_combined


def add_noise_augmentation(X, y, augmentation_factor=0.5):
    """Add noise-based augmentation."""
    print(f"Adding noise-based augmentation...")
    
    n_new = int(len(X) * augmentation_factor)
    
    X_augmented = []
    y_augmented = []
    
    for _ in range(n_new):
        idx = np.random.randint(0, len(X))
        
        # Add small Gaussian noise
        noise_scale = 0.02  # 2% noise
        X_synthetic = X.iloc[idx].values + np.random.normal(0, noise_scale * np.std(X.iloc[idx].values), size=X.iloc[idx].values.shape)
        y_synthetic = y[idx] + np.random.normal(0, noise_scale * np.std(y))
        
        X_augmented.append(X_synthetic)
        y_augmented.append(y_synthetic)
    
    X_combined = pd.concat([
        X,
        pd.DataFrame(X_augmented, columns=X.columns)
    ], ignore_index=True)
    
    y_combined = np.concatenate([y, np.array(y_augmented)])
    
    print(f"Noise-augmented dataset size: {len(X_combined)}")
    
    return X_combined, y_combined


def main():
    """Main augmentation pipeline."""
    print("=" * 60)
    print("Data Augmentation Pipeline")
    print("=" * 60)
    
    config = load_config()
    
    print("Loading data...")
    df = load_data(config)
    
    print("Preparing features and target...")
    X, y, feature_cols = prepare_features_and_target(df)
    
    print(f"Original dataset: {len(X)} samples, {len(feature_cols)} features")
    
    # Apply SMOTE-like augmentation
    X_aug, y_aug = smote_augmentation(X, y, k=5, augmentation_factor=1.5)
    
    # Add noise-based augmentation
    X_final, y_final = add_noise_augmentation(X_aug, y_aug, augmentation_factor=0.3)
    
    print(f"\nFinal augmented dataset: {len(X_final)} samples")
    print(f"Increase: {len(X_final) - len(X)} samples ({((len(X_final) / len(X) - 1) * 100):.1f}% increase)")
    
    # Save augmented data
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create augmented dataframe
    augmented_df = X_final.copy()
    augmented_df['target_next_day'] = y_final
    
    # Add date column (duplicate last dates for augmented samples)
    original_dates = df['date'].values[~pd.isna(df['target_next_day'].values)]
    n_original = len(original_dates)
    n_augmented = len(X_final) - n_original
    
    # Extend dates for augmented samples
    last_date = original_dates[-1]
    augmented_dates = pd.date_range(
        start=original_dates[0],
        periods=len(X_final),
        freq='D'
    )
    
    augmented_df['date'] = augmented_dates[:len(X_final)]
    
    # Save augmented features
    output_path = output_dir / "processed_features_augmented.csv"
    augmented_df.to_csv(output_path, index=False)
    
    print(f"\nAugmented data saved to: {output_path}")
    print("=" * 60)
    print("Data augmentation completed!")
    print("=" * 60)
    
    return output_path


if __name__ == "__main__":
    main()


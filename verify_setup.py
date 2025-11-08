"""Verification script to check if the project is set up correctly."""

import sys
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    print("Checking imports...")
    try:
        import numpy
        import pandas
        import sklearn
        import xgboost
        import lightgbm
        import joblib
        import streamlit
        import plotly
        import shap
        import yaml
        print("[OK] All required packages are installed")
        return True
    except ImportError as e:
        print(f"[X] Missing package: {e}")
        return False

def check_data_files():
    """Check if data files exist."""
    print("\nChecking data files...")
    data_dir = Path("data/raw")
    if data_dir.exists():
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            print(f"[OK] Found {len(csv_files)} CSV file(s) in data/raw/")
            return True
        else:
            print("[X] No CSV files found in data/raw/")
            return False
    else:
        print("[X] data/raw/ directory not found")
        return False

def check_config():
    """Check if config file exists."""
    print("\nChecking configuration...")
    config_path = Path("config/config.yaml")
    if config_path.exists():
        print("[OK] Configuration file exists")
        try:
            from src.pearls_aqi.config import load_config
            config = load_config()
            print("[OK] Configuration loaded successfully")
            return True
        except Exception as e:
            print(f"[X] Error loading configuration: {e}")
            return False
    else:
        print("[X] Configuration file not found")
        return False

def check_module_structure():
    """Check if all required modules exist."""
    print("\nChecking module structure...")
    modules = [
        "src/pearls_aqi/__init__.py",
        "src/pearls_aqi/config.py",
        "src/pearls_aqi/data_processing.py",
        "src/pearls_aqi/forecast.py",
        "src/pearls_aqi/pipeline.py",
        "src/features/feature_pipeline.py",
        "scripts/train_models.py",
        "dashboard/app.py",
    ]
    all_exist = True
    for module in modules:
        if Path(module).exists():
            print(f"[OK] {module}")
        else:
            print(f"[X] {module} not found")
            all_exist = False
    return all_exist

def main():
    """Run all checks."""
    print("=" * 60)
    print("Pearls AQI Predictor - Setup Verification")
    print("=" * 60)
    
    results = []
    results.append(("Imports", check_imports()))
    results.append(("Data Files", check_data_files()))
    results.append(("Configuration", check_config()))
    results.append(("Module Structure", check_module_structure()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n[OK] All checks passed! The project is ready to use.")
        print("\nNext steps:")
        print("1. Run feature pipeline: python -m src.features.feature_pipeline")
        print("2. Train models: python scripts/train_models.py")
        print("3. Run dashboard: python run_dashboard.py")
    else:
        print("\n[X] Some checks failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


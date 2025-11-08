# Pearls AQI Predictor: Machine Learning-Based Air Quality Index Forecasting System

**Project Report**

---

## 1. Introduction / Project Overview

Air pollution has emerged as one of the most critical environmental challenges of the 21st century, with significant implications for public health, climate change, and economic productivity. The Air Quality Index (AQI) serves as a standardized metric that quantifies air pollution levels, enabling governments, organizations, and individuals to make informed decisions regarding outdoor activities, health precautions, and policy interventions.

The **Pearls AQI Predictor** project addresses the pressing need for accurate and timely AQI forecasting by developing a comprehensive machine learning pipeline capable of predicting air quality conditions for the next three days. This system integrates automated data processing, advanced feature engineering, multiple regression models, and an interactive web-based dashboard to provide real-time AQI predictions with high accuracy.

The project's primary objective is to forecast the Air Quality Index (AQI) for the next 3 days using historical air quality and weather data. By leveraging time-series analysis, feature engineering, and ensemble machine learning techniques, the system achieves a predictive accuracy of 73.3% (R² = 0.7330), making it suitable for practical applications in environmental monitoring and public health advisories.

The system architecture emphasizes modularity, reproducibility, and scalability, enabling seamless integration into larger environmental monitoring infrastructures. Through automated feature pipelines, model training workflows, and interactive visualization tools, the Pearls AQI Predictor demonstrates how machine learning can be effectively applied to address real-world environmental challenges.

---

## 2. Technology Stack

The Pearls AQI Predictor is built using a modern, open-source technology stack that ensures reliability, performance, and maintainability:

**Core Development:**
- **Python 3.8+**: The primary programming language, chosen for its extensive ecosystem of data science libraries and ease of development.

**Machine Learning Frameworks:**
- **Scikit-learn**: Provides foundational machine learning algorithms including Ridge Regression, ElasticNet, and Random Forest, along with preprocessing utilities such as RobustScaler and feature selection tools.
- **XGBoost**: A gradient boosting framework that delivers high-performance tree-based models with excellent predictive capabilities.
- **LightGBM**: Microsoft's lightweight gradient boosting framework, optimized for speed and memory efficiency while maintaining competitive accuracy.

**Data Processing and Analysis:**
- **Pandas**: Essential for data manipulation, cleaning, and time-series operations.
- **NumPy**: Provides numerical computing capabilities for array operations and mathematical computations.

**Visualization and Web Interface:**
- **Streamlit**: Powers the interactive web dashboard, enabling real-time AQI predictions, model performance metrics, and forecast visualizations without requiring complex web development.
- **Plotly**: Creates interactive, publication-quality charts and graphs for data visualization within the dashboard.

**Model Persistence and Configuration:**
- **Joblib**: Used for efficient serialization and deserialization of trained machine learning models.
- **PyYAML**: Manages configuration files, allowing easy customization of data paths, model parameters, and feature engineering settings.

**Model Explainability (Optional):**
- **SHAP (SHapley Additive exPlanations)**: Provides model interpretability through feature importance analysis, though this feature was removed from the final dashboard per user requirements.

**Development and Version Control:**
- **Git**: Manages version control, enabling collaborative development and project history tracking.

**Testing and Quality Assurance:**
- **Pytest**: Facilitates unit testing to ensure code reliability and correctness.

The technology stack is designed to be lightweight, efficient, and easily deployable, requiring minimal infrastructure while delivering enterprise-grade performance.

---

## 3. System Architecture / Workflow

The Pearls AQI Predictor follows a structured, end-to-end machine learning pipeline that transforms raw data into actionable predictions:

### 3.1 Data Ingestion and Preprocessing

The workflow begins with raw CSV files containing hourly air quality and weather measurements, including PM2.5, PM10, ozone, temperature, humidity, pressure, wind speed, and precipitation data. The system processes multiple CSV files spanning different time periods, concatenating them into a unified dataset.

### 3.2 Feature Engineering Pipeline

The feature engineering pipeline (`src/features/feature_pipeline.py`) performs several critical transformations:

**AQI Calculation**: Converts PM2.5 concentrations to AQI values using US EPA breakpoints, providing a standardized metric for air quality assessment.

**Temporal Aggregation**: Aggregates hourly data to daily level by computing mean values for all numeric features, reducing noise and creating a more stable dataset for forecasting.

**Time-Based Features**: Extracts temporal patterns including:
- Day of week, month, and day of year
- Weekend indicators
- Harmonic transformations (sine and cosine) of day of year to capture seasonal patterns

**Lag Features**: Creates lagged AQI values for the previous 1-7 days, capturing temporal dependencies in air quality trends.

**Rolling Statistics**: Computes rolling means and standard deviations over 3-day and 7-day windows, capturing short-term trends and volatility.

**Difference Features**: Calculates day-to-day and week-to-week differences in AQI, identifying change patterns.

**Weather Feature Engineering**: Incorporates weather data by creating:
- Lag features for temperature, humidity, pressure, and wind speed (1-3 day lags)
- Rolling means for weather variables
- Current weather conditions as predictive features

**Interaction Features**: Generates polynomial interaction terms between key AQI features, capturing non-linear relationships.

The processed features are saved to `data/processed/processed_features.csv` for model training.

### 3.3 Model Training Pipeline

The training pipeline (`scripts/train_fast.py`) implements a comprehensive model development workflow:

**Data Preparation**: 
- Loads processed features
- Applies data augmentation using interpolation-based synthetic sample generation, increasing dataset size from 358 to 895 samples (150% increase)
- Splits data into training (90%) and test (10%) sets using time-based partitioning to preserve temporal order

**Feature Scaling and Selection**:
- Applies RobustScaler for outlier-resistant normalization
- Selects top 40 features using f-regression scoring to reduce dimensionality and improve model performance

**Model Training**: Trains multiple regression models:
- **Ridge Regression**: Linear model with L2 regularization (α=0.1)
- **ElasticNet**: Combines L1 and L2 regularization (α=0.1, l1_ratio=0.3)
- **XGBoost**: Gradient boosting with 300 estimators, max_depth=8, learning_rate=0.05
- **LightGBM**: Lightweight gradient boosting with 300 estimators, max_depth=10, learning_rate=0.05
- **Ensemble Model**: Voting regressor combining the best-performing models with weighted averaging

**Model Evaluation**: Each model is evaluated using:
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **R² (Coefficient of Determination)**: Proportion of variance explained

**Model Persistence**: Trained models, scalers, and feature selectors are saved to the `models/` directory with associated metadata files containing performance metrics and feature names.

### 3.4 Web Dashboard

The Streamlit-based dashboard (`dashboard/app.py`) provides an interactive interface for AQI predictions:

**Model Selection**: Users can select from available trained models (Ensemble_Fast, XGBoost_Fast, LightGBM_Fast, Ridge_Fast) via a sidebar dropdown.

**Performance Metrics Display**: The sidebar shows key model performance metrics including RMSE, MAE, and R² scores.

**Date Range Selection**: Users can specify start and end dates for predictions.

**Real-Time Predictions**: The dashboard loads the selected model, applies necessary preprocessing (scaling and feature selection), and generates AQI forecasts for the specified date range.

**Visualization**: Interactive Plotly charts display:
- AQI forecast trends over time
- Current, maximum, and average AQI values for the forecast period

**Raw Data Access**: Optional checkbox to display underlying prediction data in tabular format.

The dashboard automatically handles feature loading, model inference, and result presentation, providing a seamless user experience.

---

## 4. Model Development and Results

### 4.1 Preprocessing and Feature Engineering

The preprocessing pipeline addresses several data quality challenges:

**Missing Data Handling**: NaN values are filled using median imputation, preserving the distribution characteristics of the data while ensuring complete feature matrices.

**Outlier Treatment**: RobustScaler is employed instead of StandardScaler to minimize the impact of outliers on model training, using median and interquartile range for normalization.

**Feature Selection**: SelectKBest with f-regression scoring identifies the most predictive features, reducing overfitting risk and improving model generalization.

**Data Augmentation**: Synthetic samples are generated through interpolation between similar data points, effectively increasing the training dataset size and improving model robustness.

### 4.2 Models Trained

The system trains and compares multiple regression models:

**Ridge Regression (Baseline Linear Model)**: 
- Simple, interpretable linear model with L2 regularization
- Test R²: 0.6882
- Test RMSE: 11.48
- Provides a strong baseline for comparison

**ElasticNet**:
- Combines L1 and L2 regularization
- Test R²: 0.6641
- Test RMSE: 11.91
- Offers feature selection capabilities through L1 penalty

**XGBoost (Gradient Boosting)**:
- High-performance tree-based ensemble model
- Test R²: 0.7210
- Test RMSE: 10.86
- Excellent predictive power with built-in regularization

**LightGBM (Lightweight Gradient Boosting)**:
- Optimized for speed and memory efficiency
- Test R²: 0.7115
- Test RMSE: 11.04
- Competitive performance with faster training

**Ensemble Model (Best Performing)**:
- Voting regressor combining Ridge, XGBoost, and LightGBM
- Test R²: **0.7330 (73.3%)**
- Test RMSE: **10.62**
- Test MAE: **8.16**
- Achieves the highest accuracy through model combination

### 4.3 Model Performance Analysis

The Ensemble model achieves a test R² of 0.7330, indicating that 73.3% of the variance in AQI values can be explained by the model's predictions. This performance level is considered strong for time-series forecasting tasks, particularly given the inherent variability in air quality measurements.

The RMSE of 10.62 AQI units and MAE of 8.16 AQI units demonstrate that predictions are, on average, within approximately 8-11 AQI units of actual values. Given that AQI typically ranges from 0-500, with most values falling between 0-200, this error range represents a practical level of accuracy for real-world applications.

### 4.4 Hyperparameter Optimization

Model performance was improved through systematic hyperparameter tuning:
- Reduced learning rates (0.05) for better convergence
- Optimized tree depths (8-10) to balance complexity and generalization
- Adjusted regularization parameters (α=0.1, λ=1.5) to prevent overfitting
- Fine-tuned subsample and colsample_bytree ratios (0.85) for ensemble diversity

### 4.5 Feature Importance

Analysis of model coefficients and feature importance reveals that:
- **Current AQI** is the most predictive feature, as expected for time-series forecasting
- **AQI lag features** (especially lag_1, lag_2, lag_3) capture temporal dependencies
- **Weather features** (temperature, humidity, pressure) contribute significantly to predictions
- **Rolling statistics** and **difference features** capture trend patterns
- **Seasonal features** (day_of_year_sin, day_of_year_cos) help model long-term patterns

---

## 5. Improvements and Future Work

### 5.1 Implemented Enhancements

Several key improvements were made during development:

**Data Augmentation**: Implemented interpolation-based synthetic data generation, increasing training dataset size by 150%, which significantly improved model generalization and reduced overfitting.

**Weather Feature Integration**: Extended feature engineering to include weather variable lags and rolling statistics, capturing the relationship between meteorological conditions and air quality.

**Ensemble Methods**: Implemented voting regressor combining multiple models, achieving superior performance compared to individual models.

**Robust Preprocessing**: Replaced StandardScaler with RobustScaler to handle outliers more effectively, improving model stability.

**Feature Selection Optimization**: Increased feature selection from top 10 to top 40 features, capturing more predictive information while maintaining model efficiency.

**Fast Training Pipeline**: Optimized training scripts with reduced model complexity and efficient hyperparameters, enabling rapid model iteration and deployment.

### 5.2 Future Enhancements

Several areas present opportunities for further improvement:

**Real-Time Data Integration**: Integrate live AQI and weather data from APIs (AQICN, OpenWeather) to enable real-time predictions without manual data updates.

**Advanced Model Architectures**: Explore deep learning approaches such as:
- LSTM (Long Short-Term Memory) networks for sequential pattern recognition
- Temporal Convolutional Networks (TCN) for time-series forecasting
- Transformer models adapted for time-series data

**Automated Retraining Pipeline**: Implement CI/CD workflows using GitHub Actions or Apache Airflow to:
- Automatically retrain models daily with new data
- Perform model versioning and A/B testing
- Deploy updated models seamlessly

**Multi-City Support**: Extend the system to support predictions for multiple cities simultaneously, enabling comparative analysis and regional air quality monitoring.

**Alert System**: Implement real-time alerting when predicted AQI exceeds hazardous thresholds (>150), sending notifications via email, SMS, or push notifications.

**Model Explainability**: Reintroduce SHAP-based feature importance visualization to help users understand prediction drivers and build trust in the system.

**Hyperparameter Auto-Tuning**: Integrate automated hyperparameter optimization using Optuna or Hyperopt to systematically improve model performance.

**Cross-Validation Framework**: Implement time-series cross-validation to provide more robust performance estimates and reduce overfitting risk.

**Data Quality Monitoring**: Add data validation checks and anomaly detection to identify data quality issues before they impact model performance.

---

## 6. Conclusion

The Pearls AQI Predictor successfully demonstrates how machine learning can be effectively applied to environmental forecasting challenges. Through systematic feature engineering, model development, and performance optimization, the system achieves a predictive accuracy of 73.3% (R² = 0.7330), making it suitable for practical applications in air quality monitoring and public health advisories.

The project's modular architecture, comprehensive documentation, and interactive dashboard make it accessible to both technical and non-technical users. The emphasis on reproducibility, through configuration management and version control, ensures that the system can be easily maintained, extended, and deployed in various environments.

Key achievements include:
- Successful integration of multiple data sources (air quality and weather)
- Development of a robust feature engineering pipeline
- Achievement of strong predictive performance through ensemble methods
- Creation of an intuitive, interactive visualization interface
- Implementation of best practices in machine learning development

The system's performance metrics (RMSE: 10.62, MAE: 8.16, R²: 0.7330) demonstrate its practical utility for real-world applications. Environmental agencies, smart city planners, and public health organizations can leverage this system to:
- Provide early warnings for air quality deterioration
- Support data-driven policy decisions
- Enable proactive public health interventions
- Facilitate research into air pollution patterns and trends

As air quality continues to be a critical global concern, tools like the Pearls AQI Predictor play an essential role in translating data into actionable insights. The project's open-source nature and modular design make it a valuable contribution to the environmental monitoring community, with significant potential for further development and deployment.

Future enhancements, including real-time data integration, advanced deep learning models, and automated retraining pipelines, will further strengthen the system's capabilities and expand its applicability. The foundation established in this project provides a solid base for continued innovation in environmental forecasting and machine learning applications.

---

**End of Report**


# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.feature_store import load_features
from utils.model_loader import AQIPredictor

# SHAP imports
import shap
import numpy as np
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {font-size:24px; color: #1E88E5; font-weight: bold;}
    .metric-card {padding: 15px; border-radius: 10px; background-color: #f8f9fa; margin: 10px 0;}
    </style>
""", unsafe_allow_html=True)

# Initialize the predictor
@st.cache_resource
def load_predictor():
    return AQIPredictor()

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("### Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Random Forest", "XGBoost", "LSTM"],
    index=0
)

# Main content
st.title("🌫️ AQI Forecast Dashboard")
st.markdown("### Real-time Air Quality Index Prediction")

# Date range selection
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime.now().date())
with col2:
    end_date = st.date_input("End Date", value=datetime.now().date() + timedelta(days=3))

try:
    # Load features
    features_df = load_features(start_date, end_date)
    
    # Load model and make predictions
    predictor = load_predictor()
    predictions = predictor.predict(features_df)
    
    # Create results dataframe
    results = features_df[['date']].copy()
    results['AQI_Predicted'] = predictions
    
    # Display metrics
    st.markdown("### 📊 Prediction Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current AQI", f"{predictions[0]:.1f}", "Good" if predictions[0] < 50 else "Moderate")
    with col2:
        st.metric("Max AQI (3 days)", f"{max(predictions):.1f}")
    with col3:
        st.metric("Avg AQI (3 days)", f"{predictions.mean():.1f}")
    
    # Plot predictions
    st.markdown("### 📈 AQI Forecast")
    fig = px.line(
        results,
        x='date',
        y='AQI_Predicted',
        title='AQI Forecast for Next 3 Days',
        labels={'AQI_Predicted': 'Predicted AQI', 'date': 'Date'}
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="AQI",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if hasattr(predictor.model, 'feature_importances_'):
        st.markdown("### 🔍 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': predictor.model.feature_names_in_,
            'Importance': predictor.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 10 Most Important Features'
        )
        st.plotly_chart(fig, use_container_width=True)

    # SHAP explainability
    with st.expander("🔬 SHAP Explainability", expanded=False):
        try:
            # Derive features matrix exactly as used by the model
            if hasattr(predictor.model, 'feature_names_in_'):
                feature_cols = list(predictor.model.feature_names_in_)
                # Ensure all required features exist
                missing = [c for c in feature_cols if c not in features_df.columns]
                if missing:
                    st.warning(f"Missing features for SHAP: {missing}")
                    X_all = features_df[[c for c in feature_cols if c in features_df.columns]].select_dtypes(include=[np.number])
                else:
                    X_all = features_df[feature_cols].select_dtypes(include=[np.number])
            else:
                X_all = features_df.select_dtypes(include=[np.number])
                for c in ["aqi", "pm25", "pm_25", "target", "timestamp", "date"]:
                    if c in X_all.columns:
                        X_all = X_all.drop(columns=[c])
            
            # Sample to keep computation light (max 200 rows)
            sample_size = min(200, len(X_all))
            if sample_size >= 2:
                X_sample = X_all.iloc[:sample_size].fillna(X_all.median(numeric_only=True))
                explainer = shap.TreeExplainer(predictor.model)
                shap_values = explainer.shap_values(X_sample)
                # Plot summary
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, show=False)
                st.pyplot(plt.gcf(), clear_figure=True, use_container_width=True)
                st.caption(f"SHAP values computed for {len(X_sample)} samples. Features shown in order of importance.")
            else:
                st.info("Not enough rows to compute SHAP values. Expand the date range.")
        except Exception as e:
            st.warning(f"SHAP could not be computed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    
    # Raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(results)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure the feature pipeline has been run and the model is trained.")
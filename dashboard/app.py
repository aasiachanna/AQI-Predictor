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

# Initialize the predictor (cache by model_name to allow switching)
@st.cache_resource
def load_predictor(model_name=None):
    return AQIPredictor(model_name=model_name)

# Sidebar
st.sidebar.title("Settings")
st.sidebar.markdown("### Model Settings")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Ensemble_Fast", "XGBoost_Fast", "LightGBM_Fast", "Ridge_Fast"],
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
    with st.spinner("Loading features..."):
        features_df = load_features(start_date, end_date)
    
    if features_df.empty:
        st.warning("No features available for the selected date range. Please check your data.")
        st.stop()
    
    # Load model and make predictions (use selected model)
    with st.spinner(f"Loading {model_choice} model..."):
        predictor = load_predictor(model_name=model_choice)
    
    # Display model info
    if predictor.model_metadata:
        st.sidebar.markdown("### Model Metrics")
        metrics = predictor.model_metadata.get("metrics", {})
        if isinstance(metrics, dict):
            # Display key metrics
            for metric_name in ['test_rmse', 'test_mae', 'test_r2']:
                if metric_name in metrics:
                    value = metrics[metric_name]
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        # Format metric name for display
                        display_name = metric_name.replace('test_', '').upper()
                        st.sidebar.metric(display_name, f"{value:.4f}")
    
    # Make predictions
    with st.spinner("Making predictions..."):
        predictions = predictor.predict(features_df)
    
    # Create results dataframe
    if 'date' in features_df.columns:
        results = features_df[['date']].copy()
    else:
        # Create date column from index if missing
        results = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='D')[:len(predictions)]})
    
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
    
    # Raw data
    if st.checkbox("Show Raw Data"):
        st.dataframe(results)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure the feature pipeline has been run and the model is trained.")
import streamlit as st
import requests
import os
import pickle
import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

# Page configuration
st.set_page_config(
    page_title="Timelytics OTD Predictor",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Constants
MODEL_URL = "https://github.com/Ashvin1125/Timelytics/releases/download/v1.0.0/voting_model.pkl"
MODEL_PATH = "voting_model.pkl"
CACHE_DIR = ".cache"

# Create cache directory if not exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Download with retry logic
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (186 MB)... This may take 2-5 minutes"):
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            temp_path = f"{MODEL_PATH}.tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            os.rename(temp_path, MODEL_PATH)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        download_model()
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

model = load_model()

# Prediction function
def predict_delivery_time(inputs):
    try:
        prediction = model.predict(np.array([inputs]))
        return max(1, round(prediction[0]))
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

# UI Components
st.title("📦 Timelytics Order-to-Delivery Predictor")
st.markdown("""
Predict delivery times using our machine learning model.
The model will be downloaded automatically on first run (186 MB).
""")

with st.sidebar:
    st.header("🛒 Order Details")
    
    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        purchase_dow = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", 
                                 "Thursday", "Friday", "Saturday", "Sunday"][x],
            index=3
        )
    with col2:
        purchase_month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][x-1],
            index=0
        )
    
    year = st.number_input("Year", min_value=2015, max_value=2025, value=2023)
    
    # Product details
    st.subheader("Product Information")
    product_size_cm3 = st.number_input("Size (cm³)", min_value=1, value=9328)
    product_weight_g = st.number_input("Weight (grams)", min_value=1, value=1800)
    
    # Location details
    st.subheader("Location Information")
    col3, col4 = st.columns(2)
    with col3:
        geolocation_state_customer = st.selectbox(
            "Customer State",
            options=list(range(1, 28)),
            index=9
        )
    with col4:
        geolocation_state_seller = st.selectbox(
            "Seller State",
            options=list(range(1, 28)),
            index=19
        )
    
    distance = st.number_input("Distance (km)", min_value=0.1, value=475.35, step=1.0)
    
    submit = st.button("🚀 Predict Delivery Time", type="primary")

# Prediction logic
if submit:
    if model is None:
        st.error("Model not loaded. Please check the download.")
    else:
        inputs = [
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance
        ]
        
        prediction = predict_delivery_time(inputs)
        if prediction:
            st.success(f"## Predicted Delivery Time: {prediction} days")
            st.balloons()

# Sample data section
with st.expander("📋 Sample Inputs for Testing"):
    sample_data = {
        "Scenario": ["Small Package", "Heavy Item", "Long Distance"],
        "Day": ["Thursday", "Monday", "Tuesday"],
        "Month": ["Jan", "Jun", "Mar"],
        "Size (cm³)": [5000, 15000, 10000],
        "Weight (g)": [500, 5000, 2000],
        "Distance (km)": [100, 250, 800],
        "Typical OTD": ["3-5 days", "7-10 days", "12-15 days"]
    }
    st.table(pd.DataFrame(sample_data))

# Help section
with st.expander("❓ Help & Information"):
    st.markdown("""
    **Model Information:**
    - Size: 186 MB (downloaded from GitHub Releases)
    - Ensemble of XGBoost, Random Forest, and SVM
    - Accuracy: ±1.5 days MAE
    
    **Troubleshooting:**
    - Slow download? Wait 2-5 minutes on first run
    - Error? Refresh the page to retry
    """)



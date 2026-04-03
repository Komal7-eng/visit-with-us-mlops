import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Visit With Us Predictor", layout="centered")

# Load the saved model
@st.cache_resource
def load_model_from_hf():
    try:
        repo_id = "repo_id = "kaushalya7/mlops-visit-with-us-model"" 
        # Check if file exists in the hub and download
        model_path = hf_hub_download(repo_id=repo_id, filename="model.joblib", repo_type="model")
        return joblib.load(model_path)
    except Exception as e:
        # Yeh error UI par dikhega agar download fail hua
        st.error(f"⚠️ Model Loading Failed: {e}")
        return None

# Initializing model
model = load_model_from_hf()

# Match exact sequence from train_model.py
FEATURES = ['Age', 'MonthlyIncome', 'Passport', 'NumberOfTrips', 'PitchSatisfactionScore', 'Designation']

st.title("🌲 Visit With Us: Wellness Package Predictor")

# 1. UI Inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 70, 30)
    income = st.number_input("Monthly Income", 0, 150000, 25000)
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    passport = 1 if passport == "Yes" else 0
with col2:
    trips = st.number_input("Number of Trips", 0, 15, 2)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# 2. Prediction Logic
if st.button("Predict Purchase"):
    if model is None:
        st.error("❌ Model is not loaded. Please check logs.")
    else:
        # Encoding logic
        desig_mapping = {
             "Executive": 0,
             "Manager": 1,
             "Senior Manager": 2,
             "AVP": 3,
             "VP": 4
        }
        
        input_data = {
            'Age': age,
            'MonthlyIncome': income,
            'Passport': passport,
            'NumberOfTrips': trips,
            'PitchSatisfactionScore': satisfaction,
            'Designation': desig_mapping.get(designation, 0)
        }
        
        # Create DataFrame and ensure column order
        input_df = pd.DataFrame([input_data])[FEATURES]
        
        # Get prediction and probability
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1] # Probability of buying
        
        if prediction[0] == 1:
            st.success(f"✅ Prediction: Customer is LIKELY to buy! (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"❌ Prediction: Customer is UNLIKELY to buy. (Confidence: {(1-prob)*100:.1f}%)")

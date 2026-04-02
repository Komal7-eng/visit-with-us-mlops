import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Visit With Us Predictor", layout="centered")

# ✅ Load model from Hugging Face
@st.cache_resource
def load_model_from_hf():
    try:
        repo_id = "kaushalya7/mlops-visit-with-us-model"
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.joblib",
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"⚠️ Model Loading Failed: {e}")
        print(e)
        return None

# ✅ Load model with spinner
with st.spinner("Loading model..."):
    model = load_model_from_hf()

if model is None:
    st.error("❌ Model not loaded. Please check Hugging Face logs.")
    st.stop()

# ✅ Feature order (must match training)
FEATURES = [
    'Age',
    'MonthlyIncome',
    'Passport',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'Designation'
]

# ✅ Title
st.title("🌿 Visit With Us: Customer Purchase Predictor")
st.markdown("Predict whether a customer will purchase a wellness package.")

# ✅ UI Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    income = st.number_input("Monthly Income", 0, 150000, 25000)
    passport = st.selectbox("Has Passport? (1=Yes, 0=No)", [0, 1])

with col2:
    trips = st.number_input("Number of Trips", 0, 15, 2)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

# ✅ IMPORTANT: Correct encoding (must match training)
desig_mapping = {
    "Executive": 0,
    "Manager": 1,
    "Senior Manager": 2,
    "AVP": 3,
    "VP": 4
}

# ✅ Prediction
if st.button("Predict Purchase"):
    input_data = {
        'Age': age,
        'MonthlyIncome': income,
        'Passport': passport,
        'NumberOfTrips': trips,
        'PitchSatisfactionScore': satisfaction,
        'Designation': desig_mapping.get(designation, 0)
    }

    input_df = pd.DataFrame([input_data])[FEATURES]

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.success(f"✅ Likely to Buy (Confidence: {prob*100:.1f}%)")
    else:
        st.error(f"❌ Unlikely to Buy (Confidence: {(1-prob)*100:.1f}%)")
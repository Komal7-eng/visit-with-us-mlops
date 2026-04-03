import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Page config
st.set_page_config(page_title="Visit With Us", layout="centered")

# 💎 Luxury UI Styling (PUT HERE — top of file)
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #f8ede3, #f5d0c5);
}

.block-container {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(20px);
    border-radius: 25px;
    padding: 2rem;
    box-shadow: 0 10px 40px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-size: 48px;
    color: #3e2f2f;
}

h3 {
    text-align: center;
    font-size: 22px;
    color: #7a5c5c;
    margin-bottom: 30px;
}

.stButton>button {
    background: linear-gradient(135deg, #b88a5a, #8c6239);
    color: white;
    border-radius: 30px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

.stNumberInput input, .stSelectbox div {
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        repo_id = "kaushalya7/mlops-visit-with-us-model"
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename="model.joblib",
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# Stop if model fails
if model is None:
    st.stop()

# Features
FEATURES = ['Age', 'MonthlyIncome', 'Passport', 'NumberOfTrips', 'PitchSatisfactionScore', 'Designation']

# Titles
st.markdown("<h1>Visit With Us</h1>", unsafe_allow_html=True)
st.markdown("<h3>Customer Purchase Prediction</h3>", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 70, 30)
    income = st.number_input("Monthly Income", 0, 150000, 25000)
    passport = st.selectbox("Has Passport", ["No", "Yes"])

with col2:
    trips = st.number_input("Number of Trips", 0, 15, 2)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

# Encoding
passport_val = 1 if passport == "Yes" else 0

desig_mapping = {
    "Executive": 0,
    "Manager": 1,
    "Senior Manager": 2,
    "AVP": 3,
    "VP": 4
}

st.markdown("<br>", unsafe_allow_html=True)

# Prediction
if st.button("Predict"):
    input_data = {
        'Age': age,
        'MonthlyIncome': income,
        'Passport': passport_val,
        'NumberOfTrips': trips,
        'PitchSatisfactionScore': satisfaction,
        'Designation': desig_mapping[designation]
    }

    input_df = pd.DataFrame([input_data])[FEATURES]

    prediction = model.predict(input_df)
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("### Result")

    if prediction[0] == 1:
        st.success(f"Likely to Buy (Confidence: {prob*100:.1f}%)")
    else:
        st.error(f"Unlikely to Buy (Confidence: {(1-prob)*100:.1f}%)")

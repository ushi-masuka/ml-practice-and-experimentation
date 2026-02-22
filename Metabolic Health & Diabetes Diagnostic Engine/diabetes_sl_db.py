import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration (Looks professional instantly)
st.set_page_config(page_title="Diabetes Diagnostic Engine", layout="centered")

st.title("🩺 Metabolic Health & Diabetes Risk Predictor")
st.write("Enter patient vitals below to generate a real-time diabetes risk assessment using LightGBM.")

# 2. Load the pre-trained model
# (In a real scenario, you'd cache this using @st.cache_resource so it doesn't reload every click)
@st.cache_resource
def load_model():
    # Assuming you saved your trained model using joblib.dump(clf, 'lgbm_model.pkl')
    return joblib.load('lgbm_diabetes_model.pkl')

model = load_model()

# 3. Create the UI for Patient Inputs
st.subheader("Patient Vitals")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=45)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    
with col2:
    systolic_bp = st.number_input("Systolic Blood Pressure", min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=190)

# 4. Prediction Logic
if st.button("Calculate Risk Score"):
    
    # Pack the inputs into a dictionary, then a DataFrame (matches training format)
    input_data = {
        'age': [age],
        'bmi': [bmi],
        'systolic_bp': [systolic_bp],
        'cholesterol_total': [cholesterol]
    }
    
    features_df = pd.DataFrame(input_data)
    
    # Predict probability
    risk_proba = model.predict_proba(features_df)[0][1] * 100
    
    # 5. Display the output cleanly
    st.divider()
    if risk_proba > 50.0:
        st.error(f"⚠️ High Risk Detected: {risk_proba:.1f}% probability of diabetes.")
    else:
        st.success(f"✅ Low Risk: {risk_proba:.1f}% probability of diabetes.")
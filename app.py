import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    layout="centered"
)

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# Layout: 2 columns
col1, col2 = st.columns(2)

features = []

# Left side inputs
with col1:
    features.append(st.number_input("Pregnancies", value=None))
    features.append(st.number_input("Glucose", value=None))
    features.append(st.number_input("BloodPressure", value=None))
    features.append(st.number_input("SkinThickness", value=None))

# Right side inputs
with col2:
    features.append(st.number_input("Insulin", value=None))
    features.append(st.number_input("BMI", value=None))
    features.append(st.number_input("DiabetesPedigreeFunction", value=None))
    features.append(st.number_input("Age", value=None))

# Red Predict Button
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: red;
        color: white;
        font-size: 18px;
        padding: 10px 26px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Diabetic (Risk: {probability:.2f}%)")
    else:
        st.success(f"✅ Not Diabetic (Risk: {probability:.2f}%)")

import streamlit as st
import numpy as np
import joblib

model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction", layout="centered")

st.title("🩺 Diabetes Prediction App")

st.write("Enter patient details to predict diabetes")

features = []

fields = [
    "Pregnancies", "Glucose", "BloodPressure",
    "SkinThickness", "Insulin", "BMI",
    "DiabetesPedigreeFunction", "Age"
]

for field in fields:
    value = st.number_input(field, min_value=0.0)
    features.append(value)

if st.button("Predict"):
    input_data = np.array(features).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Diabetic (Risk: {probability:.2f}%)")
    else:
        st.success(f"✅ Not Diabetic (Risk: {probability:.2f}%)")

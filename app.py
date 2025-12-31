import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("productivity_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Employee Productivity Predictor", layout="centered")

st.title("👩‍💼 Employee Productivity Prediction App")
st.markdown("Predict employee productivity using Machine Learning")

st.divider()

# Input fields
age = st.number_input("Employee Age", 18, 65)
hours = st.slider("Working Hours per Day", 1, 12)
experience = st.number_input("Years of Experience", 0, 40)
salary = st.number_input("Monthly Salary", 10000, 300000)
team_size = st.slider("Team Size", 1, 50)

# Create input array
input_data = np.array([[age, hours, experience, salary, team_size]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict Productivity 🚀"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.success("🟥 Productivity Level: LOW")
    elif prediction[0] == 1:
        st.info("🟨 Productivity Level: MEDIUM")
    else:
        st.success("🟩 Productivity Level: HIGH")

st.divider()
st.caption("ML Model: Logistic Regression | Dataset: Kaggle Employee Productivity")

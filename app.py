import streamlit as st
import pandas as pd
import pickle

# ===============================
# Load Model Artifacts
# ===============================
model = pickle.load(open("productivity_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📊",
    layout="centered"
)

# ===============================
# Custom Styling
# ===============================
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# Header
# ===============================
st.markdown('<div class="title">👩‍💼 Employee Productivity Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Based Classification App</div>', unsafe_allow_html=True)
st.write("")

# ===============================
# Input Card
# ===============================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📝 Enter Employee Details")

user_input = {}

for feature in features:
    user_input[feature] = st.number_input(
        label=feature.replace("_", " "),
        min_value=0.0,
        value=0.0,
        step=1.0
    )

st.markdown('</div>', unsafe_allow_html=True)
st.write("")

# ===============================
# Prediction
# ===============================
if st.button("🚀 Predict Productivity"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Prediction Result")

    if prediction == 0:
        st.error("🔴 Productivity Level: LOW")
    elif prediction == 1:
        st.warning("🟡 Productivity Level: MEDIUM")
    else:
        st.success("🟢 Productivity Level: HIGH")

    st.markdown('</div>', unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("📊 Model: Logistic Regression | Dataset: Kaggle Employee Productivity")

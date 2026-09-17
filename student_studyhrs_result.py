import streamlit as st
import joblib
import os

st.title("Student Pass/Fail based on Study Hours")

MODEL_PATH = "Christ_College_francisanto_Deplyment.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the model: {type(e).__name__}: {e}")
    st.stop()

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

if st.button("Predict"):
    prediction = model.predict([[hours]])

    if prediction[0] == 1:
        st.success("Student will PASS")
    else:
        st.error("Student will FAIL")

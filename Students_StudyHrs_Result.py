
import streamlit as st
import joblib
import pandas as pd

model = joblib.load("logistic_regression_StudyHrs_model.pkl")

st.title("Student Pass/Fail Prediction")

st.write("Enter student details to predict the result.")

hours = st.number_input(
    "Enter Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

if st.button("Predict"):

    student = pd.DataFrame(
        [[hours]],
        columns=["StudyHours"]
    )

    prediction = model.predict(student)

    probability = model.predict_proba(student)

    if prediction[0] == 1:
        st.success("Student will PASS")
    else:
        st.error("Student will FAIL")

    st.write(
        "Probability of Pass:",
        round(probability[0][1] * 100, 2),
        "%"
    )

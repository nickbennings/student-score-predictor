import joblib
import pandas as pd
import streamlit as st

# Load the model you trained
model = joblib.load("student_score_model.pkl")

st.title("Student Score Predictor")
st.write(
    "Enter the number of hours studied to get a predicted exam score."
)

hours = st.slider("Hours studied", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

if st.button("Predict score"):
    input_data = pd.DataFrame({"hours_studied": [hours]})
    prediction = model.predict(input_data)[0]

    st.success(f"Predicted exam score: {prediction:.1f} out of 100")
    st.caption("For demonstration only: this model uses a small, made-up dataset.")

    
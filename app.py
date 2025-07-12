import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("Linear_electricity_bill_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app title
st.title("Electricity Bill Prediction App")

st.write("Fill the details below to predict your electricity bill:")

# User input form
with st.form("bill_form"):
    units = st.number_input("Units Consumed", min_value=0)
    ac_hours = st.slider("AC Usage Hours per Day", 0.0, 24.0, 0.0)
    heater_hours = st.slider("Heater Usage Hours per Day", 0.0, 24.0, 0.0)
    
    submit = st.form_submit_button("Predict Bill")

if submit:
    # Prepare data
    input_data = np.array([[units, ac_hours, heater_hours]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    # Display prediction
    st.success(f"Predicted Monthly Electricity Bill: ₹ {round(prediction[0], 2)}")

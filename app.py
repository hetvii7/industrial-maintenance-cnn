import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model("predictive_maintenance_cnn.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("AI-Based Predictive Maintenance System")

st.write("Enter Machine Sensor Values")

inputs = []

for i in range(1,10):
    val = st.number_input(f"Metric {i}")
    inputs.append(val)

if st.button("Predict Failure"):

    data = np.array([inputs])
    data_scaled = scaler.transform(data)
    data_cnn = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)

    prediction = model.predict(data_cnn)

    if prediction > 0.5:
        st.error("⚠️ Machine Failure Risk")
    else:
        st.success("✅ Machine is Safe")
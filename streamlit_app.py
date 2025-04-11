import streamlit as st
import pickle

import joblib

model = joblib.load("maintenance_classifier.pkl")

# UI
st.title("Complaint Categorization Portal 🛠️")
st.write("Enter a complaint and get its predicted category.")

complaint = st.text_area("Enter your complaint:")

if st.button("Predict Category"):
    if complaint.strip() == "":
        st.warning("Please enter a complaint.")
    else:
        pred = model.predict([complaint])[0]
        st.success(f"Predicted Category: {pred}")

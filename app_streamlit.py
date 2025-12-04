import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load your ML model
model = joblib.load("model/sign_model.pkl")

st.title("Sign Language Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR")

    # Here, you can run your Mediapipe and ML inference
    st.write("Prediction: ...")  # Replace with your actual model prediction

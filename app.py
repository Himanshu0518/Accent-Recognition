# app.py

import streamlit as st
import tempfile
import os
import av
import wave
import numpy as np
from src.pipeline.prediction_pipeline import AudioPredictor

st.set_page_config(page_title="Accent Recognition", layout="centered")
st.title("🗣️ Accent Recognition (Upload)")


predictor = AudioPredictor()

# 1. Upload Audio File

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if st.button("🎯 Predict Accent"):
            try:
                result = predictor.predict(tmp_path)
                st.success(f"Predicted Accent: **{result.capitalize()}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.warning("Ensure the audio file is in WAV format and contains clear speech.")


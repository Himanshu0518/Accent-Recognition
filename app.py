# streamlit_app.py

import streamlit as st
import tempfile
import librosa
import numpy as np
from src.pipeline.prediction_pipeline import AudioPredictor
from visualizer import (
    plot_waveform, plot_mel_spectrogram, plot_zcr, plot_rmse, plot_feature_importance
)

DAGSHUB_TRACKING_URL = "https://dagshub.com/Himanshu0518/Accent-Recognition.mlflow"

st.set_page_config(page_title="Accent Recognition", layout="centered")
st.markdown("<h1 style='text-align: center;'>🗣️ Accent Recognition App</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
    Upload a <code>.wav</code> file of speech and this app will predict the speaker's <b>accent</b>.
</div>
""", unsafe_allow_html=True)

st.divider()
predictor = AudioPredictor()

# Upload
st.subheader("📤 Upload Your Audio File")
uploaded_file = st.file_uploader("Choose a `.wav` file", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        y, sr = librosa.load(tmp_path, sr=None)

        # Dropdown for visualizations
        st.subheader("📊 Visualize Features")
        viz_option = st.selectbox(
            "Choose visualization type",
            ["Waveform", "Mel Spectrogram", "Zero Crossing Rate (ZCR)", "RMSE"]
        )

        if viz_option == "Waveform":
            st.pyplot(plot_waveform(y, sr))
        elif viz_option == "Mel Spectrogram":
            st.pyplot(plot_mel_spectrogram(y, sr))
        elif viz_option == "Zero Crossing Rate (ZCR)":
            st.pyplot(plot_zcr(y))
        elif viz_option == "RMSE":
            st.pyplot(plot_rmse(y))

    except Exception as e:
        st.warning(f"⚠️ Could not display visualizations: {e}")

    # Prediction
    st.subheader("🎯 Predict Accent")
    if st.button("🔍 Run Prediction"):
        try:
            result = predictor.predict(tmp_path)

            if isinstance(result, dict):
                st.success(f"Predicted Accent: **{max(result, key=result.get).capitalize()}**")
                st.subheader("Prediction Confidence")
                st.bar_chart(result)

                # Example feature importance
                st.subheader("🧠 Feature Importance (Example)")
                feature_importance = {
                    "MFCC": 0.35,
                    "ZCR": 0.2,
                    "RMSE": 0.15,
                    "Chroma": 0.1,
                    "Spectral Centroid": 0.2
                }
                st.pyplot(plot_feature_importance(feature_importance))

            else:
                st.success(f"Predicted Accent: **{result.capitalize()}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.warning("Ensure the audio is clear and in correct format (.wav)")

st.divider()

with st.expander("📌 How This Works"):
    st.markdown("""
    - ✅ Audio is **preprocessed** to extract features like MFCCs, Zero-Crossing Rate, and Energy.
    - 🔊 We apply **augmentation techniques** (noise, pitch shift, stretch) for robustness.
    - 🧠 A trained **machine learning model** (e.g., Logistic Regression) predicts the accent.
    - 📦 Entire pipeline is tracked and versioned using **DVC & Git** for reproducibility.
    """)

st.markdown("### 📈 MLflow Experiment Tracking")
st.markdown(f"[🔗 View on DagsHub]({DAGSHUB_TRACKING_URL})", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; font-size: small;'>Built with ❤️ using Streamlit | Project Tracked via DagsHub</div>", unsafe_allow_html=True)

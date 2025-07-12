import streamlit as st
import tempfile
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from src.pipeline.prediction_pipeline import AudioPredictor

st.set_page_config(page_title="Accent Recognition", layout="centered")
st.title("🗣️ Accent Recognition")

st.markdown("Upload a `.wav` file of speech, and this app will predict the **accent**.")

predictor = AudioPredictor()

uploaded_file = st.file_uploader("📤 Upload your WAV file", type=["wav"])

if uploaded_file:
    # Display audio player
    st.audio(uploaded_file, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Show waveform and spectrogram
    try:
        y, sr = librosa.load(tmp_path, sr=None)
        fig1, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        S = librosa.feature.melspectrogram(y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, ax=ax2, x_axis='time', y_axis='mel')
        ax2.set_title("Mel Spectrogram")
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"Could not display audio visualizations: {e}")

    # Predict button
    if st.button("🎯 Predict Accent"):
        try:
            result = predictor.predict(tmp_path)

            if isinstance(result, dict):
                # If result contains probabilities
                st.success(f"Predicted Accent: **{max(result, key=result.get).capitalize()}**")
                st.subheader("Prediction Confidence")
                st.bar_chart(result)
            else:
                st.success(f"Predicted Accent: **{result.capitalize()}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.warning("Ensure the audio file is in WAV format and contains clear speech.")

# Optional: Explain the pipeline
with st.expander("📌 How This Works"):
    st.markdown("""
    - ✅ The audio is **preprocessed** to extract features like MFCCs, Zero-Crossing Rate, and Energy.
    - 🔊 We apply **augmentation techniques** (noise, pitch shift, stretch) to improve robustness.
    - 🧠 A trained **machine learning model** (e.g., Logistic Regression) predicts the accent.
    - 📦 The entire pipeline is tracked and versioned using **DVC & Git** for reproducibility.
    """)


import os
import librosa
import numpy as np
import pandas as pd
from src.components.feature_extraction import FeatureExtractor
from src.utils.main_utils import load_object
from from_root import from_root
from src.logger import logging
from src.constants import ARTIFACT_DIR

class AudioPredictor:
    def __init__(self, model_path=None, preprocessor_path=None):
        try:
            self.model_path = model_path or os.path.join(from_root(), "models", "model.joblib")
            self.preprocessor_path = preprocessor_path or os.path.join(ARTIFACT_DIR, "preprocessor.joblib")
            self.label_encoder_path = os.path.join(ARTIFACT_DIR, "label_encoder.joblib")
            self.label_encoder = load_object(self.label_encoder_path)
            self.model = load_object(self.model_path)
            self.preprocessor = load_object(self.preprocessor_path)
            self.fe = FeatureExtractor()
            
            logging.info("AudioPredictor initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            raise

    def predict(self, audio_path):
        try:
            features = self.fe.extract_features(audio_path)

            if features is None:
                raise ValueError("No features extracted from the audio file.")

            # Define column names same as used during training
            columns = [f"mfcc_{i+1}" for i in range(13)] + ["zcr", "rmse"]
            features_df = pd.DataFrame([features], columns=columns)

            logging.info(f"Extracted features: {features_df.shape}, Columns: {features_df.columns.tolist()}")
            transformed_features = self.preprocessor.transform(features_df)
            prediction = self.model.predict(transformed_features)
         
            logging.info(f"Prediction: {prediction[0]}")

            decoded_label = self.label_encoder.inverse_transform([prediction])[0]
            return decoded_label  # Return the decoded label directly

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return None  # Return None instead of crashing

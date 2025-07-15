import os
import librosa
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from src.constants import MLFLOW_TRACKING_URI
from src.components.feature_extraction import FeatureExtractor
from src.utils.main_utils import load_object
from src.constants import SAMPLE_RATE, DURATION, MODEL_DIR
from src.logger import logging


mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)


class AudioPredictor:
    """
    Predicts speaker accent from an audio file by:
      1. Loading a registered MLflow model (with preprocessing inside)
      2. Extracting MFCC+ZCR+RMSE features from raw audio
      3. Feeding features directly to the model
      4. Decoding the predicted label with a saved label encoder
    """

    def __init__(self, model_version: str = "2"):
        try:
            # --- Load registered MLflow model ------------------------
            self.model_uri = f"models:/AccentClassifier/{model_version}"
            self.model = mlflow.pyfunc.PyFuncModel = mlflow.pyfunc.load_model(
                self.model_uri
            )

           
            self.label_encoder = load_object(
                os.path.join(MODEL_DIR, "label_encoder.joblib")
            )
            self.fe = FeatureExtractor()

            logging.info("AudioPredictor initialised (model %s).", self.model_uri)
        except Exception as exc:
            logging.exception("Failed to initialise AudioPredictor: %s", exc)
            raise


    def predict(self, audio_path: str) -> str | None:
        """
        Parameters
        ----------
        audio_path : str
            Path to a .wav / .flac / .mp3 file.

        Returns
        -------
        str | None
            Predicted accent label, or None if prediction failed.
        """
        try:
            # --- Load & trim/pad audio --------------------------------
            y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, duration=DURATION)
            if y.size == 0:
                raise ValueError("Audio file is empty or unreadable")

            # --- Feature extraction -----------------------------------
            features = self.fe.extract_features(y)
            if features is None:
                raise ValueError("Feature extractor returned None")

            cols = [f"mfcc_{i+1}" for i in range(13)] + ["zcr", "rmse"]
            X = pd.DataFrame([features], columns=cols)

            # --- Inference (pre‑processing is inside model) -----------
            y_pred = self.model.predict(X)
            decoded = self.label_encoder.inverse_transform(np.ravel(y_pred))[0]

            logging.info("Prediction complete: %s", decoded)
            return decoded

        except Exception as exc:
            logging.exception("Prediction failed: %s", exc)
            return None




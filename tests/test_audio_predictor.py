import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

# ─────────────────────────────────────────────────────────────────────────────
# Test ‑‑ predict returns the decoded class label
# ─────────────────────────────────────────────────────────────────────────────
@patch("src.pipeline.prediction_pipeline.librosa.load")
@patch("src.pipeline.prediction_pipeline.FeatureExtractor")
@patch("src.pipeline.prediction_pipeline.load_object")
def test_predict_success(mock_load_object, mock_feature_extractor_cls, mock_librosa_load):
    """
    GIVEN a valid *fake* audio path
    WHEN  AudioPredictor.predict is called
    THEN  it should return the decoded label supplied by the mocked label‑encoder
    """
    # ---------- Arrange ------------------------------------------------------
    # 1) Fake waveform returned by librosa.load
    sample_rate = 16_000
    duration_sec = 2
    fake_wave = np.random.randn(sample_rate * duration_sec)
    mock_librosa_load.return_value = (fake_wave, sample_rate)

    # 2) Mock FeatureExtractor instance → extract_features → 15‑value feature vector
    fake_features = [0] * 15
    mock_fe_instance = MagicMock()
    mock_fe_instance.extract_features.return_value = fake_features
    mock_feature_extractor_cls.return_value = mock_fe_instance

    # 3) Mock objects returned by load_object()
    #    We’ll decide which one to hand back based on the path suffix.
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0])

    fake_preprocessor = MagicMock()
    fake_preprocessor.transform.return_value = np.array([[0] * 15])

    class FakeLabelEncoder:
        def inverse_transform(self, arr):
            return np.array(["speech"])

    def _load_object_side_effect(path):
        if path.endswith("model.joblib"):
            return fake_model
        if path.endswith("preprocessor.joblib"):
            return fake_preprocessor
        if path.endswith("label_encoder.joblib"):
            return FakeLabelEncoder()
        raise ValueError("Unexpected path in load_object side‑effect")

    mock_load_object.side_effect = _load_object_side_effect

    # ---------- Act ----------------------------------------------------------
    from src.pipeline.prediction_pipeline import AudioPredictor  # local import after patches
    predictor = AudioPredictor()
    label = predictor.predict("dummy/audio.wav")

    # ---------- Assert -------------------------------------------------------
    assert label == "speech"
    mock_librosa_load.assert_called_once()              # loaded audio
    mock_fe_instance.extract_features.assert_called_once_with(fake_wave)
    fake_preprocessor.transform.assert_called_once()
    fake_model.predict.assert_called_once()

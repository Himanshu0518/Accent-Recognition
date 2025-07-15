import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

@patch("src.pipeline.prediction_pipeline.mlflow.pyfunc.load_model")
@patch("src.pipeline.prediction_pipeline.librosa.load")
@patch("src.pipeline.prediction_pipeline.FeatureExtractor")
@patch("src.pipeline.prediction_pipeline.load_object")
def test_predict_success(mock_load_object, mock_feature_extractor_cls, mock_librosa_load, mock_mlflow_model):
    """
    GIVEN a valid *fake* audio path
    WHEN  AudioPredictor.predict is called
    THEN  it should return the decoded label supplied by the mocked label‑encoder
    """

    # 1. Fake waveform returned by librosa.load
    sample_rate = 16_000
    duration_sec = 2
    fake_wave = np.random.randn(sample_rate * duration_sec)
    mock_librosa_load.return_value = (fake_wave, sample_rate)

    # 2. Mock FeatureExtractor instance → extract_features → 15-value feature vector
    fake_features = [0] * 15
    mock_fe_instance = MagicMock()
    mock_fe_instance.extract_features.return_value = fake_features
    mock_feature_extractor_cls.return_value = mock_fe_instance

    # 3. Mock mlflow model
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0])
    mock_mlflow_model.return_value = fake_model

    # 4. Mock preprocessor
    fake_preprocessor = MagicMock()
    fake_preprocessor.transform.return_value = np.array([[0]*15])

    # 5. Mock label encoder
    class FakeLabelEncoder:
        def inverse_transform(self, arr):
            return np.array(["speech"])

    # 6. Return correct mock based on file name
    def _load_object_side_effect(path):
        if path.endswith("label_encoder.joblib"):
            return FakeLabelEncoder()
        if path.endswith("preprocessor.joblib"):
            return fake_preprocessor
        return None

    mock_load_object.side_effect = _load_object_side_effect

    # 7. Local import (after mocks are applied)
    from src.pipeline.prediction_pipeline import AudioPredictor
    predictor = AudioPredictor()

    # 8. Call predict
    label = predictor.predict("dummy/audio.wav")

    # 9. Assert the prediction
    assert label == "speech"
    mock_librosa_load.assert_called_once()
    mock_fe_instance.extract_features.assert_called_once_with(fake_wave)
    fake_preprocessor.transform.assert_called_once()
    fake_model.predict.assert_called_once()

import numpy as np
from unittest.mock import MagicMock, patch


@patch("src.pipeline.prediction_pipeline.mlflow.pyfunc.load_model")
@patch("src.pipeline.prediction_pipeline.librosa.load")
@patch("src.pipeline.prediction_pipeline.FeatureExtractor")
@patch("src.pipeline.prediction_pipeline.load_object")
def test_predict_success(
    mock_load_object,
    mock_feature_extractor_cls,
    mock_librosa_load,
    mock_mlflow_load_model,
):
    """AudioPredictor.predict should return the decoded label."""

    # 1️⃣  Fake audio returned by librosa.load
    sr = 16_000
    fake_wave = np.random.randn(sr * 2)  # 2‑second clip
    mock_librosa_load.return_value = (fake_wave, sr)

    # 2️⃣  Mock FeatureExtractor → extract_features
    fake_features = [0] * 15
    fe_instance = MagicMock()
    fe_instance.extract_features.return_value = fake_features
    mock_feature_extractor_cls.return_value = fe_instance

    # 3️⃣  Mock MLflow model
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0])
    mock_mlflow_load_model.return_value = fake_model

    # 4️⃣  Mock label encoder
    class FakeLabelEncoder:
        def inverse_transform(self, arr):
            return np.array(["speech"])

    def _load_object_side_effect(path):
        # Only label_encoder.joblib is expected now
        if path.endswith("label_encoder.joblib"):
            return FakeLabelEncoder()
        raise FileNotFoundError(path)

    mock_load_object.side_effect = _load_object_side_effect

    # 5️⃣  Import under test (after mocks)
    from src.pipeline.prediction_pipeline import AudioPredictor

    predictor = AudioPredictor(model_version="2")
    label = predictor.predict("dummy.wav")

    # 6️⃣  Assertions
    assert label == "speech"
    mock_librosa_load.assert_called_once()
    fe_instance.extract_features.assert_called_once_with(fake_wave)
    fake_model.predict.assert_called_once()

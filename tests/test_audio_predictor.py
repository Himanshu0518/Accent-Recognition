import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.prediction_pipeline import AudioPredictor

@patch("src.pipeline.prediction_pipeline.mlflow.pyfunc.load_model")
@patch("src.pipeline.prediction_pipeline.load_object")
@patch("src.pipeline.prediction_pipeline.FeatureExtractor")
@patch("src.pipeline.prediction_pipeline.librosa.load")
def test_predict_success(
    mock_librosa_load,
    mock_feature_extractor_cls,
    mock_load_object,
    mock_mlflow_load_model,
):
    # Mock the audio loading
    mock_librosa_load.return_value = ( [0.1] * 22050, 16000 )  # 1 second of dummy audio

    # Mock the feature extractor
    mock_fe_instance = MagicMock()
    mock_fe_instance.extract_features.return_value = [0.1] * 15
    mock_feature_extractor_cls.return_value = mock_fe_instance

    # Mock the label encoder
    mock_label_encoder = MagicMock()
    mock_label_encoder.inverse_transform.return_value = ["Indian"]
    mock_load_object.return_value = mock_label_encoder

    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.return_value = [[0]]
    mock_mlflow_load_model.return_value = mock_model

    # Run the predictor
    predictor = AudioPredictor(model_version="2")
    result = predictor.predict("dummy.wav")

    assert result == "Indian"

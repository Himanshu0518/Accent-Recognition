from unittest.mock import patch, MagicMock
from src.pipeline.prediction_pipeline import AudioPredictor

@patch("src.pipeline.prediction_pipeline.mlflow.pyfunc.load_model")
@patch("src.pipeline.prediction_pipeline.load_object")
@patch("src.pipeline.prediction_pipeline.FeatureExtractor")
@patch("src.pipeline.prediction_pipeline.librosa.load")
def test_predict(mock_load, mock_FE, mock_load_obj, mock_model_load):
    mock_load.return_value = ([0.1] * 16000, 16000)

    mock_FE.return_value.extract_features.return_value = [0.1] * 15
    mock_load_obj.return_value.inverse_transform.return_value = ["Indian"]
    mock_model_load.return_value.predict.return_value = [[0]]

    result = AudioPredictor().predict("test.wav")
    assert result == "Indian"

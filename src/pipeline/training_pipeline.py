# src/pipeline/training_pipeline.py

import os
import mlflow
import joblib
from dotenv import load_dotenv
from src.logger import logging
from from_root import from_root
from src.components.feature_extraction import FeatureExtractor
from src.components.data_preprocessing import DataPreprocessor
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluator

# Load environment variables
load_dotenv()

# Set MLflow/DagsHub tracking URI and credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("DAGS_HUB_URI"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "Accent Recognition"))

def run_feature_extraction():
    data_dir = os.path.join(from_root(), "data", "raw")
    output_csv = os.path.join(from_root(), "data", "processed", "features.csv")
    logging.info("Starting feature extraction process.")
    FeatureExtractor(sr=22050, duration=10).initialize_feature_extractor(data_dir, output_csv)
    logging.info("Feature extraction completed successfully.")

def run_data_preprocessing():
    input_csv = os.path.join(from_root(), "data", "processed", "features.csv")
    output_dir = os.path.join(from_root(), "data", "preprocessed")
    logging.info("Starting data preprocessing...")
    DataPreprocessor(input_csv=input_csv, output_dir=output_dir).initiate_data_preprocessing()
    logging.info("Data preprocessing finished.")

def run_model_training():
    train_csv = os.path.join(from_root(), "data", "preprocessed", "train_data.csv")
    logging.info("Starting model training...")
    trainer = ModelTrainer(model_path="models/model.joblib")
    model, model_name, model_params = trainer.initiate_model_training(train_csv)
    logging.info("Model training completed.")
    return model, model_name, model_params

def run_model_evaluation():
    test_csv = os.path.join(from_root(), "data", "preprocessed", "test_data.csv")
    logging.info("Starting model evaluation...")
    evaluator = ModelEvaluator(model_path="models/model.joblib")
    metrics, y_pred, y_true = evaluator.initiate_model_evaluation(test_csv)
    logging.info("Model evaluation completed.")
    return metrics, y_pred, y_true

# Entry Point
if __name__ == "__main__":
    try:
        run_feature_extraction()
        run_data_preprocessing()
        model, model_name, model_params = run_model_training()
        metrics, y_pred, y_true = run_model_evaluation()

        # MLflow Logging
        with mlflow.start_run():
            logging.info("Logging parameters and metrics to MLflow...")

            mlflow.log_param("model_name", model_name)
            mlflow.log_params(model_params)

            for key, value in metrics.items():
                mlflow.log_metric(key, value)

            model_artifact_path = os.path.join(from_root(), "models", "model.joblib")
            

            try:
                import mlflow.sklearn
                mlflow.sklearn.log_model(model,artifact_path="model", registered_model_name=model_name)
                logging.info("Model logged using mlflow.sklearn.")
            except Exception as e:
                mlflow.log_artifact(model_artifact_path, artifact_path="model")
                logging.warning(f"Could not use mlflow.sklearn. Used log_artifact instead. Reason: {e}")

            logging.info("All parameters, metrics, and model logged to MLflow.")

    except Exception as e:
        logging.error(f"Training + Evaluation pipeline failed: {e}")

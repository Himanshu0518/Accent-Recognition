# src/pipeline/training_pipeline.py

import os
from src.logger import logging
from from_root import from_root

from src.components.feature_extraction import FeatureExtractor
from src.components.data_preprocessing import DataPreprocessor
from src.components.model_training import ModelTrainer
from src.components.model_evaluation import ModelEvaluator

def run_feature_extraction():
    data_dir = os.path.join(from_root(), "data/raw")
    output_csv = os.path.join(from_root(), "data/interim/features.csv")

    logging.info("Starting feature extraction process.")
    feature_extractor = FeatureExtractor(sr=22050, duration=10)
    feature_extractor.initialize_feature_extractor(data_dir, output_csv)
    logging.info("Feature extraction completed successfully.")


def run_data_preprocessing():
    input_csv = os.path.join(from_root(), "data/processed/features.csv")
    output_dir = os.path.join(from_root(), "data/preprocessed")

    logging.info("Starting data preprocessing...")
    preprocessor = DataPreprocessor(input_csv=input_csv, output_dir=output_dir)
    preprocessor.initiate_data_preprocessing()
    logging.info("Data preprocessing finished.")


def run_model_training():
    train_csv = os.path.join(from_root(), "data/preprocessed/train_data.csv")

    logging.info("Starting model training...")
    trainer = ModelTrainer(model_path="models/model.joblib")
    trainer.initiate_model_training(train_csv)
    logging.info("Model training completed.")


def run_model_evaluation():
    test_csv = os.path.join(from_root(), "data/preprocessed/test_data.csv")

    logging.info("Starting model evaluation...")
    evaluator = ModelEvaluator(model_path="models/model.joblib")
    evaluator.initiate_model_evaluation(test_csv)
    logging.info("Model evaluation completed.")


# 🟢 Entry Point
if __name__ == "__main__":
    try:
        run_feature_extraction()
    except Exception as e:
        logging.error(f"Feature extraction failed: {e}")

    try:
        run_data_preprocessing()
    except Exception as e:
        logging.error(f"Data preprocessing failed: {e}")

    try:
        run_model_training()
    except Exception as e:
        logging.error(f"Model training failed: {e}")

    try:
        run_model_evaluation()
    except Exception as e:
        logging.error(f"Model evaluation failed: {e}")

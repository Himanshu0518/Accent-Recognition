import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.main_utils import load_dataframe, load_object, write_yaml
from src.logger import logging
from from_root import from_root
from src.constants import *

class ModelEvaluator:
    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.metrics_path = METRICS_PATH
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def initiate_model_evaluation(self, test_csv: str):
        try:
            # Step 1: Load test data
            df = load_dataframe(test_csv)
            X_test = df.drop(columns=['label'])
            y_true = df['label']

            # Step 2: Load trained model
            model = load_object(self.model_path)
            logging.info("Model loaded successfully.")

            # Step 3: Predict
            logging.info("Predicting on test data...")
            y_pred = model.predict(X_test)

            # Step 4: Calculate metrics
            logging.info("Calculating evaluation metrics...")
            metrics = {
                'accuracy': round(accuracy_score(y_true, y_pred), 4),
                'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
                'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
                'f1_score': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            }

    
            # Step 7: Save metrics locally
            write_yaml(self.metrics_path, metrics)
            logging.info(f"Metrics saved locally at: {self.metrics_path}")
            return metrics , y_pred , y_true
            
        except Exception as e:
            logging.error(f"Model evaluation failed: {e}")
            raise e


import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.utils.main_utils import load_dataframe, load_object, write_yaml
from src.logger import logging
from from_root import from_root


class ModelEvaluator:
    def __init__(self, model_path="models/model.joblib"):
        self.model_path = os.path.join(from_root(), model_path)
        self.metrics_path = os.path.join(from_root(), "reports", "metrics.yaml")
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

    def initiate_model_evaluation(self, test_csv: str):
        try:
           
            df = load_dataframe(test_csv)
            X_test = df.drop(columns=['label'])
            y_true = df['label']

            
            model = load_object(self.model_path)

            logging.info("Predicting on test data...")
            y_pred = model.predict(X_test)

            logging.info("Calculating evaluation metrics...")
            metrics = {
                'accuracy': round(accuracy_score(y_true, y_pred), 4),
                'precision': round(precision_score(y_true, y_pred, average='weighted', zero_division=0), 4),
                'recall': round(recall_score(y_true, y_pred, average='weighted', zero_division=0), 4),
                'f1_score': round(f1_score(y_true, y_pred, average='weighted', zero_division=0), 4)
            }

            logging.info(f"Evaluation metrics:\n{metrics}")
            write_yaml(self.metrics_path, metrics)
            logging.info(f"Metrics saved to: {self.metrics_path}")

        except Exception as e:
            logging.error(f"Model evaluation failed: {e}")
            raise e

if __name__ == "__main__":
    test_csv = os.path.join(from_root(), "data/preprocessed/test_data.csv")

    logging.info("Starting model evaluation process.")
    evaluator = ModelEvaluator(model_path="models/model.joblib")
    evaluator.initiate_model_evaluation(test_csv)
    logging.info("Model evaluation completed successfully.")
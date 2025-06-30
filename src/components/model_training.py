import os
from from_root import from_root
from src.logger import logging
from src.utils.main_utils import load_dataframe, read_yaml, save_object
from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators


class ModelTrainer:
    def __init__(self, model_path="models/model.joblib"):
        self.model_path = os.path.join(from_root(), model_path)

    def _get_model_instance(self, model_name: str, model_params: dict) -> ClassifierMixin:
        """
        Dynamically retrieve model class from sklearn by name.
        """
        estimators = dict(all_estimators(type_filter='classifier'))
        if model_name not in estimators:
            raise ValueError(f" Unsupported model type: {model_name}")
        return estimators[model_name](**model_params)

    def initiate_model_training(self, train_csv: str) -> None:
        try:
            logging.info("Loading training data...")
            df = load_dataframe(train_csv)
            X = df.drop(columns=['label'])
            y = df['label']

            logging.info("Reading training configuration from YAML...")
            config_path = os.path.join(from_root(), "params.yaml")
            params = read_yaml(config_path)
            model_name = params['model_training']['model']
            model_params = params['model_training'].get('hyperparameters', {})

            logging.info(f"Initializing model: {model_name} with params: {model_params}")
            model = self._get_model_instance(model_name, model_params)

            logging.info("Training model...")
            model.fit(X, y)


            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            save_object(self.model_path, model)

            logging.info(f"Model training complete. Saved to {self.model_path}")
            return model,model_name, model_params 
        
        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise e

if __name__ == "__main__":
    train_csv = os.path.join(from_root(), "data/preprocessed/train_data.csv")

    logging.info("Starting model training process.")
    trainer = ModelTrainer(model_path="models/model.joblib")
    trainer.initiate_model_training(train_csv)
    logging.info("Model training completed successfully.")
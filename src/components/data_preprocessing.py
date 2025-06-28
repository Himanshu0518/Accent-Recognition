import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import joblib
from src.utils.main_utils import load_dataframe, save_dataframe

from src.logger import logging


class DataPreprocessor:
    def __init__(self, input_csv: str, output_dir: str, test_size=0.2, random_state=42):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state

        os.makedirs(self.output_dir, exist_ok=True)

    def preprocess(self, X, y):
        """
        Encode labels and scale features.
        Returns processed DataFrame and fitted encoders.
        """
        logging.info("🔠 Encoding labels and scaling features...")

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        preprocessor = ColumnTransformer(transformers=[
            ('scaler', StandardScaler(), X.columns)
        ], remainder='passthrough')

        X_scaled = preprocessor.fit_transform(X)

        df_processed = pd.DataFrame(X_scaled, columns=X.columns)
        df_processed['label'] = y_encoded

        return df_processed, label_encoder, preprocessor

    def initiate_data_preprocessing(self):
        try:
            
            df = load_dataframe(self.input_csv)

            X = df.drop('label', axis=1)
            y = df['label']

            logging.info("🔀 Performing train-test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )

            logging.info("⚙️ Preprocessing training data...")
            train_df, label_encoder, scaler = self.preprocess(X_train, y_train)

            logging.info("⚙️ Preprocessing testing data...")
            test_df, _, _ = self.preprocess(X_test, y_test)  # Reuse encoders if needed

            # Save processed CSVs
            train_path = os.path.join(self.output_dir, 'train_data.csv')
            test_path = os.path.join(self.output_dir, 'test_data.csv')

           
            save_dataframe(train_df, train_path)
            save_dataframe(test_df, test_path)

            # # Save the label encoder and scaler
            # joblib.dump(label_encoder, os.path.join(self.output_dir, 'label_encoder.joblib'))
            # joblib.dump(scaler, os.path.join(self.output_dir, 'scaler.joblib'))

            logging.info("✅ Data preprocessing completed and saved successfully.")

        except Exception as e:
            logging.exception(f"❌ Error during preprocessing: {e}")
            raise

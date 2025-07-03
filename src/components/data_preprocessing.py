import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from src.utils.main_utils import load_dataframe, save_dataframe
from from_root import from_root
from src.logger import logging
from src.utils.main_utils import save_object
from src.constants import *

class DataPreprocessor:
    def __init__(self, input_csv: str, output_dir: str, test_size=TEST_SIZE, random_state=RANDOM_STATE):
        self.input_csv = input_csv
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.artifact_dir = ARTIFACT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
    
    def label_encode(self, y):
        """
        Encode labels using LabelEncoder.
        Returns encoded labels and fitted encoder.
        """
        logging.info("Encoding labels...")
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        return y_encoded, label_encoder
    
    def preprocess(self, X):
        """
        scale features.
        Returns processed DataFrame and fitted .
        """
  
        preprocessor = ColumnTransformer(transformers=[
            ('scaler', StandardScaler(), X.columns)
        ], remainder='passthrough')

        X_scaled = preprocessor.fit_transform(X)
        df_processed = pd.DataFrame(X_scaled, columns=X.columns)
       
        return df_processed, preprocessor

    def initiate_data_preprocessing(self):
        try:
            
            df = load_dataframe(self.input_csv)

            X = df.drop('label', axis=1)
            y = df['label']

            logging.info("Performing train-test split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
            )
            
            logging.info("Preprocessing training data...")
            train_df, preprocessor = self.preprocess(X_train)
            y_train_encoded, label_encoder = self.label_encode(y_train)
            train_df['label'] = y_train_encoded
            

            logging.info("Preprocessing testing data...")
            
            test_df,  _ = self.preprocess(X_test)  
            test_df['label'] = label_encoder.transform(y_test)  
            # Save processed CSVs
            train_path = os.path.join(self.output_dir, 'train_data.csv')
            test_path = os.path.join(self.output_dir, 'test_data.csv')

           
            save_dataframe(train_df, train_path)
            save_dataframe(test_df, test_path)


            # # Save the label encoder and scaler
           
            save_object( os.path.join(self.artifact_dir, 'label_encoder.joblib') , label_encoder)
            save_object( os.path.join(self.artifact_dir, 'preprocessor.joblib') , preprocessor)
         

            logging.info("Data preprocessing completed and saved successfully.")

        except Exception as e:
            logging.exception(f"Error during preprocessing: {e}")
            raise

if __name__ == "__main__":
    input_csv = FEATURES_CSV
    output_dir = PREPROCESSED_DATA_DIR
    logging.info("Starting data preprocessing...")
    preprocessor = DataPreprocessor(input_csv=input_csv, output_dir=output_dir)
    preprocessor.initiate_data_preprocessing()
    logging.info("Data preprocessing finished.")
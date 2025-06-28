import os
import pandas as pd
import joblib
from src.logger import logging
import yaml



def load_dataframe(csv_path: str) -> pd.DataFrame:
    try:
        logging.info(f"Loading DataFrame from {csv_path}")
        df = pd.read_csv(csv_path)
        logging.info("DataFrame loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"Failed to load DataFrame: {e}")
        raise


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logging.info(f"DataFrame saved successfully at {path}")
    except Exception as e:
        logging.error(f"Failed to save DataFrame to {path}: {e}")
        raise


def save_object(file_path: str, obj) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        logging.error(f"Failed to save object: {e}")
        raise


def load_object(file_path: str):
    try:
        logging.info(f"Loading object from {file_path}")
        obj = joblib.load(file_path)
        logging.info("Object loaded successfully.")
        return obj
    except Exception as e:
        logging.error(f"Failed to load object: {e}")
        raise


def read_yaml(file_path: str) -> dict:
    """
    Reads a YAML file and returns its content as a dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            content = yaml.safe_load(f)
        logging.info(f"YAML file loaded from {file_path}")
        return content
    except Exception as e:
        logging.error(f"Failed to read YAML file: {e}")
        raise


def write_yaml(file_path: str, content: dict) -> None:
    """
    Writes a dictionary to a YAML file.
    """
    try:
        with open(file_path, 'w') as f:
            yaml.safe_dump(content, f)
        logging.info(f"YAML file written to {file_path}")
    except Exception as e:
        logging.error(f"Failed to write YAML file: {e}")
        raise

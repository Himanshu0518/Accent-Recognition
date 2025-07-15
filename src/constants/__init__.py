import os
from from_root import from_root


# Feature extraction constants 


SAMPLE_RATE = 22050       # Audio sampling rate
DURATION = 10             # Duration of audio to load (in seconds)

FRAME_LENGTH = 1024       # Frame size for FFT
HOP_LENGTH = 512          # Hop size for STFT

NUM_FILES = 90            # Number of files per accent
MFCC_COUNT = 13           # Number of MFCCs to extract


LABELS = {
    "indian_accent": "indian",
    "american_accent": "american",
    "british_accent": "british"
}

RAW_DATA_DIR = os.path.join(from_root(), "data", "raw")
PREPROCESSED_DATA_DIR = os.path.join(from_root(), "data", "preprocessed")
INTERIM_DATA_DIR = os.path.join(from_root(), "data", "interim")
DATA_AUGMENTED_DIR = os.path.join(from_root(),"data","augmented_data")

# Data Preprocessing constants

FEATURES_CSV = os.path.join(from_root(), "data", "interim", "features.csv")
TRAIN_DATA = os.path.join(from_root(), "data", "preprocessed", "train_data.csv")
TEST_DATA = os.path.join(from_root(), "data", "preprocessed", "test_data.csv")
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Training constants
MODEL_PATH = os.path.join(from_root(), "models", "model.joblib")
MODEL_DIR = os.path.join(from_root(), "models")
METRICS_PATH = os.path.join(from_root(), "reports", "metrics.yaml")
REPORTS_DIR = os.path.join(from_root(), "reports")

MLFLOW_TRACKING_URI = "https://dagshub.com/Himanshu0518/Accent-Recognition.mlflow"
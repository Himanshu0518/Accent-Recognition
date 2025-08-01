# src/components/feature_extraction.py

import numpy as np
import pandas as pd
import librosa
import os
from tqdm import tqdm
from src.logger import logging
from src.constants import *
from src.utils.main_utils import save_dataframe
from src.components.data_augmentation import DataAugmentor
from src.exception import MyException
import sys 

class FeatureExtractor:
    def __init__(self, sr=SAMPLE_RATE):
        self.sr = sr
        self.frame_length = FRAME_LENGTH
        self.hop_length = HOP_LENGTH
        self.n_mfcc = MFCC_COUNT
        self.duration = DURATION 
        logging.info(f"[INIT] FeatureExtractorFromCSV initialized.")

    def extract_features(self, y):
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc,
                                     hop_length=self.hop_length, n_fft=self.frame_length)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length,
                                                 hop_length=self.hop_length)[0]
        rmse = librosa.feature.rms(y=y, frame_length=self.frame_length,
                                   hop_length=self.hop_length)[0]

        mfccs_mean = np.mean(mfccs, axis=1)
        zcr_mean = np.mean(zcr)
        rmse_mean = np.mean(rmse)

        return list(mfccs_mean) + [zcr_mean, rmse_mean]

 
    def initiate_featur_extraction_pipeline(self, data_dir, output_csv_path):
        features_list = []
        labels = []
        
        logging.info("Starting feature extraction from samples...")
        da = DataAugmentor(self.sr, self.duration) 
        try:
            for i in tqdm(range(NUM_FILES), desc="🔊 Augmenting"):
                for accent, label in LABELS.items():
                    file_path = os.path.join(data_dir, f"{accent}/{accent}_{i}.wav")

                    if not os.path.exists(file_path):
                        logging.warning(f"[SKIP] File not found: {file_path}")
                        continue

                    y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)
                    
                    y = da._pad_or_trim(y)
                  
                    augmented = da.augment_one(y)

                    if len(augmented) != 3:
                        logging.warning(f"[SKIP] Augmentation failed: {file_path}")
                        continue

                    # Original + 3 augmented versions
                    for aug in [y] + augmented:
                        features = self.extract_features(aug)
                        features_list.append(features)
                        labels.append(label)

        except Exception as e:
            raise MyException(e,sys)

        mfcc_columns = [f"mfcc_{i+1}" for i in range(self.n_mfcc)]
        all_columns = mfcc_columns + ["zcr", "rmse"]

        final_df = pd.DataFrame(features_list, columns=all_columns)
        final_df["label"] = labels

        save_dataframe(final_df, output_csv_path)
        logging.info(f"Feature extraction complete → {output_csv_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.initiate_featur_extraction_pipeline(
        data_dir=RAW_DATA_DIR,
        output_csv_path=os.path.join(INTERIM_DATA_DIR, "features.csv")
    )

# src/components/feature_extraction.py

import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.logger import logging
from from_root import from_root
from src.utils.main_utils import save_dataframe
from src.constants import *
from src.constants import * 

class FeatureExtractor:
    def __init__(self, sr=SAMPLE_RATE, duration=DURATION):
        self.sr = sr
        self.duration = duration
        self.FRAME_LENGTH = FRAME_LENGTH
        self.HOP_LENGTH = HOP_LENGTH
        logging.info(f"[INIT] FeatureExtractor initialized with sr={sr}, duration={duration}")

    def extract_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            logging.debug(f"[LOAD] Loaded file: {audio_path}")

            if len(y) < 2048:
                y = np.pad(y, (0, 2048 - len(y)), mode='constant')
                logging.debug(f"[PAD] Audio padded: {audio_path}")

            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=MFCC_COUNT,
                                         hop_length=self.HOP_LENGTH, n_fft=self.FRAME_LENGTH)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.FRAME_LENGTH,
                                                     hop_length=self.HOP_LENGTH)[0]
            rmse = librosa.feature.rms(y=y, frame_length=self.FRAME_LENGTH,
                                       hop_length=self.HOP_LENGTH)[0]

            mfccs_mean = np.mean(mfccs, axis=1)
            zcr_mean = np.mean(zcr)
            rmse_mean = np.mean(rmse)

            logging.debug(f"[FEATURES] Extracted from: {audio_path}")
            return list(mfccs_mean) + [zcr_mean, rmse_mean]

        except Exception as e:
            logging.error(f"[ERROR] Failed to extract features from {audio_path}: {e}")
            return None

    def initialize_feature_extractor(self, data_dir, output_csv):
        feature_list = []
        labels = []

        logging.info("Starting feature extraction for all audio files...")

        try:
            for i in tqdm(range(NUM_FILES), desc="🔍 Extracting features"):
                for accent, label in LABELS.items():
                    try:
                        file_path = os.path.join(from_root(), data_dir, f"{accent}/{accent}_{i}.wav")

                        if not os.path.exists(file_path):
                            logging.warning(f"[SKIP] File not found: {file_path}")
                            continue

                        features = self.extract_features(file_path)

                        if features:
                            feature_list.append(features)
                            labels.append(label)
                        else:
                            logging.warning(f"[SKIP] No features extracted from: {file_path}")

                    except Exception as inner_e:
                        logging.error(f"[ERROR] Exception while processing {file_path}: {inner_e}")
                        continue

            if not feature_list:
                logging.critical("No features extracted. Please check paths or data.")
                return

            mfcc_columns = [f"mfcc_{i+1}" for i in range(MFCC_COUNT)]
            columns = mfcc_columns + ["zcr", "rmse"]
            df = pd.DataFrame(feature_list, columns=columns)
            df["label"] = labels
            df = df.sample(frac=1).reset_index(drop=True)

            save_dataframe(df, output_csv)

        except Exception as outer_e:
            logging.critical(f"[CRITICAL] Feature extraction failed entirely: {outer_e}")


if __name__ == "__main__":
    data_dir = RAW_DATA_DIR
    output_csv = FEATURES_CSV

    logging.info("Starting feature extraction process.")
    feature_extractor = FeatureExtractor()
    feature_extractor.initialize_feature_extractor(data_dir, output_csv)
    logging.info("Feature extraction completed successfully.")

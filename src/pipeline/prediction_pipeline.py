# src/components/feature_extraction.py

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

class FeatureExtractor:
    def __init__(self, sr=22050, duration=10):
        self.sr = sr
        self.duration = duration

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, sr=self.sr, duration=self.duration)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        zcr = librosa.feature.zero_crossing_rate(y)
        rmse = librosa.feature.rms(y=y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        features = np.concatenate([
            np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
            [np.mean(zcr), np.mean(rmse), np.mean(centroid), np.mean(bandwidth)]
        ])
        return features

    def process_dataset(self, metadata_csv, data_dir, output_csv):
        df = pd.read_csv(metadata_csv)
        feature_list = []
        label_list = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                path = os.path.join(data_dir, row['filepath'])
                features = self.extract_features(path)
                feature_list.append(features)
                label_list.append(row['label'])
            except Exception as e:
                print(f"Error in file {row['filepath']}: {e}")

        X = pd.DataFrame(feature_list)
        X['label'] = label_list
        X.to_csv(output_csv, index=False)
        print(f"✅ Saved extracted features to {output_csv}")

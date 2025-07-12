# src/components/augment_and_save.py

import librosa
import numpy as np
from src.logger import logging
from src.constants import *


class DataAugmentor:
    def __init__(self, sr=SAMPLE_RATE, duration=DURATION):
        self.sr = sr
        self.duration = duration
        self.target_len = int(sr * duration)
        logging.info(f"[INIT] DataAugmentor initialized")

    def _pad_or_trim(self, y):
        if len(y) > self.target_len:
            return y[:self.target_len]
        if len(y) < self.target_len:
            return np.pad(y, (0, self.target_len - len(y)))
        return y

    def augment_one(self, y):
        try:
            y_stretch = librosa.effects.time_stretch(y, rate=1.2)
            y_shift = librosa.effects.pitch_shift(y, sr=self.sr, n_steps=2)
            y_noise = y + 0.005 * np.random.randn(len(y))

            return [
                self._pad_or_trim(y_stretch),
                 self._pad_or_trim(y_shift),
                self._pad_or_trim(y_noise),
            ]
        except Exception as e:
            logging.error(f"[AUGMENT] Failed: {e}")
            return []

  
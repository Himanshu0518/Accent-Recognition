# visualizer.py

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    return fig

def plot_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, ax=ax, x_axis='time', y_axis='mel')
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

def plot_zcr(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    ax.plot(zcr, color='purple')
    ax.set_title("Zero Crossing Rate")
    ax.set_ylabel("ZCR")
    ax.set_xlabel("Frames")
    return fig

def plot_rmse(y):
    fig, ax = plt.subplots(figsize=(10, 4))
    rmse = librosa.feature.rms(y=y)[0]
    ax.plot(rmse, color='green')
    ax.set_title("Root Mean Square Energy (RMSE)")
    ax.set_ylabel("Energy")
    ax.set_xlabel("Frames")
    return fig

def plot_feature_importance(importance_dict):
    fig, ax = plt.subplots(figsize=(10, 4))
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())
    
    sns.barplot(x=importances, y=features, ax=ax, palette="rocket")
    ax.set_title("Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    return fig

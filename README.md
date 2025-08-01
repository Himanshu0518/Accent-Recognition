# 🗣️ Accent Recognition using Machine Learning

Predict the speaker’s accent from audio using MFCCs, data augmentation, and classical machine learning techniques.

This project is built as an end-to-end MLOps pipeline that includes:

- 🎧 **Audio preprocessing** & **augmentation**
- 🔍 **Feature extraction** (MFCC, ZCR, RMSE)
- 🤖 **Model training** and **evaluation**
- 📦 **Data versioning via DVC**
- 🌐 **Streamlit app** for real-time accent prediction
- 🚀 **CI/CD Deployment** on **AWS EC2** using **GitHub Actions**
- 📊 **MLflow tracking**
- 📁 **Custom dataset**, collected and published on [Kaggle](https://www.kaggle.com/)

---

## 📥 Data Collection

All accent audio data was collected manually using Python-based scripts. The raw `.wav` files were then preprocessed and labeled appropriately. The finalized dataset has been published publicly on **Kaggle** for reproducibility and benchmarking.

---

## 📂 Project Structure

```bash
Accent-Recognition/
│
├── .dvc/                     # DVC internal files
├── dvc.yaml                  # DVC pipeline stages
├── dvc.lock                  # DVC lock file
├── data/                     # Dataset directories
│   ├── raw/                  # Raw .wav files
│   ├── interim/              # Extracted features (CSV)
│   ├── processed/            # Cleaned dataset
│   └── raw.dvc               # DVC tracking file
│
├── models/                   # Serialized ML models
│   └── model.joblib
│   └── preprocessor.joblib
│   └── labelencoder.joblib
│
├── src/                      # Source code
│   ├── constants.py
│   ├── logger.py
│   ├── from_root.py
│   ├── utils/
│   │   └── main_utils.py
│   ├── components/
│   │   ├── data_preprocessing.py
│   │   ├── data_augmentation.py
│   │   ├── feature_extraction.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── visualisation.py
│   └── pipeline/
│       ├── prediction_pipeline.py
│       └── trainning_pipeline.py
│
├── app.py                   # Streamlit web app
├── requirements.txt
├── setup.py
├── pyproject.toml
├── params.yaml
└── README.md
```
## 🎛 Features Extracted

🎵 MFCCs – Mel-Frequency Cepstral Coefficients

🎚️ ZCR – Zero Crossing Rate

💡 RMSE – Root Mean Square Energy


## 🔄 Audio Augmentation

To improve model robustness:

⏱️ Time Stretching

🎤 Pitch Shifting

📢 Noise Injection

These augmentations increase diversity and generalizability across speaker conditions.


## 🤖 Model Training

Models Compared:

RandomForestClassifier

LogisticRegression 


## Evaluation Metrics:

Accuracy

Confusion Matrix

Learning Curve

✅ Final Model: Logistic Regression

✅ Test Accuracy: ~96%


## 📦 Data Versioning with DVC

To ensure reproducibility and data tracking:

```bash
dvc init
dvc add data/raw
dvc repro
```
All stages — from raw data to feature extraction to final model — are versioned with DVC.


## 🌐 Web Application 

A simple and interactive flask app allows users to:

Upload a .wav audio file

Visualize waveform & spectrogram

Get real-time accent prediction


## 🧪 CI/CD Deployment (AWS EC2 + Docker + GitHub Actions)

Dockerized app is built and pushed to Amazon ECR

On every push to main, GitHub Actions triggers:

✅ CI: Docker Build → Push to ECR

✅ CD: SSH into EC2 → Pull & Run Docker container

✅ EC2 Port: 8000


### 📊 MLflow Tracking

To explore model parameters, metrics, and experiment runs and model registry :

[![MLflow Tracking](https://img.shields.io/badge/MLflow-enabled-blue)](https://dagshub.com/Himanshu0518/Accent-Recognition.mlflow/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D)


🧱 How to Run This Project

** 1. Install Dependencies **
```bash
pip install -r requirements.txt
```

** Run flask App **
```bash
python app.py
```

**🐳 Run via Docker**
(preferred)
You can directly pull and run the app using Docker:

🔻 Pull Docker Image
```bash
docker pull himanshu0518/accent-recognition-app:latest
```

Run the container

```bash
docker run -p 8000:8000 himanshu0518/accent-recognition-app:latest
```

## 🎯 Future Improvements

🎙️ Real-time accent detection from microphone input

🔁 Accent-to-Accent TTS conversion


## 👨‍💻 Author

**Himanshu Singh**  

Pre-final year B.Tech ECE @ IIIT Una

- 🐙 GitHub: [@Himanshu0518](https://github.com/Himanshu0518) 

- 📊 DagsHub: [@Himanshu0518](https://dagshub.com/Himanshu0518/Accent-Recognition)  

- 🐳 DockerHub: [himanshu0518/accent-detector](https://hub.docker.com/repository/docker/himanshu0518/accent-recognition-app/general)

⚠️ Disclaimer
The project is currently deployed using AWS Free Tier services. The services are removed after testing to save credits.
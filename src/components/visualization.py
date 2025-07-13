import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LearningCurveDisplay
import mlflow
from src.logger import logging
from from_root import from_root


def log_confusion_matrix(y_true, y_pred, labels=None):
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        path = os.path.join(from_root(), "reports", "confusion_matrix.png")
        plt.savefig(path)
        mlflow.log_artifact(path, artifact_path="plots")
        plt.close()
        logging.info("Confusion matrix saved and logged to MLflow.")
    except Exception as e:
        logging.warning(f"Could not log confusion matrix: {e}")


def log_learning_curve(model, train_csv, target_col="label"):
    try:
        df = pd.read_csv(train_csv)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        display = LearningCurveDisplay.from_estimator(
            model,
            X,
            y,
            train_sizes=np.linspace(0.1, 1.0, 5),
            cv=5,
            score_type="both",
            scoring="accuracy",
            n_jobs=-1
        )

        path = os.path.join(from_root(), "reports", "learning_curve.png")
        plt.savefig(path)
        mlflow.log_artifact(path, artifact_path="plots")
        plt.close()
        logging.info("Learning curve saved and logged to MLflow.")
    except Exception as e:
        logging.warning(f"Could not log learning curve: {e}")


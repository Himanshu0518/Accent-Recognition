from flask import Flask, request, jsonify, render_template
import tempfile
import librosa
import numpy as np
import os
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.pipeline.prediction_pipeline import AudioPredictor
from visualizer import (
    plot_waveform, plot_mel_spectrogram, plot_zcr, plot_rmse, plot_feature_importance
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

DAGSHUB_TRACKING_URL = "https://dagshub.com/Himanshu0518/Accent-Recognition.mlflow"

# Initialize predictor
predictor = AudioPredictor()

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img = BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight', dpi=100, facecolor='white')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    return plot_url

@app.route('/')
def index():
    return render_template('index.html', dagshub_url=DAGSHUB_TRACKING_URL)

@app.route('/upload', methods=['POST'])
def upload_audio():
    print("Upload route called")
    print("Request files:", request.files.keys())
    print("Request form:", request.form.keys())
    
    if 'audio' not in request.files:
        print("No 'audio' key in request.files")
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    print(f"File received: {file.filename}")
    
    if file.filename == '':
        print("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.wav'):
        print(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Please upload a .wav file'}), 400
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        print(f"File saved to: {tmp_path}")
        
        # Load audio for basic info
        y, sr = librosa.load(tmp_path, sr=None)
        duration = len(y) / sr
        
        print(f"Audio loaded: duration={duration}, sr={sr}")
        
        # Store file path in session or temporary storage
        # For production, consider using Redis or database
        app.config['TEMP_FILE_PATH'] = tmp_path
        
        return jsonify({
            'success': True,
            'duration': duration,
            'sample_rate': sr,
            'filename': file.filename
        })
    
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json
    viz_type = data.get('type', 'waveform')
    
    if 'TEMP_FILE_PATH' not in app.config:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    tmp_path = app.config['TEMP_FILE_PATH']
    
    try:
        y, sr = librosa.load(tmp_path, sr=None)
        
        if viz_type == 'waveform':
            fig = plot_waveform(y, sr)
        elif viz_type == 'mel_spectrogram':
            fig = plot_mel_spectrogram(y, sr)
        elif viz_type == 'zcr':
            fig = plot_zcr(y)
        elif viz_type == 'rmse':
            fig = plot_rmse(y)
        else:
            return jsonify({'error': 'Invalid visualization type'}), 400
        
        plot_url = fig_to_base64(fig)
        
        return jsonify({
            'success': True,
            'plot': plot_url
        })
    
    except Exception as e:
        return jsonify({'error': f'Error generating visualization: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    if 'TEMP_FILE_PATH' not in app.config:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    tmp_path = app.config['TEMP_FILE_PATH']
    
    try:
        result = predictor.predict(tmp_path)
        
        if isinstance(result, dict):
            # Get the predicted accent
            predicted_accent = max(result, key=result.get).capitalize()
            
            # Generate feature importance plot
            feature_importance = {
                "MFCC": 0.35,
                "ZCR": 0.2,
                "RMSE": 0.15,
                "Chroma": 0.1,
                "Spectral Centroid": 0.2
            }
            
            fig = plot_feature_importance(feature_importance)
            feature_plot = fig_to_base64(fig)
            
            return jsonify({
                'success': True,
                'predicted_accent': predicted_accent,
                'confidence': result,
                'feature_importance': feature_plot
            })
        else:
            return jsonify({
                'success': True,
                'predicted_accent': result.capitalize(),
                'confidence': None,
                'feature_importance': None
            })
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0", port=5000)
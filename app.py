from flask import Flask, request, jsonify, render_template, session
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
    plot_waveform, plot_mel_spectrogram, plot_zcr, plot_rmse
)
import logging
import uuid
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

DAGSHUB_TRACKING_URL = "https://dagshub.com/Himanshu0518/Accent-Recognition.mlflow"

# Global dictionary to store temporary file paths (use Redis in production)
temp_files = {}

# Initialize predictor with error handling
try:
    predictor = AudioPredictor()
    logging.info("AudioPredictor initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize AudioPredictor: {str(e)}")
    predictor = None

def cleanup_temp_file(file_path):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
            logging.info(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logging.error(f"Error cleaning up temp file {file_path}: {str(e)}")

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    try:
        img = BytesIO()
        fig.savefig(img, format='png', bbox_inches='tight', dpi=100, facecolor='white')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close(fig)
        return plot_url
    except Exception as e:
        logging.error(f"Error converting figure to base64: {str(e)}")
        plt.close(fig)
        raise

@app.route('/')
def index():
    return render_template('index.html', dagshub_url=DAGSHUB_TRACKING_URL)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'predictor_loaded': predictor is not None})

@app.route('/upload', methods=['POST'])
def upload_audio():
    logging.info("Upload route called")
    logging.info(f"Request files: {list(request.files.keys())}")
    logging.info(f"Request form: {list(request.form.keys())}")
    
    # Check if predictor is available
    if predictor is None:
        logging.error("AudioPredictor not initialized")
        return jsonify({'error': 'Audio prediction service not available'}), 503
    
    if 'audio' not in request.files:
        logging.error("No 'audio' key in request.files")
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    logging.info(f"File received: {file.filename}")
    
    if file.filename == '':
        logging.error("Empty filename")
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.wav'):
        logging.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Please upload a .wav file'}), 400
    
    try:
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        logging.info(f"File saved to: {tmp_path}")
        
        # Load audio for basic info with timeout
        try:
            y, sr = librosa.load(tmp_path, sr=None)
            duration = len(y) / sr
            
            logging.info(f"Audio loaded: duration={duration}, sr={sr}")
            
            # Validate audio file
            if duration > 30:  # 30 seconds limit
                cleanup_temp_file(tmp_path)
                return jsonify({'error': 'Audio file too long. Maximum 30 seconds allowed.'}), 400
            
            if duration < 0.5:  # Minimum 0.5 seconds
                cleanup_temp_file(tmp_path)
                return jsonify({'error': 'Audio file too short. Minimum 0.5 seconds required.'}), 400
            
        except Exception as e:
            cleanup_temp_file(tmp_path)
            logging.error(f"Error loading audio: {str(e)}")
            return jsonify({'error': f'Invalid audio file: {str(e)}'}), 400
        
        # Store file path with session ID
        temp_files[session_id] = tmp_path
        
        # Clean up old files (simple cleanup - use Redis with TTL in production)
        if len(temp_files) > 100:  # Keep only last 100 files
            old_session = next(iter(temp_files))
            old_path = temp_files.pop(old_session)
            cleanup_temp_file(old_path)
        
        return jsonify({
            'success': True,
            'duration': duration,
            'sample_rate': sr,
            'filename': file.filename,
            'session_id': session_id
        })
    
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        data = request.json
        viz_type = data.get('type', 'waveform')
        session_id = data.get('session_id')
        
        if not session_id or session_id not in temp_files:
            return jsonify({'error': 'No audio file uploaded or session expired'}), 400
        
        tmp_path = temp_files[session_id]
        
        if not os.path.exists(tmp_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
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
        logging.error(f"Error generating visualization: {str(e)}")
        return jsonify({'error': f'Error generating visualization: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if predictor is None:
            return jsonify({'error': 'Audio prediction service not available'}), 503
        
        if not session_id or session_id not in temp_files:
            return jsonify({'error': 'No audio file uploaded or session expired'}), 400
        
        tmp_path = temp_files[session_id]
        
        if not os.path.exists(tmp_path):
            return jsonify({'error': 'Audio file not found'}), 404
        
        logging.info(f"Starting prediction for session: {session_id}")
        
        result = predictor.predict(tmp_path)
        
        if isinstance(result, dict):
            # Get the predicted accent
            predicted_accent = max(result, key=result.get).capitalize()
            
            return jsonify({
                'success': True,
                'predicted_accent': predicted_accent,
                'confidence': result,
            })
        else:
            return jsonify({
                'success': True,
                'predicted_accent': result.capitalize(),
                'confidence': None,
                'feature_importance': None
            })
    
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up temporary files for a session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        
        if session_id and session_id in temp_files:
            tmp_path = temp_files.pop(session_id)
            cleanup_temp_file(tmp_path)
            
        return jsonify({'success': True})
    
    except Exception as e:
        logging.error(f"Error during cleanup: {str(e)}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    logging.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

# Cleanup on app shutdown
def cleanup_all_temp_files():
    """Clean up all temporary files on shutdown"""
    for session_id, file_path in temp_files.items():
        cleanup_temp_file(file_path)
    temp_files.clear()

def signal_handler(sig, frame):
    logging.info("Shutting down gracefully...")
    cleanup_all_temp_files()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    
    logging.info(f"Starting Flask app on port {port}")
    logging.info(f"Debug mode: {debug_mode}")
    
    try:
        app.run(host="0.0.0.0", port=port, debug=debug_mode)
    except Exception as e:
        logging.error(f"Failed to start Flask app: {str(e)}")
        sys.exit(1)
    finally:
        cleanup_all_temp_files()
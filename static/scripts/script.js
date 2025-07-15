// DOM Elements
const audioFileInput = document.getElementById('audioFile');
const audioPlayer = document.getElementById('audioPlayer');
const audioElement = document.getElementById('audioElement');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const visualizationSection = document.getElementById('visualizationSection');
const predictionSection = document.getElementById('predictionSection');
const vizType = document.getElementById('vizType');
const generateVizBtn = document.getElementById('generateViz');
const visualizationResult = document.getElementById('visualizationResult');
const predictBtn = document.getElementById('predictBtn');
const predictionResult = document.getElementById('predictionResult');
const loader = document.getElementById('loader');

// Global variables
let uploadedFile = null;
let isFileUploaded = false;

// Utility functions
function showLoader() {
    loader.classList.remove('hidden');
}

function hideLoader() {
    loader.classList.add('hidden');
}

function showError(message) {
    return `<div class="error"><i class="fas fa-exclamation-triangle"></i> ${message}</div>`;
}

function showSuccess(message) {
    return `<div class="success"><i class="fas fa-check-circle"></i> ${message}</div>`;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// File upload handling
audioFileInput.addEventListener('change', handleFileUpload);

// Drag and drop functionality
const fileLabel = document.querySelector('.file-label');

fileLabel.addEventListener('dragover', (e) => {
    e.preventDefault();
    fileLabel.style.borderColor = '#667eea';
    fileLabel.style.backgroundColor = '#edf2f7';
});

fileLabel.addEventListener('dragleave', (e) => {
    e.preventDefault();
    fileLabel.style.borderColor = '#cbd5e0';
    fileLabel.style.backgroundColor = '#f8fafc';
});

fileLabel.addEventListener('drop', (e) => {
    e.preventDefault();
    fileLabel.style.borderColor = '#cbd5e0';
    fileLabel.style.backgroundColor = '#f8fafc';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        // Create a new FileList-like object
        const dt = new DataTransfer();
        dt.items.add(files[0]);
        audioFileInput.files = dt.files;
        handleFileUpload();
    }
});

async function handleFileUpload() {
    const file = audioFileInput.files[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.wav')) {
        audioPlayer.classList.add('hidden');
        visualizationSection.classList.add('hidden');
        predictionSection.classList.add('hidden');
        
        // Show error in a visible location
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = showError('Please upload a .wav file');
        audioFileInput.parentNode.insertBefore(errorDiv, audioFileInput.nextSibling);
        
        // Remove error after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
        return;
    }

    // Validate file size (16MB limit)
    if (file.size > 16 * 1024 * 1024) {
        audioPlayer.classList.add('hidden');
        visualizationSection.classList.add('hidden');
        predictionSection.classList.add('hidden');
        
        // Show error in a visible location
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = showError('File size must be less than 16MB');
        audioFileInput.parentNode.insertBefore(errorDiv, audioFileInput.nextSibling);
        
        // Remove error after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
        return;
    }

    // Clear any previous errors
    const existingErrors = audioFileInput.parentNode.querySelectorAll('.error');
    existingErrors.forEach(error => error.remove());

    uploadedFile = file;
    showLoader();

    try {
        const formData = new FormData();
        formData.append('audio', file);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            // Show audio player
            audioPlayer.classList.remove('hidden');
            audioElement.src = URL.createObjectURL(file);
            fileName.textContent = file.name;
            fileSize.textContent = `${formatFileSize(file.size)} • ${formatDuration(data.duration)} • ${data.sample_rate} Hz`;

            // Show visualization and prediction sections
            visualizationSection.classList.remove('hidden');
            predictionSection.classList.remove('hidden');
            
            isFileUploaded = true;
            
            // Clear previous results
            visualizationResult.innerHTML = '';
            predictionResult.innerHTML = '';
            
            // Show success message
            const successDiv = document.createElement('div');
            successDiv.innerHTML = showSuccess('File uploaded successfully!');
            audioPlayer.insertBefore(successDiv, audioPlayer.firstChild);
            
            // Remove success message after 3 seconds
            setTimeout(() => {
                if (successDiv.parentNode) {
                    successDiv.parentNode.removeChild(successDiv);
                }
            }, 3000);
            
        } else {
            // Show error in a visible location
            const errorDiv = document.createElement('div');
            errorDiv.innerHTML = showError(data.error);
            audioFileInput.parentNode.insertBefore(errorDiv, audioFileInput.nextSibling);
            
            // Remove error after 5 seconds
            setTimeout(() => {
                if (errorDiv.parentNode) {
                    errorDiv.parentNode.removeChild(errorDiv);
                }
            }, 5000);
        }
    } catch (error) {
        // Show error in a visible location
        const errorDiv = document.createElement('div');
        errorDiv.innerHTML = showError('Upload failed. Please try again.');
        audioFileInput.parentNode.insertBefore(errorDiv, audioFileInput.nextSibling);
        
        // Remove error after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
        
        console.error('Upload error:', error);
    } finally {
        hideLoader();
    }
}

// Visualization handling
generateVizBtn.addEventListener('click', generateVisualization);

async function generateVisualization() {
    if (!isFileUploaded) {
        visualizationResult.innerHTML = showError('Please upload an audio file first');
        return;
    }

    const selectedVizType = vizType.value;
    showLoader();

    try {
        const response = await fetch('/visualize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ type: selectedVizType })
        });

        const data = await response.json();

        if (data.success) {
            visualizationResult.innerHTML = `
                <div class="viz-result">
                    <h3>${getVizTitle(selectedVizType)}</h3>
                    <img src="data:image/png;base64,${data.plot}" alt="${selectedVizType} visualization">
                </div>
            `;
        } else {
            visualizationResult.innerHTML = showError(data.error);
        }
    } catch (error) {
        visualizationResult.innerHTML = showError('Visualization failed. Please try again.');
        console.error('Visualization error:', error);
    } finally {
        hideLoader();
    }
}

function getVizTitle(vizType) {
    const titles = {
        'waveform': 'Audio Waveform',
        'mel_spectrogram': 'Mel Spectrogram',
        'zcr': 'Zero Crossing Rate (ZCR)',
        'rmse': 'Root Mean Square Energy (RMSE)'
    };
    return titles[vizType] || 'Visualization';
}

// Prediction handling
predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
    if (!isFileUploaded) {
        predictionResult.innerHTML = showError('Please upload an audio file first');
        return;
    }

    showLoader();

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({})
        });

        const data = await response.json();

        if (data.success) {
            let resultHTML = `
                <div class="prediction-success">
                    <h3><i class="fas fa-check-circle"></i> Predicted Accent: ${data.predicted_accent}</h3>
                </div>
            `;

            // Show confidence scores if available
            if (data.confidence) {
                resultHTML += `
                    <div class="confidence-chart">
                        <h4><i class="fas fa-chart-bar"></i> Prediction Confidence</h4>
                `;
                
                // Sort confidence scores in descending order
                const sortedConfidence = Object.entries(data.confidence)
                    .sort(([,a], [,b]) => b - a);
                
                sortedConfidence.forEach(([accent, confidence]) => {
                    const percentage = (confidence * 100).toFixed(1);
                    resultHTML += `
                        <div class="confidence-item">
                            <span>${accent.charAt(0).toUpperCase() + accent.slice(1)}</span>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${percentage}%"></div>
                            </div>
                            <span>${percentage}%</span>
                        </div>
                    `;
                });
                
                resultHTML += `</div>`;
            }

            // Show feature importance if available
            if (data.feature_importance) {
                resultHTML += `
                    <div class="feature-importance">
                        <h4><i class="fas fa-brain"></i> Feature Importance</h4>
                        <img src="data:image/png;base64,${data.feature_importance}" alt="Feature importance plot">
                    </div>
                `;
            }

            predictionResult.innerHTML = resultHTML;
        } else {
            predictionResult.innerHTML = showError(data.error);
        }
    } catch (error) {
        predictionResult.innerHTML = showError('Prediction failed. Please try again.');
        console.error('Prediction error:', error);
    } finally {
        hideLoader();
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + U for upload
    if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
        e.preventDefault();
        audioFileInput.click();
    }
    
    // Ctrl/Cmd + V for visualization (if file is uploaded)
    if ((e.ctrlKey || e.metaKey) && e.key === 'v' && isFileUploaded) {
        e.preventDefault();
        generateVisualization();
    }
    
    // Ctrl/Cmd + P for prediction (if file is uploaded)
    if ((e.ctrlKey || e.metaKey) && e.key === 'p' && isFileUploaded) {
        e.preventDefault();
        runPrediction();
    }
});

// Auto-scroll to sections when they become visible
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Add animation classes
const animatedElements = document.querySelectorAll('.upload-section, .visualization-section, .prediction-section, .info-card');
animatedElements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
});

// Initialize animations
document.addEventListener('DOMContentLoaded', () => {
    // Animate header on load
    const header = document.querySelector('.header');
    header.style.opacity = '0';
    header.style.transform = 'translateY(-20px)';
    
    setTimeout(() => {
        header.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
        header.style.opacity = '1';
        header.style.transform = 'translateY(0)';
    }, 100);
    
    // Animate upload section
    setTimeout(() => {
        const uploadSection = document.querySelector('.upload-section');
        uploadSection.style.opacity = '1';
        uploadSection.style.transform = 'translateY(0)';
    }, 300);
});

// Error handling for audio element
audioElement.addEventListener('error', (e) => {
    console.error('Audio playback error:', e);
    fileName.textContent = 'Error loading audio file';
    fileSize.textContent = 'Please try uploading again';
});

// Add tooltips for better UX
const tooltips = {
    'waveform': 'Shows the amplitude of the audio signal over time',
    'mel_spectrogram': 'Shows the frequency content of the audio in mel scale',
    'zcr': 'Shows how often the audio signal crosses zero amplitude',
    'rmse': 'Shows the energy/loudness of the audio signal over time'
};

vizType.addEventListener('change', (e) => {
    const selectedOption = e.target.value;
    if (tooltips[selectedOption]) {
        vizType.title = tooltips[selectedOption];
    }
});

// Add smooth scrolling for better navigation
function smoothScrollTo(element) {
    element.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Auto-scroll to sections when they become visible
audioFileInput.addEventListener('change', () => {
    setTimeout(() => {
        if (isFileUploaded) {
            smoothScrollTo(visualizationSection);
        }
    }, 1000);
});

generateVizBtn.addEventListener('click', () => {
    setTimeout(() => {
        if (visualizationResult.innerHTML) {
            smoothScrollTo(visualizationResult);
        }
    }, 1500);
});

predictBtn.addEventListener('click', () => {
    setTimeout(() => {
        if (predictionResult.innerHTML) {
            smoothScrollTo(predictionResult);
        }
    }, 1500);
});

// Add loading states to buttons
function setButtonLoading(button, isLoading) {
    if (isLoading) {
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    } else {
        button.disabled = false;
        // Restore original button text
        if (button === generateVizBtn) {
            button.innerHTML = '<i class="fas fa-eye"></i> Generate Visualization';
        } else if (button === predictBtn) {
            button.innerHTML = '<i class="fas fa-search"></i> Run Prediction';
        }
    }
}

// Update button states during operations
const originalGenerateViz = generateVisualization;
const originalRunPrediction = runPrediction;

generateVisualization = async function() {
    setButtonLoading(generateVizBtn, true);
    try {
        await originalGenerateViz.call(this);
    } finally {
        setButtonLoading(generateVizBtn, false);
    }
};

runPrediction = async function() {
    setButtonLoading(predictBtn, true);
    try {
        await originalRunPrediction.call(this);
    } finally {
        setButtonLoading(predictBtn, false);
    }
};
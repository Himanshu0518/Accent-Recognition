 let uploadedFile = null;

        document.getElementById('audioFile').addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                uploadedFile = e.target.files[0];
                showFileInfo(uploadedFile);
                document.getElementById('uploadBtn').style.display = 'inline-block';
            }
        });

        function showFileInfo(file) {
            const fileDetails = document.getElementById('fileDetails');
            const fileSize = (file.size / 1024 / 1024).toFixed(2);
            fileDetails.innerHTML = `
                <strong>Name:</strong> ${file.name}<br>
                <strong>Size:</strong> ${fileSize} MB<br>
                <strong>Type:</strong> ${file.type}
            `;
            document.getElementById('fileInfo').style.display = 'block';
        }

        function showError(message) {
            const errorMsg = document.getElementById('errorMsg');
            errorMsg.textContent = message;
            errorMsg.style.display = 'block';
            setTimeout(() => {
                errorMsg.style.display = 'none';
            }, 5000);
        }

        function showLoading(show) {
            document.getElementById('loading').style.display = show ? 'block' : 'none';
        }

        function uploadFile() {
            if (!uploadedFile) {
                showError('Please select an audio file first.');
                return;
            }
             console.log("Calling uploadFile with file:");
            const formData = new FormData();
            formData.append('audio', uploadedFile);

            showLoading(true);
            hideResults();
            
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    document.getElementById('predictBtn').style.display = 'inline-block';
                    document.getElementById('vizButtons').style.display = 'flex';
                    document.getElementById('uploadBtn').style.display = 'none';
                } else {
                    showError(data.error || 'Upload failed');
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Upload failed: ' + error.message);
            });
        }

        function predictAccent() {
            showLoading(true);
            
            fetch('/predict', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    displayResults(data);
                } else {
                    showError(data.error || 'Prediction failed');
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Prediction failed: ' + error.message);
            });
        }

        function showVisualization(type) {
            showLoading(true);
            
            fetch('/visualize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    type: type
                })
            })
            .then(response => response.json())
            .then(data => {
                showLoading(false);
                if (data.success) {
                    const vizImage = document.getElementById('vizImage');
                    const vizTitle = document.getElementById('vizTitle');
                    
                    vizImage.src = 'data:image/png;base64,' + data.plot;
                    vizTitle.textContent = type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ');
                    
                    document.getElementById('visualization').style.display = 'block';
                } else {
                    showError(data.error || 'Visualization failed');
                }
            })
            .catch(error => {
                showLoading(false);
                showError('Visualization failed: ' + error.message);
            });
        }

        function displayResults(data) {
            // Show predicted accent
            document.getElementById('predictedAccent').textContent = data.predicted_accent;
            
            document.getElementById('results').style.display = 'block';
        }

        function hideResults() {
            document.getElementById('results').style.display = 'none';
            document.getElementById('visualization').style.display = 'none';
        }

        function resetApp() {
            uploadedFile = null;
            document.getElementById('audioFile').value = '';
            document.getElementById('fileInfo').style.display = 'none';
            document.getElementById('uploadBtn').style.display = 'none';
            document.getElementById('predictBtn').style.display = 'none';
            document.getElementById('vizButtons').style.display = 'none';
            hideResults();
        }
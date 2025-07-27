FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files into the container
COPY . /app

# Install system-level dependencies required by librosa and pydub
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port used by gunicorn
EXPOSE 10000

# Use gunicorn to run the Flask app in production
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:10000", "app:app"]

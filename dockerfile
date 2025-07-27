FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Tell Render what port we plan to bind to (Render sets PORT env var)
EXPOSE 10000

# Entrypoint that runs the Flask app (must bind to $PORT in app.py)
CMD ["python", "app.py"]

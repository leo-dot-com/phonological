FROM python:3.9-slim-bullseye

WORKDIR /app

# Install system dependencies (including eSpeak‑NG)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    sox \
    libsox-fmt-mp3 \
    libsndfile1 \
    ffmpeg \
    libatlas-base-dev \
    gfortran \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p uploads/temp models static/phonemes

# Copy application code
COPY . .

# Generate phoneme audio files (will run during build)
RUN python generate_phoneme_audio.py

# Expose port
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

CMD ["python", "app.py"]

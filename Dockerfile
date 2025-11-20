FROM python:3.9-slim-bullseye

# Set working directory
WORKDIR /mfa_api

# Install system dependencies for MFA and audio processing
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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download MFA models (this can take a while)
RUN python -c "
import os
os.makedirs('/app/models', exist_ok=True)
print('MFA models will be downloaded on first run')
"

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads/temp

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "app.py"]

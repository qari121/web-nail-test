# Use Python 3.10 slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and TensorFlow
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY templates/ templates/
COPY nails_seg_s_yolov8_v1_float16.tflite .

# Create uploads directory
RUN mkdir -p uploads

# Expose port (RunPod will set PORT env var)
EXPOSE 8080

# Set environment variables for optimization
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4

# Run the application with gunicorn for production (better performance)
# Use gunicorn with 1 worker and multiple threads for model inference
# Model is loaded once per worker, so 1 worker prevents duplicate loading
# Multiple threads handle concurrent requests efficiently
# Use shell form to allow environment variable expansion
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 120 --worker-class gthread --preload app:app


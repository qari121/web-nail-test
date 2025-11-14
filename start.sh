#!/bin/bash

# RunPod startup script
# This script ensures the application starts correctly on RunPod

echo "Starting Nail Detection Application..."

# Check if model file exists
if [ ! -f "nails_seg_s_yolov8_v1_float16.tflite" ]; then
    echo "ERROR: Model file not found!"
    exit 1
fi

# Set port (RunPod provides PORT env var, default to 8080)
export PORT=${PORT:-8080}

# Check if gunicorn is available (preferred for production)
if command -v gunicorn &> /dev/null; then
    echo "Starting with Gunicorn on port $PORT..."
    gunicorn --bind 0.0.0.0:$PORT \
             --workers 1 \
             --threads 4 \
             --timeout 120 \
             --worker-class gthread \
             --access-logfile - \
             --error-logfile - \
             app:app
else
    echo "Gunicorn not found, using Flask development server..."
    echo "Starting Flask server on port $PORT..."
    python app.py
fi

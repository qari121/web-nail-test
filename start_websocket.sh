#!/bin/bash
# Production startup script for WebSocket version using Gunicorn + eventlet

# Install eventlet if not already installed
pip install eventlet gunicorn[eventlet] > /dev/null 2>&1

# Get port from environment or default to 8080
PORT=${PORT:-8080}

# Run with Gunicorn + eventlet for production WebSocket support
exec gunicorn \
    --bind 0.0.0.0:$PORT \
    --workers 1 \
    --worker-class eventlet \
    --worker-connections 1000 \
    --timeout 120 \
    --preload \
    app_websocket:app


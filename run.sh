#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Sedang membuat virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install essential dependencies
echo "Sedang menginstal dependensi (Flask, OpenCV, Numpy)..."
pip install Flask opencv-python-headless numpy

# Optional: Tensorflow (Can take minutes to install)
# pip install tensorflow

# Run the app
echo "Aplikasi siap dijalankan di http://0.0.0.0:5000"
python3 app.py

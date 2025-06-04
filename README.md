# Fingerprint Recognition System

This project implements a basic fingerprint recognition system using Python. It uses computer vision techniques to extract features from fingerprint images and machine learning to perform matching.

## Features
- Fingerprint image preprocessing
- Feature extraction using minutiae points
- Basic fingerprint matching
- Visualization of fingerprint features

## Setup
1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Place your fingerprint images in the `fingerprints` directory
2. Run the main script:
```bash
python fingerprint_model.py
```

## Project Structure
- `fingerprint_model.py`: Main script containing the fingerprint recognition model
- `utils.py`: Helper functions for image processing and feature extraction
- `requirements.txt`: Project dependencies
- `fingerprints/`: Directory for storing fingerprint images

## Note
This is a basic implementation and may not be suitable for production use. For production-grade fingerprint recognition, consider using specialized libraries or commercial solutions. 
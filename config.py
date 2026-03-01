"""Project paths and settings."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
ENCODINGS_FILE = os.path.join(KNOWN_FACES_DIR, "encodings.pkl")

# OpenCV DNN face detector (optional; uses Haar if these are missing)
MODELS_DIR = os.path.join(BASE_DIR, "models")
DNN_PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
DNN_CAFFEMODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

# Recognition (higher = more lenient; 0.6 default, 0.65–0.7 helps webcam match)
FACE_MATCH_TOLERANCE = 0.65

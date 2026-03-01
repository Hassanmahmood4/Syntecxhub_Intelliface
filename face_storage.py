"""Load and save known face encodings and names."""
import os
import pickle
from pathlib import Path

try:
    import face_recognition
except ImportError:
    face_recognition = None  # optional: need it only for add_person

import config


def _ensure_dir():
    Path(config.KNOWN_FACES_DIR).mkdir(parents=True, exist_ok=True)


def load_known_faces():
    """Load (names, encodings) from disk. Returns ([names], [encodings])."""
    if not os.path.isfile(config.ENCODINGS_FILE):
        return [], []
    with open(config.ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    names = data.get("names", [])
    encodings = data.get("encodings", [])
    return names, encodings


def save_known_faces(names, encodings):
    """Persist names and encodings to disk."""
    _ensure_dir()
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)


def add_person(name, image_rgb):
    """
    Detect face in image_rgb (RGB numpy array), compute encoding, append to known faces.
    Returns True if a face was found and added.
    """
    if face_recognition is None:
        return False
    boxes = face_recognition.face_locations(image_rgb)
    if not boxes:
        return False
    # Use first face
    encodings_list = face_recognition.face_encodings(image_rgb, [boxes[0]])
    if not encodings_list:
        return False
    encoding = encodings_list[0]
    names, encodings = load_known_faces()
    names.append(name)
    encodings.append(encoding)
    save_known_faces(names, encodings)
    return True

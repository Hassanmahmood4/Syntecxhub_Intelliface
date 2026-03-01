"""
Web UI for face detection & recognition.
Run: python3 app.py  →  open http://127.0.0.1:5000
"""
import base64
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

# Add project root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from face_storage import load_known_faces, add_person

# Import after path is set
from detect_and_recognize import get_face_detector, process_frame_for_web, _draw_boxes_on_frame

app = Flask(__name__)

# Load at startup and after each registration
KNOWN_NAMES, KNOWN_ENCODINGS = load_known_faces()
DETECTOR_TYPE, DETECTOR = get_face_detector()


def _reload_known_faces():
    global KNOWN_NAMES, KNOWN_ENCODINGS
    KNOWN_NAMES, KNOWN_ENCODINGS = load_known_faces()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    """Accept a frame (base64 JPEG), run detection+recognition, return annotated image."""
    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' (base64 JPEG)"}), 400

    try:
        raw = base64.b64decode(data["frame"].split(",")[-1] if "," in data["frame"] else data["frame"])
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Keep larger size so face is easier to detect (was 640)
    frame = np.ascontiguousarray(frame)

    # Process’s raw orientation so recognition matches registration
    try:
        # Get boxes and names on raw frame, then flip and redraw so labels stay readable
        out, boxes_with_names = process_frame_for_web(
            frame, DETECTOR_TYPE, DETECTOR, KNOWN_NAMES, KNOWN_ENCODINGS
        )
        out = cv2.flip(out, 1)
        w = out.shape[1]
        # Transform box x for flipped image: new_x = w - x - box_w
        flipped_boxes = [(w - x - bw, y, bw, bh, name) for (x, y, bw, bh, name) in boxes_with_names]
        _draw_boxes_on_frame(out, flipped_boxes)
        _, jpeg = cv2.imencode(".jpg", out)
        b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")
        return jsonify({"image": f"data:image/jpeg;base64,{b64}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/register", methods=["POST"])
def api_register():
    """Register a new face: JSON { frame: base64 JPEG, name: string }. Returns { ok } or { error }."""
    data = request.get_json()
    if not data or "frame" not in data or "name" not in data:
        return jsonify({"error": "Missing 'frame' or 'name'"}), 400

    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Name cannot be empty"}), 400

    try:
        raw = base64.b64decode(data["frame"].split(",")[-1] if "," in data["frame"] else data["frame"])
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not add_person(name, frame_rgb):
        return jsonify({"error": "No face detected. Look at the camera and try again."}), 400

    _reload_known_faces()
    return jsonify({"ok": True, "name": name})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

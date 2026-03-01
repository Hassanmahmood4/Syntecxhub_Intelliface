"""
Add new people to the face recognition database.
Captures from webcam (or optional image path), detects one face, and registers it.
"""
import argparse
import sys
from pathlib import Path

import cv2

try:
    import face_recognition
except ImportError:
    face_recognition = None

import config
from face_storage import add_person


def _face_boxes_for_preview(frame_bgr):
    """Face boxes for preview: use face_recognition if available, else OpenCV Haar."""
    if face_recognition is not None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        return [(left, top, right - left, bottom - top) for (top, right, bottom, left) in boxes]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))


def capture_from_webcam():
    """Capture one frame from webcam. Returns BGR frame or None."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.", file=sys.stderr)
        return None
    print("Press SPACE to capture, Q to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                return None
            # Show face detection preview
            for (x, y, w, h) in _face_boxes_for_preview(frame):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame, "SPACE=capture, Q=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            cv2.imshow("Register Face - Capture", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None
            if key == ord(" "):
                return frame
    finally:
        cap.release()
        cv2.destroyAllWindows()


def register_from_image(image_path: str, name: str) -> bool:
    """Register a face from an image file. Returns True if a face was found and added."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Could not read image: {image_path}", file=sys.stderr)
        return False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return add_person(name, rgb)


def register_from_webcam(name: str) -> bool:
    """Capture from webcam and register. Returns True if a face was added."""
    frame = capture_from_webcam()
    if frame is None:
        return False
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return add_person(name, rgb)


def main():
    parser = argparse.ArgumentParser(
        description="Capture and register a new face for recognition."
    )
    parser.add_argument("name", help="Display name for this person")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Register from image file instead of webcam",
    )
    args = parser.parse_args()

    if face_recognition is None:
        print(
            "Face registration requires 'face_recognition'. Install CMake (e.g. brew install cmake), then: pip install face_recognition",
            file=sys.stderr,
        )
        sys.exit(1)

    name = args.name.strip()
    if not name:
        print("Please provide a non-empty name.", file=sys.stderr)
        sys.exit(1)

    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)
        ok = register_from_image(str(path), name)
    else:
        ok = register_from_webcam(name)

    if ok:
        print(f"Registered '{name}' successfully.")
    else:
        print("No face detected. Please try again with a clear front-facing face.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Face detection (OpenCV) + recognition (face_recognition).
Draws bounding boxes and labels on frames; handles multiple faces.
Supports image file, video file, or webcam.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import face_recognition
except ImportError:
    face_recognition = None  # detection-only mode if not installed

import config
from face_storage import load_known_faces


def get_face_detector():
    """Use OpenCV DNN if model files exist, else Haar cascade."""
    if Path(config.DNN_PROTOTXT).exists() and Path(config.DNN_CAFFEMODEL).exists():
        net = cv2.dnn.readNetFromCaffe(config.DNN_PROTOTXT, config.DNN_CAFFEMODEL)
        return ("dnn", net)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return ("haar", cascade)


def detect_faces_opencv(frame_bgr, detector_type, detector):
    """Return list of (x, y, w, h) bounding boxes."""
    if detector_type == "haar":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Stricter to cut false positives: higher minNeighbors, sensible minSize
        boxes = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return [tuple(map(int, (x, y, w, h))) for (x, y, w, h) in boxes]

    # DNN
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    detector.setInput(blob)
    detections = detector.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        if w > 0 and h > 0:
            boxes.append((x, y, w, h))
    return boxes


def recognize_face(frame_rgb, box_xywh, known_encodings, known_names):
    """Get name for one face; box in (top, right, bottom, left) for face_recognition."""
    x, y, w, h = box_xywh
    top, right, bottom, left = y, x + w, y + h, x
    encodings = face_recognition.face_encodings(
        frame_rgb, [(top, right, bottom, left)]
    )
    if not encodings:
        return "Unknown"
    face_encoding = encodings[0]
    if not known_encodings:
        return "Unknown"
    matches = face_recognition.compare_faces(
        known_encodings, face_encoding, tolerance=config.FACE_MATCH_TOLERANCE
    )
    distances = face_recognition.face_distance(known_encodings, face_encoding)
    best = np.argmin(distances)
    return known_names[best] if matches[best] else "Unknown"


def process_frame(frame_bgr, detector_type, detector, known_names, known_encodings):
    """Detect faces, recognize, draw boxes and labels. Returns frame_bgr."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Prefer face_recognition detector for webcam (more reliable); fall back to OpenCV
    boxes = []
    if face_recognition is not None:
        raw_boxes = face_recognition.face_locations(frame_rgb, model="hog")
        boxes = [(left, top, right - left, bottom - top) for (top, right, bottom, left) in raw_boxes]
    if len(boxes) == 0:
        boxes = detect_faces_opencv(frame_bgr, detector_type, detector)

    # Scale line thickness and font with image size so boxes stay visible
    img_h, img_w = frame_bgr.shape[:2]
    thickness = max(3, min(8, img_w // 120))
    font_scale = max(0.5, min(1.4, img_w / 350))

    for (x, y, bw, bh) in boxes:
        if face_recognition is None:
            name = "Unknown"
        else:
            name = recognize_face(
                frame_rgb, (x, y, bw, bh), known_encodings, known_names
            )
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), color, thickness)
        label = name
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        cv2.rectangle(frame_bgr, (x, y - th - 14), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame_bgr, label, (x + 2, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, thickness - 1)
        )
    return frame_bgr


def _draw_boxes_on_frame(frame_bgr, boxes_with_names):
    """Draw boxes and labels on frame_bgr. boxes_with_names = [(x, y, bw, bh, name), ...]."""
    _, w = frame_bgr.shape[:2]
    thickness = max(3, min(8, w // 120))
    font_scale = max(0.5, min(1.4, w / 350))
    if not boxes_with_names:
        h, w = frame_bgr.shape[:2]
        msg = "No face detected"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(1.2, w / 400)
        thick = max(2, int(scale))
        (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.rectangle(frame_bgr, (cx - 10, cy - th - 10), (cx + tw + 10, cy + 10), (0, 0, 0), -1)
        cv2.putText(frame_bgr, msg, (cx, cy), font, scale, (0, 0, 0), thick + 2)
        cv2.putText(frame_bgr, msg, (cx, cy), font, scale, (255, 255, 255), thick)
    for (x, y, bw, bh, name) in boxes_with_names:
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame_bgr, (x, y), (x + bw, y + bh), color, thickness)
        (tw, th), _ = cv2.getTextSize(
            name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, max(1, thickness - 1)
        )
        cv2.rectangle(frame_bgr, (x, y - th - 14), (x + tw + 4, y), color, -1)
        cv2.putText(
            frame_bgr, name, (x + 2, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), max(1, thickness - 1)
        )


def _iou(box1, box2):
    """Rough overlap: 1 if boxes overlap a lot, 0 if not."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi = max(x1, x2)
    yi = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    if xi2 <= xi or yi2 <= yi:
        return 0.0
    inter = (xi2 - xi) * (yi2 - yi)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter / min(area1, area2) if min(area1, area2) > 0 else 0.0


def process_frame_for_web(frame_bgr, detector_type, detector, known_names, known_encodings):
    """
    Detect and recognize, return (frame_bgr, [(x,y,w,h,name), ...]).
    Uses both face_recognition and OpenCV detectors and merges results so at least one finds the face.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes_fr = []
    # Run OpenCV Haar first (aggressive settings) then face_recognition; merge results
    boxes_haar = detect_faces_opencv(frame_bgr, detector_type, detector)
    boxes_fr = []
    if face_recognition is not None:
        raw = face_recognition.face_locations(
            frame_rgb, model="hog", number_of_times_to_upsample=2
        )
        boxes_fr = [(left, top, right - left, bottom - top) for (top, right, bottom, left) in raw]
    boxes = list(boxes_haar)
    for b in boxes_fr:
        if not any(_iou(b, existing) > 0.5 for existing in boxes):
            boxes.append(b)

    # Keep only boxes where we can get a valid face encoding (drops false positives)
    boxes_with_names = []
    for (x, y, bw, bh) in boxes:
        if face_recognition is None:
            name = "Unknown"
            boxes_with_names.append((x, y, bw, bh, name))
        else:
            top, right, bottom, left = y, x + bw, y + bh, x
            encodings = face_recognition.face_encodings(frame_rgb, [(top, right, bottom, left)])
            if not encodings:
                continue  # Not a real face, skip this box
            name = recognize_face(frame_rgb, (x, y, bw, bh), known_encodings, known_names)
            boxes_with_names.append((x, y, bw, bh, name))

    return frame_bgr, boxes_with_names


def run_image(path: str, detector_type, detector, known_names, known_encodings):
    """Process single image and show/save."""
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read image: {path}", file=sys.stderr)
        return
    out = process_frame(frame, detector_type, detector, known_names, known_encodings)
    out_path = path.replace(".", "_out.")
    if out_path == path:
        out_path = path + "_out.jpg"
    cv2.imwrite(out_path, out)
    print(f"Saved: {out_path}")
    cv2.imshow("Face Detection & Recognition", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(source, detector_type, detector, known_names, known_encodings):
    """source: path or 0 for webcam."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}", file=sys.stderr)
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out = process_frame(
                frame, detector_type, detector, known_names, known_encodings
            )
            cv2.imshow("Face Detection & Recognition", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Face detection (OpenCV) + recognition; draw boxes and labels."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Image path, video path, or omit for webcam",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="For image: only save output file, do not show window",
    )
    args = parser.parse_args()

    known_names, known_encodings = load_known_faces()
    detector_type, detector = get_face_detector()

    if args.input is None:
        run_video(0, detector_type, detector, known_names, known_encodings)
        return
    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)
    # Heuristic: image vs video by extension
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if path.suffix.lower() in image_ext:
        if args.no_show:
            frame = cv2.imread(str(path))
            if frame is not None:
                out = process_frame(
                    frame, detector_type, detector, known_names, known_encodings
                )
                out_path = str(path).replace(".", "_out.", 1)
                if out_path == str(path):
                    out_path = str(path) + "_out.jpg"
                cv2.imwrite(out_path, out)
                print(f"Saved: {out_path}")
        else:
            run_image(
                str(path), detector_type, detector, known_names, known_encodings
            )
    else:
        run_video(
            str(path), detector_type, detector, known_names, known_encodings
        )


if __name__ == "__main__":
    main()

# Project 1: Face Detection & Recognition

Face detection using **OpenCV** (Haar cascades or DNN) and face recognition using the **face_recognition** library (face embeddings). Supports multiple faces per frame with bounding boxes and labels, and scripts to register new people.

## Requirements

- Python 3.8+
- OpenCV, face_recognition, numpy

## Setup

```bash
cd Intelliface
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use **`python3`** (or the venv’s `python`) to run scripts—macOS often has no `python` command.

**Note:** `face_recognition` depends on `dlib`, which needs CMake to build. If you don't have system CMake:
  ```bash
  pip install cmake
  # Then put pip’s cmake on PATH and install (macOS/Linux):
  export PATH="$(python3 -c "import cmake, os; print(os.path.join(os.path.dirname(cmake.__file__), 'data', 'bin'))"):$PATH"
  pip install face_recognition
  ```

## Usage

### 1. Register new people

Add someone so they can be recognized later. You can use the **webcam** or an **image file**.

**From webcam** (press SPACE to capture, Q to quit):

```bash
python3 register_face.py "Alice"
```

**From an image file:**

```bash
python3 register_face.py "Bob" --image path/to/photo.jpg
```

Registered faces are stored in `known_faces/encodings.pkl`.

### 2. Detect and recognize faces

- **Web UI (localhost):**  
  ```bash
  python3 app.py
  ```
  Open **http://127.0.0.1:5000** in your browser. Click “Start camera” for live detection and recognition in the browser.

- **Webcam (live, desktop):**  
  ```bash
  python3 detect_and_recognize.py
  ```
  Press **Q** to quit.

- **Video file:**  
  ```bash
  python3 detect_and_recognize.py path/to/video.mp4
  ```

- **Image file:**  
  ```bash
  python3 detect_and_recognize.py path/to/image.jpg
  ```
  Output is saved as `path/to/image_out.jpg` and a window shows the result. Use `--no-show` to only save the file without opening a window.

## Features

| Requirement | Implementation |
|-------------|----------------|
| **Face detection** | OpenCV (Haar cascade by default; optional DNN if model files are in `models/`) |
| **Face recognition** | `face_recognition` library (128-D face embeddings), match against registered encodings |
| **Bounding boxes & labels** | Drawn on each frame; green = recognized, red = unknown |
| **Multiple faces** | All faces in each frame are detected, recognized, and labeled |
| **Add new people** | `register_face.py` — capture from webcam or register from image |

## Optional: OpenCV DNN face detector

For more accurate detection you can use the OpenCV DNN face detector:

1. Create a `models` folder in the project root.
2. Download:
   - [deploy.prototxt](https://raw.githubusercontent.com/opencv/opencv/3.4.0/samples/dnn/face_detector/deploy.prototxt) → save as `models/deploy.prototxt`
   - **res10_300x300_ssd_iter_140000.caffemodel** (Caffe weights) → save as `models/res10_300x300_ssd_iter_140000.caffemodel`  
     (Search for "opencv face detector caffemodel" for a download link.)
3. Place both files in `models/`. The script will automatically use DNN when both are present; otherwise it uses the built-in Haar cascade.

## Project structure

```
Intelliface/
├── app.py                 # Web UI (Flask) — run and open http://127.0.0.1:5000
├── config.py              # Paths and settings
├── face_storage.py        # Load/save known face encodings
├── detect_and_recognize.py  # Main pipeline: detect → recognize → draw
├── register_face.py       # Add new people (webcam or image)
├── requirements.txt
├── templates/             # index.html for web UI
├── known_faces/           # encodings.pkl (registered faces)
└── models/                # Optional: DNN prototxt + caffemodel
```

## Example

1. Register yourself:  
   `python3 register_face.py "Your Name"`
2. Run recognition on webcam:  
   `python3 detect_and_recognize.py`
3. Your face should appear with a green box and your name; others as "Unknown" in red.

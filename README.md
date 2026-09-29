# Face Detection with OpenCV

Real-time face detection from a webcam (or a video file) using OpenCV's Haar Cascade classifier.
Draws a box around each face and shows a live face count and FPS.

## Requirements

- Python 3.8+
- A webcam (or any video file)

## Setup

```bash
git clone https://github.com/Akhiljoshi03/face-detection-opencv.git
cd face-detection-opencv

python -m venv venv
# Windows:  venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
python main.py                    # default webcam
python main.py --camera 1         # a different webcam
python main.py --video clip.mp4   # a video file
```

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--camera` | `0` | Webcam index |
| `--video` | none | Video file path (overrides `--camera`) |
| `--scale-factor` | `1.2` | Smaller = more accurate, slower |
| `--min-neighbors` | `5` | Higher = fewer false detections |
| `--min-size` | `60` | Smallest face size in pixels |

### Controls

- `ESC` or `q`: quit
- `s`: save a screenshot

## How it works

1. Reads frames from the webcam with `cv2.VideoCapture`.
2. Converts each frame to grayscale and equalizes the histogram.
3. Runs `detectMultiScale` with the Haar Cascade bundled inside OpenCV.
4. Draws rectangles on detected faces and displays the frame.

## Troubleshooting

- **"could not open camera"**: close other apps using the webcam, or try `--camera 1`.
- **On macOS**: allow camera access for your terminal in System Settings, Privacy & Security.
- **Too many false detections**: raise `--min-neighbors` or `--min-size`.
- **Faces missed**: lower `--scale-factor` (e.g. `1.1`) or `--min-size`.

## Author

Akhil Joshi

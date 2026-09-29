"""Real-time face detection using OpenCV and a Haar Cascade classifier.

Usage:
    python main.py                    # default webcam
    python main.py --camera 1         # another webcam
    python main.py --video clip.mp4   # a video file instead of a webcam

Controls:
    ESC or q  - quit
    s         - save the current frame as a screenshot
"""

import argparse
import sys
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time face detection with OpenCV")
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="path to a video file (overrides --camera)")
    parser.add_argument("--scale-factor", type=float, default=1.2,
                        help="detectMultiScale scaleFactor; smaller = more accurate but slower (default: 1.2)")
    parser.add_argument("--min-neighbors", type=int, default=5,
                        help="higher = fewer false positives (default: 5)")
    parser.add_argument("--min-size", type=int, default=60,
                        help="smallest face size in pixels (default: 60)")
    return parser.parse_args()


def load_cascade():
    """Load the Haar cascade that ships with OpenCV, so no extra XML file is needed."""
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        sys.exit(f"Error: could not load cascade file at {cascade_path}")
    return cascade


def open_source(args):
    source = args.video if args.video else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        what = f"video file '{args.video}'" if args.video else f"camera {args.camera}"
        sys.exit(f"Error: could not open {what}. Check the path/index and that no other app is using the camera.")
    return cap


def main():
    args = parse_args()
    face_cascade = load_cascade()
    cap = open_source(args)

    prev_time = time.time()
    fps = 0.0
    shot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("No more frames (end of video or camera disconnected).")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # improves detection in uneven lighting

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale_factor,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_size, args.min_size),
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # FPS (smoothed) and face count overlay
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps:.0f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Detection (ESC/q to quit, s to save)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                filename = f"screenshot_{shot_count}.png"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
                shot_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

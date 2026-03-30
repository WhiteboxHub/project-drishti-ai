# Project Drishti AI

Real-time AI webcam proxy built with OpenCV, MediaPipe Face Mesh, NumPy, and PyVirtualCam.

## Features

- Webcam capture with configurable resolution and FPS
- MediaPipe face mesh tracking with iris landmarks enabled
- Head pose estimation using `cv2.solvePnP`
- Lightweight gaze estimation from eye and iris geometry
- CPU-friendly correction pipeline with EMA smoothing
- Face and eye ROI warping with seamless blending
- Virtual camera output for video conferencing apps
- Debug overlays and local preview window

## Project Layout

```text
.
├── main.py
├── config.py
├── capture/
├── face/
├── pose/
├── gaze/
├── correction/
├── render/
├── output/
├── utils/
└── requirements.txt
```

## Setup

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure your OS has a virtual camera backend available.

- macOS / Windows: installing OBS often provides a compatible virtual camera backend.
- Linux: `v4l2loopback` is commonly required.

## Run

Start the proxy:

```bash
python main.py --show-window
```

Useful options:

- `--camera-index 0`
- `--width 1280 --height 720`
- `--fps 30`
- `--head-correction 0.7`
- `--gaze-correction 0.5`
- `--smoothing 0.8`
- `--debug`
- `--disable-virtual-cam`

## Use In Zoom / Teams / Meet

1. Launch the app with `python main.py`.
2. Open Zoom, Microsoft Teams, or Google Meet.
3. Select the PyVirtualCam / OBS virtual camera device as your webcam.
4. Fine-tune correction strength with CLI arguments if the effect is too subtle or too aggressive.

## Notes

- The system is designed to stay lightweight and CPU-friendly.
- Head pose normalization and gaze alignment are intentionally conservative to reduce artifacts.
- For the most reliable eye tracking, keep your face well lit and centered in frame.

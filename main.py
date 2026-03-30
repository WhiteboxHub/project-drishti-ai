from __future__ import annotations

import argparse
import signal
import sys
import time
from typing import Optional

from capture.camera import CameraCapture
from config import (
    CAMERA_INDEX,
    DEBUG,
    FPS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    GAZE_CORRECTION,
    HEAD_CORRECTION,
    SMOOTHING,
)
from correction.corrector import FrameCorrector
from face.landmarks import FaceLandmarkDetector
from gaze.gaze_estimator import GazeEstimator
from output.virtual_cam import VirtualCameraOutput
from pose.head_pose import HeadPoseEstimator
from render.renderer import FrameRenderer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time AI webcam proxy with head pose and gaze correction."
    )
    parser.add_argument("--camera-index", type=int, default=CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--head-correction", type=float, default=HEAD_CORRECTION)
    parser.add_argument("--gaze-correction", type=float, default=GAZE_CORRECTION)
    parser.add_argument("--smoothing", type=float, default=SMOOTHING)
    parser.add_argument("--debug", action="store_true", default=DEBUG)
    parser.add_argument(
        "--show-window",
        action="store_true",
        help="Display the processed frame locally for debugging.",
    )
    parser.add_argument(
        "--disable-virtual-cam",
        action="store_true",
        help="Process frames without publishing to a virtual camera.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    should_stop = False

    def handle_signal(_signum: int, _frame: Optional[object]) -> None:
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    capture = CameraCapture(
        index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    detector = FaceLandmarkDetector()
    pose_estimator = HeadPoseEstimator(frame_width=args.width, frame_height=args.height)
    gaze_estimator = GazeEstimator()
    corrector = FrameCorrector(
        head_strength=args.head_correction,
        gaze_strength=args.gaze_correction,
        smoothing=args.smoothing,
    )
    renderer = FrameRenderer(debug=args.debug)
    virtual_cam = None

    try:
        capture.open()
        if not args.disable_virtual_cam:
            virtual_cam = VirtualCameraOutput(
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
            virtual_cam.open()

        while not should_stop:
            frame = capture.read()
            if frame is None:
                continue

            timestamp = time.perf_counter()
            try:
                face = detector.detect(frame)
                if face is None:
                    corrected_frame = renderer.render_passthrough(frame, debug_text="No face")
                else:
                    pose = pose_estimator.estimate(face)
                    gaze = gaze_estimator.estimate(face)
                    correction = corrector.compute(face, pose, gaze)
                    corrected_frame = renderer.render(
                        frame=frame,
                        face=face,
                        pose=pose,
                        gaze=gaze,
                        correction=correction,
                    )
            except Exception as frame_exc:
                corrected_frame = renderer.render_passthrough(
                    frame,
                    debug_text=f"Frame fallback: {type(frame_exc).__name__}",
                )

            if virtual_cam is not None:
                virtual_cam.send(corrected_frame)

            if args.show_window:
                renderer.show_preview(corrected_frame, timestamp)
                if renderer.poll_exit():
                    break
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        return 1
    finally:
        detector.close()
        capture.close()
        if virtual_cam is not None:
            virtual_cam.close()
        renderer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from correction.corrector import FrameCorrection
from face.mesh_utils import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    FaceDetectionResult,
)
from gaze.gaze_estimator import GazeEstimate
from pose.head_pose import HeadPose


class FrameRenderer:
    def __init__(self, debug: bool = False) -> None:
        self._debug = debug
        self._last_fps_timestamp = time.perf_counter()
        self._last_fps_frame_count = 0
        self._fps = 0.0

    def render(
        self,
        frame: np.ndarray,
        face: FaceDetectionResult,
        pose: HeadPose,
        gaze: GazeEstimate,
        correction: FrameCorrection,
    ) -> np.ndarray:
        corrected = frame.copy()
        corrected = self._apply_face_shift(corrected, face, correction)
        corrected = self._apply_eye_shift(corrected, face, LEFT_EYE_INDICES, correction)
        corrected = self._apply_eye_shift(corrected, face, RIGHT_EYE_INDICES, correction)

        if self._debug:
            self._draw_debug(corrected, face, pose, gaze, correction)

        return corrected

    def render_passthrough(self, frame: np.ndarray, debug_text: Optional[str] = None) -> np.ndarray:
        output = frame.copy()
        if self._debug and debug_text:
            cv2.putText(
                output,
                debug_text,
                (24, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return output

    def _apply_face_shift(
        self,
        frame: np.ndarray,
        face: FaceDetectionResult,
        correction: FrameCorrection,
    ) -> np.ndarray:
        x1, y1, x2, y2 = face.bounding_rect(pad=24)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        center = ((x2 - x1) / 2.0, (y2 - y1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, correction.roll_correction, 1.0)
        matrix[0, 2] += correction.face_shift_x
        matrix[1, 2] += correction.face_shift_y

        warped = cv2.warpAffine(
            roi,
            matrix,
            (roi.shape[1], roi.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, face.face_hull(), 255)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)
        local_mask = (mask[y1:y2, x1:x2].astype(np.float32) / 255.0)[..., None]

        blended = roi.astype(np.float32) * (1.0 - local_mask) + warped.astype(np.float32) * local_mask
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def _apply_eye_shift(
        self,
        frame: np.ndarray,
        face: FaceDetectionResult,
        eye_indices: list[int],
        correction: FrameCorrection,
    ) -> np.ndarray:
        x1, y1, x2, y2 = face.eye_box(eye_indices)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or x2 <= x1 or y2 <= y1:
            return frame

        matrix = np.array(
            [[1.0, 0.0, correction.gaze_shift_x], [0.0, 1.0, correction.gaze_shift_y]],
            dtype=np.float32,
        )
        warped = cv2.warpAffine(
            roi,
            matrix,
            (roi.shape[1], roi.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
        eye_points = face.points(eye_indices)[:, :2].copy()
        eye_points[:, 0] -= x1
        eye_points[:, 1] -= y1
        cv2.fillConvexPoly(mask, cv2.convexHull(eye_points.astype(np.int32)), 255)
        mask = cv2.GaussianBlur(mask, (17, 17), 0)
        mask_f = (mask.astype(np.float32) / 255.0)[..., None]

        blended = roi.astype(np.float32) * (1.0 - mask_f) + warped.astype(np.float32) * mask_f
        frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        return frame

    def _draw_debug(
        self,
        frame: np.ndarray,
        face: FaceDetectionResult,
        pose: HeadPose,
        gaze: GazeEstimate,
        correction: FrameCorrection,
    ) -> None:
        for x, y, _ in face.landmarks[::8]:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), -1)

        lines = [
            f"yaw={pose.yaw:.1f} pitch={pose.pitch:.1f} roll={pose.roll:.1f}",
            f"gaze_h={gaze.horizontal:.3f} gaze_v={gaze.vertical:.3f}",
            f"shift=({correction.face_shift_x:.1f}, {correction.face_shift_y:.1f})",
        ]
        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, 30 + idx * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def show_preview(self, frame: np.ndarray, timestamp: float) -> None:
        self._last_fps_frame_count += 1
        elapsed = timestamp - self._last_fps_timestamp
        if elapsed >= 1.0:
            self._fps = self._last_fps_frame_count / elapsed
            self._last_fps_timestamp = timestamp
            self._last_fps_frame_count = 0

        preview = frame.copy()
        cv2.putText(
            preview,
            f"preview_fps={self._fps:.1f}",
            (20, preview.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("AI Webcam Proxy", preview)

    def poll_exit(self) -> bool:
        return (cv2.waitKey(1) & 0xFF) == ord("q")

    def close(self) -> None:
        cv2.destroyAllWindows()


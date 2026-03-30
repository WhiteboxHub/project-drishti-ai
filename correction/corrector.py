from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import MAX_PITCH_DEGREES, MAX_YAW_DEGREES
from face.mesh_utils import FaceDetectionResult
from gaze.gaze_estimator import GazeEstimate
from pose.head_pose import HeadPose
from utils.math_utils import clamp
from utils.smoothing import ExponentialSmoother


@dataclass
class FrameCorrection:
    face_shift_x: float
    face_shift_y: float
    roll_correction: float
    gaze_shift_x: float
    gaze_shift_y: float
    pose_yaw: float
    pose_pitch: float
    pose_roll: float


class FrameCorrector:
    def __init__(self, head_strength: float, gaze_strength: float, smoothing: float) -> None:
        self._head_strength = clamp(head_strength, 0.0, 1.0)
        self._gaze_strength = clamp(gaze_strength, 0.0, 1.0)
        self._pose_smoother = ExponentialSmoother(alpha=smoothing)
        self._gaze_smoother = ExponentialSmoother(alpha=smoothing)

    def compute(
        self,
        face: FaceDetectionResult,
        pose: HeadPose,
        gaze: GazeEstimate,
    ) -> FrameCorrection:
        x1, y1, x2, y2 = face.bounding_rect()
        face_width = max(float(x2 - x1), 1.0)
        face_height = max(float(y2 - y1), 1.0)

        smoothed_pose = self._pose_smoother.update(
            np.array([pose.yaw, pose.pitch, pose.roll], dtype=np.float32)
        )
        smoothed_gaze = self._gaze_smoother.update(
            np.array([gaze.horizontal, gaze.vertical], dtype=np.float32)
        )

        face_shift_x = (
            -smoothed_pose[0] / MAX_YAW_DEGREES * face_width * 0.16 * self._head_strength
        )
        face_shift_y = (
            -smoothed_pose[1] / MAX_PITCH_DEGREES * face_height * 0.12 * self._head_strength
        )

        gaze_shift_x = -clamp(
            float(smoothed_gaze[0]) * face_width * 0.035 * self._gaze_strength,
            -face_width * 0.03,
            face_width * 0.03,
        )
        gaze_shift_y = -clamp(
            float(smoothed_gaze[1]) * face_height * 0.025 * self._gaze_strength,
            -face_height * 0.02,
            face_height * 0.02,
        )

        return FrameCorrection(
            face_shift_x=face_shift_x,
            face_shift_y=face_shift_y,
            roll_correction=-smoothed_pose[2] * 0.35 * self._head_strength,
            gaze_shift_x=gaze_shift_x,
            gaze_shift_y=gaze_shift_y,
            pose_yaw=float(smoothed_pose[0]),
            pose_pitch=float(smoothed_pose[1]),
            pose_roll=float(smoothed_pose[2]),
        )

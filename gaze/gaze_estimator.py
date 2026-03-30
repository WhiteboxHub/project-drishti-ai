from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from config import MAX_GAZE_SHIFT
from face.mesh_utils import (
    LEFT_EYE_INDICES,
    LEFT_IRIS_INDICES,
    RIGHT_EYE_INDICES,
    RIGHT_IRIS_INDICES,
    FaceDetectionResult,
)
from utils.math_utils import clamp


@dataclass
class EyeGaze:
    horizontal: float
    vertical: float


@dataclass
class GazeEstimate:
    left_eye: EyeGaze
    right_eye: EyeGaze
    horizontal: float
    vertical: float


class GazeEstimator:
    def estimate(self, face: FaceDetectionResult) -> GazeEstimate:
        left = self._estimate_eye(face, LEFT_EYE_INDICES, LEFT_IRIS_INDICES)
        right = self._estimate_eye(face, RIGHT_EYE_INDICES, RIGHT_IRIS_INDICES)
        return GazeEstimate(
            left_eye=left,
            right_eye=right,
            horizontal=(left.horizontal + right.horizontal) * 0.5,
            vertical=(left.vertical + right.vertical) * 0.5,
        )

    def _estimate_eye(
        self,
        face: FaceDetectionResult,
        eye_indices: list[int],
        iris_indices: list[int],
    ) -> EyeGaze:
        eye_pts = face.points(eye_indices)[:, :2]
        iris_pts = face.points(iris_indices)[:, :2]

        eye_center = np.mean(eye_pts, axis=0)
        iris_center = np.mean(iris_pts, axis=0)

        width = np.linalg.norm(eye_pts[0] - eye_pts[1]) + 1e-6
        height = np.linalg.norm(eye_pts[2] - eye_pts[3]) + 1e-6

        horizontal = clamp(float((iris_center[0] - eye_center[0]) / width), -MAX_GAZE_SHIFT, MAX_GAZE_SHIFT)
        vertical = clamp(float((iris_center[1] - eye_center[1]) / height), -MAX_GAZE_SHIFT, MAX_GAZE_SHIFT)
        return EyeGaze(horizontal=horizontal, vertical=vertical)


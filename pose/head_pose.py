from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from config import MAX_PITCH_DEGREES, MAX_ROLL_DEGREES, MAX_YAW_DEGREES
from face.mesh_utils import FaceDetectionResult
from utils.math_utils import clamp


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass
class HeadPose:
    yaw: float
    pitch: float
    roll: float
    rotation_vector: np.ndarray
    translation_vector: np.ndarray


class HeadPoseEstimator:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        focal_length = frame_width
        self._camera_matrix = np.array(
            [
                [focal_length, 0, frame_width / 2.0],
                [0, focal_length, frame_height / 2.0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def estimate(self, face: FaceDetectionResult) -> HeadPose:
        image_points = face.head_pose_image_points()
        ok, rotation_vector, translation_vector = cv2.solvePnP(
            MODEL_POINTS,
            image_points,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("Head pose estimation failed")

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)

        pitch = clamp(float(euler_angles[0]), -MAX_PITCH_DEGREES, MAX_PITCH_DEGREES)
        yaw = clamp(float(euler_angles[1]), -MAX_YAW_DEGREES, MAX_YAW_DEGREES)
        roll = clamp(float(euler_angles[2]), -MAX_ROLL_DEGREES, MAX_ROLL_DEGREES)

        return HeadPose(
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            rotation_vector=rotation_vector,
            translation_vector=translation_vector,
        )


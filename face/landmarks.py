from __future__ import annotations

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import (
    FACE_MESH_MAX_FACES,
    FACE_MESH_MIN_DETECTION_CONFIDENCE,
    FACE_MESH_MIN_TRACKING_CONFIDENCE,
)
from face.mesh_utils import FaceDetectionResult, normalized_to_pixel_coords


class FaceLandmarkDetector:
    def __init__(self) -> None:
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=FACE_MESH_MAX_FACES,
            refine_landmarks=True,
            min_detection_confidence=FACE_MESH_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=FACE_MESH_MIN_TRACKING_CONFIDENCE,
        )

    def detect(self, frame: np.ndarray) -> Optional[FaceDetectionResult]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None

        frame_height, frame_width = frame.shape[:2]
        landmarks = normalized_to_pixel_coords(
            result.multi_face_landmarks[0].landmark,
            frame_width,
            frame_height,
        )
        return FaceDetectionResult(
            frame_width=frame_width,
            frame_height=frame_height,
            landmarks=landmarks,
        )

    def close(self) -> None:
        self._face_mesh.close()


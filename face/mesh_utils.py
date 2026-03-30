from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


HEAD_POSE_LANDMARKS: Dict[str, int] = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

LEFT_EYE_INDICES = [33, 133, 159, 145]
RIGHT_EYE_INDICES = [362, 263, 386, 374]
LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
FACE_OVAL_INDICES = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


@dataclass
class FaceDetectionResult:
    frame_width: int
    frame_height: int
    landmarks: np.ndarray

    def point(self, index: int) -> np.ndarray:
        return self.landmarks[index]

    def points(self, indices: Iterable[int]) -> np.ndarray:
        idx = list(indices)
        return self.landmarks[idx]

    def eye_box(self, indices: Iterable[int], scale: float = 1.8) -> Tuple[int, int, int, int]:
        pts = self.points(indices)[:, :2]
        min_xy = np.min(pts, axis=0)
        max_xy = np.max(pts, axis=0)
        center = (min_xy + max_xy) * 0.5
        size = (max_xy - min_xy) * scale
        size = np.maximum(size, 12.0)
        x1 = int(np.clip(center[0] - size[0] * 0.5, 0, self.frame_width - 1))
        y1 = int(np.clip(center[1] - size[1] * 0.5, 0, self.frame_height - 1))
        x2 = int(np.clip(center[0] + size[0] * 0.5, 0, self.frame_width - 1))
        y2 = int(np.clip(center[1] + size[1] * 0.5, 0, self.frame_height - 1))
        return x1, y1, x2, y2

    def face_hull(self) -> np.ndarray:
        hull_pts = self.points(FACE_OVAL_INDICES)[:, :2].astype(np.int32)
        return cv2.convexHull(hull_pts)

    def bounding_rect(self, pad: int = 16) -> Tuple[int, int, int, int]:
        xy = self.landmarks[:, :2]
        min_xy = np.min(xy, axis=0) - pad
        max_xy = np.max(xy, axis=0) + pad
        x1 = int(np.clip(min_xy[0], 0, self.frame_width - 1))
        y1 = int(np.clip(min_xy[1], 0, self.frame_height - 1))
        x2 = int(np.clip(max_xy[0], 0, self.frame_width - 1))
        y2 = int(np.clip(max_xy[1], 0, self.frame_height - 1))
        return x1, y1, x2, y2

    def head_pose_image_points(self) -> np.ndarray:
        return np.array(
            [self.point(idx)[:2] for idx in HEAD_POSE_LANDMARKS.values()],
            dtype=np.float64,
        )

    def draw_landmarks(self, frame: np.ndarray, radius: int = 1) -> np.ndarray:
        output = frame.copy()
        for x, y, _ in self.landmarks:
            cv2.circle(output, (int(x), int(y)), radius, (0, 255, 0), -1)
        return output


def normalized_to_pixel_coords(
    normalized_landmarks: List[object],
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    coords = np.zeros((len(normalized_landmarks), 3), dtype=np.float32)
    for idx, landmark in enumerate(normalized_landmarks):
        coords[idx, 0] = landmark.x * frame_width
        coords[idx, 1] = landmark.y * frame_height
        coords[idx, 2] = landmark.z
    return coords


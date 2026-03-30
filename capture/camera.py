from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class CameraCapture:
    index: int
    width: int
    height: int
    fps: int

    def __post_init__(self) -> None:
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._capture = cv2.VideoCapture(self.index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Optional[np.ndarray]:
        if self._capture is None:
            raise RuntimeError("Camera is not open")

        ok, frame = self._capture.read()
        if not ok:
            return None

        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


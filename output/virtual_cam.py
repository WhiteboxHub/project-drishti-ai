from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pyvirtualcam


@dataclass
class VirtualCameraOutput:
    width: int
    height: int
    fps: int

    def __post_init__(self) -> None:
        self._camera: Optional[pyvirtualcam.Camera] = None

    def open(self) -> None:
        self._camera = pyvirtualcam.Camera(
            width=self.width,
            height=self.height,
            fps=self.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
        )

    def send(self, frame: np.ndarray) -> None:
        if self._camera is None:
            raise RuntimeError("Virtual camera is not open")

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self._camera.send(frame)
        self._camera.sleep_until_next_frame()

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None


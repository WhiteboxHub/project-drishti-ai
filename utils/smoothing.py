from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from utils.math_utils import clamp


@dataclass
class ExponentialSmoother:
    alpha: float

    def __post_init__(self) -> None:
        self.alpha = clamp(self.alpha, 0.0, 0.99)
        self._state: Optional[np.ndarray] = None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        if self._state is None:
            self._state = value.copy()
        else:
            self._state = self.alpha * self._state + (1.0 - self.alpha) * value
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


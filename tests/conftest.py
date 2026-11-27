import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Normalised landmark y per finger, knuckle -> fingertip (image y grows downwards).
WRIST = (0.5, 0.9)
FINGERS_UP = {
    "thumb": [0.80, 0.70, 0.60, 0.50],
    "index": [0.60, 0.50, 0.40, 0.30],
    "middle": [0.60, 0.48, 0.36, 0.25],
    "ring": [0.62, 0.52, 0.42, 0.33],
    "pinky": [0.66, 0.58, 0.50, 0.43],
}


class FakeCapture:
    """Stands in for cv2.VideoCapture: serves the given frames, then reports end of stream.

    A None in `frames` is a failed read, like a camera that drops a frame or gets unplugged.
    """

    def __init__(self, frames: list | None = None) -> None:
        self.frames = list(frames or [])
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return bool(self.frames) and not self.released

    def read(self) -> tuple:
        if not self.frames:
            return False, None
        frame = self.frames.pop(0)
        return frame is not None, frame

    def release(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def no_real_camera_or_windows() -> Iterator[MagicMock]:
    """Never touch the real webcam or open windows - even if old code runs a loop at import time."""
    with (
        patch.object(cv2, "VideoCapture", return_value=FakeCapture()) as video_capture,
        patch.object(cv2, "imshow"),
        patch.object(cv2, "waitKey", return_value=-1),
        patch.object(cv2, "destroyAllWindows"),
        patch.object(cv2, "imwrite"),
    ):
        for name in ("holistic_model", "selfie_with_gesture"):
            sys.modules.pop(name, None)
        yield video_capture


@pytest.fixture
def make_capture() -> Callable[[list], FakeCapture]:
    return FakeCapture


@pytest.fixture
def frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def make_hand() -> Callable[..., SimpleNamespace]:
    """Fake MediaPipe hand: 21 landmarks with x/y in 0..1. `curl` bends the index finger back down."""

    def _make(curl: bool = False) -> SimpleNamespace:
        points = [WRIST]
        for i, (name, ys) in enumerate(FINGERS_UP.items()):
            if curl and name == "index":
                ys = [0.60, 0.55, 0.62, 0.66]
            points += [(0.2 + 0.15 * i, y) for y in ys]
        return SimpleNamespace(landmark=[SimpleNamespace(x=x, y=y) for x, y in points])

    return _make


@pytest.fixture
def hands_result() -> Callable[..., SimpleNamespace]:
    def _result(*hands: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(multi_hand_landmarks=list(hands) or None)

    return _result

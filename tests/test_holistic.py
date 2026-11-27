"""Bug 4: the holistic window never called cv2.waitKey, so it couldn't refresh or be closed."""

import importlib
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


def _empty_results() -> SimpleNamespace:
    return SimpleNamespace(face_landmarks=None, right_hand_landmarks=None, left_hand_landmarks=None)


def test_main_when_q_pressed_stops_after_that_frame_and_releases_camera(
    frame: np.ndarray, make_capture: Callable
) -> None:
    holistic = importlib.import_module("holistic_model")
    capture = make_capture([frame, frame.copy(), frame.copy()])
    model = MagicMock()
    model.process.return_value = _empty_results()

    with (
        patch.object(cv2, "VideoCapture", return_value=capture),
        patch.object(holistic.mp_holistic, "Holistic", return_value=model),
        patch.object(cv2, "waitKey", return_value=ord("q")) as wait_key,
        patch.object(cv2, "destroyAllWindows") as destroy,
    ):
        holistic.main()

    assert model.process.call_count == 1
    wait_key.assert_called()
    assert capture.released is True
    destroy.assert_called_once()


def test_main_when_camera_returns_no_frame_stops_cleanly(frame: np.ndarray, make_capture: Callable) -> None:
    holistic = importlib.import_module("holistic_model")
    capture = make_capture([frame])
    model = MagicMock()
    model.process.return_value = _empty_results()
    capture.frames.append(None)

    with (
        patch.object(cv2, "VideoCapture", return_value=capture),
        patch.object(holistic.mp_holistic, "Holistic", return_value=model),
    ):
        holistic.main()

    assert model.process.call_count == 1
    assert capture.released is True

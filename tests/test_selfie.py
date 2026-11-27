"""Bugs 2 and 5: a photo was taken before any gesture, and landmark lists grew forever and mixed hands."""

import importlib
from collections.abc import Callable
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest


@pytest.fixture
def selfie():
    return importlib.import_module("selfie_with_gesture")


def test_hand_coordinates_when_called_returns_21_pixel_points(selfie, make_hand: Callable, frame: np.ndarray) -> None:
    coords = selfie.hand_coordinates(make_hand(), frame)

    assert sorted(coords) == list(range(21))
    assert coords[0] == pytest.approx([0.5 * 640, 0.9 * 480])


def test_hand_coordinates_when_called_per_hand_does_not_mix_hands(
    selfie, make_hand: Callable, frame: np.ndarray
) -> None:
    straight = selfie.hand_coordinates(make_hand(), frame)
    curled = selfie.hand_coordinates(make_hand(curl=True), frame)

    assert len(curled) == 21
    assert curled[7] != straight[7]


def test_should_take_photo_when_palm_open_and_fingers_up_returns_true(
    selfie, make_hand: Callable, hands_result: Callable, frame: np.ndarray
) -> None:
    assert selfie.should_take_photo(hands_result(make_hand()), frame) is True


def test_should_take_photo_when_finger_bent_returns_false(
    selfie, make_hand: Callable, hands_result: Callable, frame: np.ndarray
) -> None:
    assert selfie.should_take_photo(hands_result(make_hand(curl=True)), frame) is False


def test_should_take_photo_when_second_hand_open_returns_true(
    selfie, make_hand: Callable, hands_result: Callable, frame: np.ndarray
) -> None:
    assert selfie.should_take_photo(hands_result(make_hand(curl=True), make_hand()), frame) is True


def test_should_take_photo_when_no_hands_returns_false(selfie, hands_result: Callable, frame: np.ndarray) -> None:
    assert selfie.should_take_photo(hands_result(), frame) is False


def _run_main(selfie, make_capture: Callable, frames: list, results: list) -> MagicMock:
    hands = MagicMock()
    hands.process.side_effect = results
    with (
        patch.object(cv2, "VideoCapture", return_value=make_capture(frames)),
        patch.object(selfie.mp_hands, "Hands", return_value=hands),
        patch.object(cv2, "imwrite") as imwrite,
    ):
        selfie.main()
    return imwrite


def test_main_when_no_gesture_shown_takes_no_photo(
    selfie, hands_result: Callable, frame: np.ndarray, make_capture: Callable
) -> None:
    imwrite = _run_main(selfie, make_capture, [frame, frame.copy()], [hands_result(), hands_result()])

    imwrite.assert_not_called()


def test_main_when_open_palm_shown_saves_photo(
    selfie, make_hand: Callable, hands_result: Callable, frame: np.ndarray, make_capture: Callable
) -> None:
    imwrite = _run_main(selfie, make_capture, [frame, frame.copy()], [hands_result(), hands_result(make_hand())])

    assert imwrite.call_count == 1
    assert imwrite.call_args.args[0] == "photo.jpg"


def test_main_when_q_pressed_stops_and_releases_camera(
    selfie, hands_result: Callable, frame: np.ndarray, make_capture: Callable
) -> None:
    capture = make_capture([frame, frame.copy(), frame.copy()])
    hands = MagicMock()
    hands.process.return_value = hands_result()
    with (
        patch.object(cv2, "VideoCapture", return_value=capture),
        patch.object(selfie.mp_hands, "Hands", return_value=hands),
        patch.object(cv2, "waitKey", return_value=ord("q")),
    ):
        selfie.main()

    assert hands.process.call_count == 1
    assert capture.released is True

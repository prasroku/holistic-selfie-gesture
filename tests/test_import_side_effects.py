"""Bug 1: importing holistic_model started the webcam loop, so selfie_with_gesture never got to its own loop."""

import importlib
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("module", ["holistic_model", "selfie_with_gesture"])
def test_import_when_module_loaded_does_not_open_webcam(module: str, no_real_camera_or_windows: MagicMock) -> None:
    importlib.import_module(module)

    no_real_camera_or_windows.assert_not_called()

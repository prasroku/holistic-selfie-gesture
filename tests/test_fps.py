"""Bug 3: get_fps() reset its timer on every call, so it always showed ~0 FPS."""

import pytest

from libs.fps import FpsCounter


def test_fps_counter_when_first_frame_returns_zero() -> None:
    assert FpsCounter().tick(now=10.0) == 0.0


@pytest.mark.parametrize(("gap", "expected"), [(0.5, 2.0), (1 / 30, 30.0), (0.1, 10.0)])
def test_fps_counter_when_frames_arrive_returns_rate_from_gap(gap: float, expected: float) -> None:
    fps = FpsCounter()
    fps.tick(now=10.0)

    assert fps.tick(now=10.0 + gap) == pytest.approx(expected)


def test_fps_counter_when_clock_does_not_advance_returns_zero_instead_of_dividing_by_zero() -> None:
    fps = FpsCounter()
    fps.tick(now=10.0)

    assert fps.tick(now=10.0) == 0.0

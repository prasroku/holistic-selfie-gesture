import time


class FpsCounter:
    """Frames per second from the gap between consecutive frames."""

    def __init__(self) -> None:
        self._previous: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        previous, self._previous = self._previous, now
        if previous is None or now <= previous:
            return 0.0
        return 1.0 / (now - previous)

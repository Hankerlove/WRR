from __future__ import annotations

from collections import deque

from .backend import BaseVideoBackend
from .types import FrameObservation, WindowState


class RecentWindowWatcher:
    def __init__(self, backend: BaseVideoBackend, window_size: int) -> None:
        self.backend = backend
        self.window_size = window_size
        self.frames: deque[FrameObservation] = deque(maxlen=window_size)
        self.current_state: WindowState | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.current_state = None

    def observe(self, frame: FrameObservation) -> WindowState:
        self.frames.append(frame)
        self.current_state = self.backend.encode_window(list(self.frames))
        return self.current_state

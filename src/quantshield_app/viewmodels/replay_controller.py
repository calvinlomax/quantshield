"""Replay timeline state management."""

from __future__ import annotations

from dataclasses import dataclass, field

from quantshield_app.services.replay_service import ReplayFrame


@dataclass(slots=True)
class ReplayController:
    """Pure state controller for playback stepping and slider scrubbing."""

    frames: list[ReplayFrame] = field(default_factory=list)
    current_index: int = 0

    def set_frames(self, frames: list[ReplayFrame]) -> None:
        self.frames = list(frames)
        self.current_index = 0

    @property
    def has_frames(self) -> bool:
        return bool(self.frames)

    @property
    def max_index(self) -> int:
        return max(len(self.frames) - 1, 0)

    def current_frame(self) -> ReplayFrame:
        if not self.frames:
            raise ValueError("No replay frames are loaded.")
        return self.frames[self.current_index]

    def restart(self) -> ReplayFrame:
        if not self.frames:
            raise ValueError("No replay frames are loaded.")
        self.current_index = 0
        return self.current_frame()

    def step_forward(self) -> ReplayFrame:
        if not self.frames:
            raise ValueError("No replay frames are loaded.")
        if self.current_index < self.max_index:
            self.current_index += 1
        return self.current_frame()

    def step_backward(self) -> ReplayFrame:
        if not self.frames:
            raise ValueError("No replay frames are loaded.")
        if self.current_index > 0:
            self.current_index -= 1
        return self.current_frame()

    def scrub_to(self, index: int) -> ReplayFrame:
        if not self.frames:
            raise ValueError("No replay frames are loaded.")
        self.current_index = min(max(index, 0), self.max_index)
        return self.current_frame()

"""Checkpoint discovery and loading for the desktop inference app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from quantshield.rl import LoadedPolicyCheckpoint, load_actor_critic_checkpoint

DEFAULT_CHECKPOINT_ROOTS = (
    "outputs/rl_policy",
    "outputs/ml_tuned_objective_runs",
    "outputs",
)


@dataclass(slots=True)
class CheckpointDescriptor:
    """Lightweight metadata shown in the checkpoint selector."""

    path: Path
    tickers: list[str]
    lookback_window: int
    hidden_dim: int
    attention_heads: int
    attention_layers: int

    @property
    def display_name(self) -> str:
        tickers = ",".join(self.tickers)
        return (
            f"{self.path.as_posix()} | {tickers} | "
            f"lb={self.lookback_window} hd={self.hidden_dim} "
            f"h={self.attention_heads} l={self.attention_layers}"
        )


class CheckpointService:
    """Discover and load saved actor-critic checkpoints."""

    def __init__(self, search_roots: Iterable[str | Path] = DEFAULT_CHECKPOINT_ROOTS) -> None:
        self.search_roots = [Path(root) for root in search_roots]

    def discover_checkpoints(self) -> list[CheckpointDescriptor]:
        """Return sorted checkpoint descriptors for known model files."""
        candidates: list[Path] = []
        seen: set[Path] = set()
        for root in self.search_roots:
            if not root.exists():
                continue
            for path in root.rglob("actor_critic_policy.pt"):
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(path)

        descriptors: list[CheckpointDescriptor] = []
        for path in candidates:
            descriptors.append(self._read_descriptor(path))

        descriptors.sort(
            key=lambda descriptor: (
                0 if descriptor.path.as_posix().startswith("outputs/rl_policy") else 1,
                descriptor.path.as_posix(),
            )
        )
        return descriptors

    def load_checkpoint(self, checkpoint_path: str | Path, *, device: str | None = None) -> LoadedPolicyCheckpoint:
        """Load a full checkpoint for inference."""
        return load_actor_critic_checkpoint(checkpoint_path, device=device)

    def _read_descriptor(self, checkpoint_path: str | Path) -> CheckpointDescriptor:
        """Read metadata from a serialized checkpoint without instantiating the UI."""
        payload = torch.load(Path(checkpoint_path), map_location="cpu")
        tickers = list(payload.get("tickers", []))
        if not tickers:
            raise ValueError(f"Checkpoint {checkpoint_path} does not include ticker metadata.")
        training_config = dict(payload.get("training_config", {}))
        return CheckpointDescriptor(
            path=Path(checkpoint_path),
            tickers=tickers,
            lookback_window=int(training_config.get("lookback_window", 63)),
            hidden_dim=int(training_config.get("hidden_dim", 192)),
            attention_heads=int(training_config.get("attention_heads", 6)),
            attention_layers=int(training_config.get("attention_layers", 4)),
        )

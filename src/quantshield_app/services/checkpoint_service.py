"""Checkpoint discovery and loading for the desktop inference app."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Iterable

import pandas as pd
import torch

from quantshield.replay_durations import DEFAULT_REPLAY_DURATION_KEY, REPLAY_DURATION_PROFILES, checkpoint_root_for_duration
from quantshield.rl import LoadedPolicyCheckpoint, load_actor_critic_checkpoint
from quantshield.universe import CANONICAL_TOP_50_UNIVERSE, CANONICAL_TOP_ETF_UNIVERSE

DEFAULT_CHECKPOINT_ROOTS = (
    *(profile.checkpoint_root for profile in REPLAY_DURATION_PROFILES),
    "outputs/model_experiments",
    "outputs/model_experiments_50_suite",
    "outputs/portfolio_model_fits",
    "outputs/rl_policy",
)
PLACEHOLDER_TICKER_PATTERN = re.compile(r"^ASSET_\d+$")
SELECTED_CANDIDATE_PATTERN = re.compile(r"^Selected candidate:\s+(.+)$", re.MULTILINE)
SELECTED_EPOCH_PATTERN = re.compile(r"^Selected epoch:\s+(\d+)$", re.MULTILINE)
GENERATED_RUN_PREFIX_PATTERN = re.compile(r"^\d{8}_\d{6}_(.+)$")

DURATION_FOCUS_LABELS = {
    "1mo": "Tactical 1-Month",
    "3mo": "Short-Horizon 3-Month",
    "6mo": "Intermediate 6-Month",
    "1y": "Core 1-Year",
    "3y": "Long-Horizon 3-Year",
    "5y": "Strategic 5-Year",
}

CANDIDATE_DISPLAY_LABELS = {
    "balanced_192x6x4": "Balanced",
    "deeper_224x8x5": "Deep",
    "wider_256x8x4": "Wide",
    "regularized_256x8x5": "Regularized",
    "base_user_shape": "Baseline",
    "portfolio_balanced_192x6x4": "Balanced",
    "portfolio_wide_256x8x4": "Wide",
    "portfolio_deep_224x8x5": "Deep",
    "portfolio_regularized_256x8x5": "Regularized",
    "portfolio_oracle_single_160x4x3": "Oracle Single",
    "portfolio_oracle_top2_192x6x4": "Oracle Top-2",
    "portfolio_oracle_blend_224x8x4": "Oracle Blend",
    "portfolio_oracle_memory_best_asset": "Oracle Memory",
    "portfolio_oracle_memory_best_asset_anchor": "Oracle Memory+",
    "portfolio_experimental_320x10x6": "Experimental",
    "portfolio_titan_448x14x7": "Titan",
}


def is_placeholder_ticker(ticker: str) -> bool:
    """Return True when a ticker is a synthetic asset-slot label."""
    return bool(PLACEHOLDER_TICKER_PATTERN.fullmatch(ticker.strip().upper()))


@dataclass(slots=True)
class CheckpointDescriptor:
    """Lightweight metadata shown in the checkpoint selector."""

    path: Path
    tickers: list[str]
    lookback_window: int
    hidden_dim: int
    attention_heads: int
    attention_layers: int
    duration_key: str | None = None
    candidate_name: str | None = None
    selected_epoch: int | None = None
    validation_significant: bool | None = None
    validation_mean_excess_return: float | None = None
    all_significant: bool | None = None
    all_mean_excess_return: float | None = None
    all_t_statistic: float | None = None
    user_name: str | None = None
    description: str | None = None
    tags: list[str] | None = None
    training_mode: str | None = None
    benchmark_label: str | None = None
    universe_label: str | None = None

    @property
    def uses_placeholder_tickers(self) -> bool:
        return bool(self.tickers) and all(is_placeholder_ticker(ticker) for ticker in self.tickers)

    @property
    def inference_default_tickers(self) -> list[str]:
        if not self.uses_placeholder_tickers:
            return list(self.tickers)
        source_universe = CANONICAL_TOP_50_UNIVERSE if len(self.tickers) > 10 else CANONICAL_TOP_ETF_UNIVERSE
        return list(source_universe[: len(self.tickers)])

    @property
    def slot_count(self) -> int:
        return len(self.tickers)

    @property
    def supported_portfolio_size(self) -> int:
        return 50 if self.slot_count > 10 else 10

    @property
    def display_name(self) -> str:
        focus = DURATION_FOCUS_LABELS.get(self.duration_key or "", "Custom Horizon")
        candidate = self.user_name or CANDIDATE_DISPLAY_LABELS.get(self.candidate_name or "", "General")
        quality = "Validated" if self.validation_significant else ("Benchmark+" if self.all_significant else "Exploratory")
        return f"{focus} {candidate} ({quality})"

    @property
    def quality_tag(self) -> str:
        if self.validation_significant:
            return "Validated"
        if self.all_significant:
            return "Benchmark+"
        return "Exploratory"

    @property
    def source_label(self) -> str:
        if "portfolio_model_fits" in self.path.parts:
            return "Portfolio Fit"
        if "candidate_models" in self.path.parts:
            return "Candidate"
        if "model_experiments_50_suite" in self.path.parts:
            return "Built-in"
        if "replay_checkpoint_suites" in self.path.parts:
            return "Built-in"
        return "Custom"

    @property
    def model_type_label(self) -> str:
        if self.training_mode == "portfolio_fit":
            return "Fit Model"
        if self.training_mode == "experiment":
            return "Candidate Model"
        if self.training_mode == "rl_policy":
            return "RL Policy"
        if "portfolio_model_fits" in self.path.parts:
            return "Fit Model"
        if "candidate_models" in self.path.parts:
            return "Candidate Model"
        if "model_experiments_50_suite" in self.path.parts:
            return "Core Model"
        if "replay_checkpoint_suites" in self.path.parts:
            return "Core Model"
        return "Custom Horizon"

    @property
    def model_group_label(self) -> str:
        return "Core Models" if self.source_label in {"Built-in", "Candidate"} else "Custom Horizon Models"

    @property
    def variant_label(self) -> str:
        if self.user_name:
            return self.user_name
        return CANDIDATE_DISPLAY_LABELS.get(self.candidate_name or "", self.candidate_name or "General")

    @property
    def depth_label(self) -> str:
        return f"{self.hidden_dim}d / {self.attention_heads}h / {self.attention_layers}l"

    @property
    def updated_at(self) -> datetime:
        try:
            timestamp = self.path.stat().st_mtime
        except FileNotFoundError:
            timestamp = 0.0
        return datetime.fromtimestamp(timestamp)

    @property
    def updated_label(self) -> str:
        return self.updated_at.strftime("%Y-%m-%d")

    @property
    def data_window_label(self) -> str:
        if self.duration_key is None:
            return "Custom"
        return DURATION_FOCUS_LABELS.get(self.duration_key, self.duration_key)

    @property
    def preview_policy_predictions_path(self) -> Path:
        return self.path.parent / "policy_predictions.csv"

    @property
    def display_subtitle(self) -> str:
        ticker_scope = (
            f"synthetic {len(self.tickers)}-slot inference"
            if self.uses_placeholder_tickers
            else f"{len(self.tickers)}-ticker policy"
        )
        excess = (
            f" | excess {self.all_mean_excess_return:.3%} | t={self.all_t_statistic:.2f}"
            if self.all_mean_excess_return is not None and self.all_t_statistic is not None
            else ""
        )
        return (
            f"{ticker_scope} | lb {self.lookback_window} | {self.hidden_dim}d/{self.attention_heads}h/{self.attention_layers}l"
            f"{excess}"
        )

    @property
    def detail_text(self) -> str:
        tickers = ", ".join(self.inference_default_tickers)
        lines = [
            f"Name: {self.display_name}",
            f"Group: {self.model_group_label}",
            f"Type: {self.model_type_label}",
            f"Variant: {self.variant_label}",
            f"Tag: {self.quality_tag}",
            f"Scope: {self.display_subtitle}",
            f"Training horizon: {self.duration_key or 'custom'}",
            f"Selected epoch: {self.selected_epoch if self.selected_epoch is not None else 'unknown'}",
            f"Updated: {self.updated_label}",
            f"Tickers: {tickers}",
            f"Lookback window: {self.lookback_window}",
            f"Transformer: hidden={self.hidden_dim}, heads={self.attention_heads}, layers={self.attention_layers}",
            f"Validation significant: {self.validation_significant if self.validation_significant is not None else 'unknown'}",
            f"Validation mean excess: {self.validation_mean_excess_return:.4%}" if self.validation_mean_excess_return is not None else "Validation mean excess: unknown",
            f"All-sample significant: {self.all_significant if self.all_significant is not None else 'unknown'}",
            f"All-sample mean excess: {self.all_mean_excess_return:.4%}" if self.all_mean_excess_return is not None else "All-sample mean excess: unknown",
            f"All-sample t-statistic: {self.all_t_statistic:.3f}" if self.all_t_statistic is not None else "All-sample t-statistic: unknown",
            f"Benchmark: {self.benchmark_label}" if self.benchmark_label else "Benchmark: unknown",
            f"Universe: {self.universe_label}" if self.universe_label else "Universe: unknown",
            f"Description: {self.description}" if self.description else "Description: —",
            f"Tags: {', '.join(self.tags)}" if self.tags else "Tags: —",
            f"Path: {self.path.as_posix()}",
        ]
        return "\n".join(lines)


class CheckpointService:
    """Discover and load saved actor-critic checkpoints."""

    def __init__(self, search_roots: Iterable[str | Path] = DEFAULT_CHECKPOINT_ROOTS) -> None:
        materialized_roots = tuple(search_roots)
        self.search_roots = [Path(root) for root in materialized_roots]
        self._uses_default_roots = materialized_roots == DEFAULT_CHECKPOINT_ROOTS
        self._last_discovery_warnings: list[str] = []

    @property
    def last_discovery_warnings(self) -> list[str]:
        return list(self._last_discovery_warnings)

    def discover_checkpoints(self, *, duration_key: str | None = None) -> list[CheckpointDescriptor]:
        """Return sorted checkpoint descriptors for known model files."""
        self._last_discovery_warnings = []
        preferred_duration = duration_key or DEFAULT_REPLAY_DURATION_KEY
        preferred_root = checkpoint_root_for_duration(preferred_duration)
        roots = list(self.search_roots)
        if preferred_root in roots:
            roots = [preferred_root, *[root for root in roots if root != preferred_root]]
        candidates: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("actor_critic_policy.pt"):
                if "outputs/model_experiments/portfolio_size_50" in path.as_posix():
                    continue
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append(path)

        descriptors: list[CheckpointDescriptor] = []
        for path in candidates:
            try:
                descriptors.append(self._read_descriptor(path))
            except Exception as exc:
                self._last_discovery_warnings.append(f"Skipped unreadable checkpoint {path.as_posix()}: {exc}")
        descriptors = self._deduplicate_descriptors(descriptors)

        descriptors.sort(
            key=lambda descriptor: (
                0 if descriptor.duration_key == preferred_duration else 1,
                0 if descriptor.source_label == "Built-in" else (1 if descriptor.source_label == "Candidate" else 2),
                0 if descriptor.path.as_posix().startswith(preferred_root.as_posix()) else 1,
                descriptor.path.as_posix(),
            )
        )
        return descriptors

    @staticmethod
    def _deduplicate_descriptors(descriptors: list[CheckpointDescriptor]) -> list[CheckpointDescriptor]:
        """Keep the newest descriptor for repeated generated model families."""
        deduped: dict[tuple[object, ...], CheckpointDescriptor] = {}
        ordered: list[CheckpointDescriptor] = []
        for descriptor in sorted(descriptors, key=lambda value: value.updated_at, reverse=True):
            should_dedupe = descriptor.source_label != "Portfolio Fit"
            dedupe_key = (
                descriptor.duration_key,
                CheckpointService._descriptor_family_name(descriptor),
                descriptor.slot_count,
                descriptor.source_label,
            )
            if should_dedupe and dedupe_key in deduped:
                continue
            deduped[dedupe_key] = descriptor
            ordered.append(descriptor)
        return ordered

    @staticmethod
    def _descriptor_family_name(descriptor: CheckpointDescriptor) -> str:
        if descriptor.candidate_name:
            return descriptor.candidate_name
        match = GENERATED_RUN_PREFIX_PATTERN.match(descriptor.path.parent.name)
        if match is not None:
            return match.group(1)
        return descriptor.path.parent.name

    def load_checkpoint(self, checkpoint_path: str | Path, *, device: str | None = None) -> LoadedPolicyCheckpoint:
        """Load a full checkpoint for inference."""
        return load_actor_critic_checkpoint(checkpoint_path, device=device)

    def _read_descriptor(self, checkpoint_path: str | Path) -> CheckpointDescriptor:
        """Read metadata from a serialized checkpoint without instantiating the UI."""
        path = Path(checkpoint_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tickers = list(payload.get("tickers", []))
        if not tickers:
            raise ValueError(f"Checkpoint {checkpoint_path} does not include ticker metadata.")
        training_config = dict(payload.get("training_config", {}))
        duration_key = (
            payload.get("duration_key")
            or next((profile.key for profile in REPLAY_DURATION_PROFILES if profile.key in path.parts), None)
        )
        metadata = self._read_metadata_files(path.parent)
        return CheckpointDescriptor(
            path=path,
            tickers=tickers,
            lookback_window=int(training_config.get("lookback_window", 63)),
            hidden_dim=int(training_config.get("hidden_dim", 240)),
            attention_heads=int(training_config.get("attention_heads", 8)),
            attention_layers=int(training_config.get("attention_layers", 4)),
            duration_key=duration_key,
            candidate_name=metadata["candidate_name"],
            selected_epoch=metadata["selected_epoch"],
            validation_significant=metadata["validation_significant"],
            validation_mean_excess_return=metadata["validation_mean_excess_return"],
            all_significant=metadata["all_significant"],
            all_mean_excess_return=metadata["all_mean_excess_return"],
            all_t_statistic=metadata["all_t_statistic"],
            user_name=metadata["user_name"],
            description=metadata["description"],
            tags=metadata["tags"],
            training_mode=metadata["training_mode"],
            benchmark_label=metadata["benchmark_label"],
            universe_label=metadata["universe_label"],
        )

    @staticmethod
    def _read_metadata_files(directory: Path) -> dict[str, object | None]:
        metadata: dict[str, object | None] = {
            "candidate_name": None,
            "selected_epoch": None,
            "validation_significant": None,
            "validation_mean_excess_return": None,
            "all_significant": None,
            "all_mean_excess_return": None,
            "all_t_statistic": None,
            "user_name": None,
            "description": None,
            "tags": None,
            "training_mode": None,
            "benchmark_label": None,
            "universe_label": None,
        }

        metadata_path = directory / "model_metadata.json"
        if metadata_path.exists():
            try:
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            if isinstance(payload, dict):
                metadata["user_name"] = str(payload.get("name")).strip() if payload.get("name") else None
                metadata["description"] = str(payload.get("description")).strip() if payload.get("description") else None
                tags = payload.get("tags")
                if isinstance(tags, list):
                    metadata["tags"] = [str(tag).strip() for tag in tags if str(tag).strip()]
                metadata["training_mode"] = str(payload.get("training_mode")).strip() if payload.get("training_mode") else None
                metadata["benchmark_label"] = (
                    str(payload.get("benchmark_label")).strip() if payload.get("benchmark_label") else None
                )
                tickers = payload.get("tickers")
                if isinstance(tickers, list) and tickers:
                    metadata["universe_label"] = ", ".join(str(ticker).strip().upper() for ticker in tickers if str(ticker).strip())

        for summary_path in (
            directory / "random_sp500_training_summary.txt",
            directory / "portfolio_fit_summary.txt",
        ):
            if not summary_path.exists():
                continue
            summary_text = summary_path.read_text(encoding="utf-8")
            candidate_match = SELECTED_CANDIDATE_PATTERN.search(summary_text)
            epoch_match = SELECTED_EPOCH_PATTERN.search(summary_text)
            if candidate_match is not None:
                metadata["candidate_name"] = candidate_match.group(1).strip()
            if epoch_match is not None:
                metadata["selected_epoch"] = int(epoch_match.group(1))

        fit_metadata_path = directory / "fit_metadata.csv"
        if fit_metadata_path.exists():
            fit_metadata = pd.read_csv(fit_metadata_path)
            if not fit_metadata.empty:
                row = fit_metadata.iloc[0]
                if metadata["candidate_name"] is None and not pd.isna(row.get("selected_candidate")):
                    metadata["candidate_name"] = str(row["selected_candidate"]).strip()
                if metadata["selected_epoch"] is None and not pd.isna(row.get("selected_epoch")):
                    metadata["selected_epoch"] = int(row["selected_epoch"])

        benchmark_path = directory / "benchmark_summary.csv"
        if benchmark_path.exists():
            benchmark = pd.read_csv(benchmark_path).set_index("Split")
            if "validation" in benchmark.index:
                validation_row = benchmark.loc["validation"]
                metadata["validation_significant"] = CheckpointService._coerce_optional_bool(
                    validation_row.get("significant_outperformance")
                )
                metadata["validation_mean_excess_return"] = CheckpointService._coerce_optional_float(
                    validation_row.get("policy_mean_excess_return")
                )
            if "all" in benchmark.index:
                all_row = benchmark.loc["all"]
                metadata["all_significant"] = CheckpointService._coerce_optional_bool(all_row.get("significant_outperformance"))
                metadata["all_mean_excess_return"] = CheckpointService._coerce_optional_float(
                    all_row.get("policy_mean_excess_return")
                )
                metadata["all_t_statistic"] = CheckpointService._coerce_optional_float(all_row.get("t_statistic"))

        return metadata

    @staticmethod
    def _coerce_optional_bool(value: object) -> bool | None:
        if pd.isna(value):
            return None
        return bool(value)

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        if pd.isna(value):
            return None
        return float(value)

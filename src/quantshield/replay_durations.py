"""Shared replay duration profiles for desktop inference and checkpoint suites."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay


@dataclass(frozen=True, slots=True)
class ReplayDurationProfile:
    """Duration-specific replay and checkpoint metadata."""

    key: str
    label: str
    approximate_business_days: int
    lookback_window: int
    checkpoint_root: str


REPLAY_DURATION_PROFILES: tuple[ReplayDurationProfile, ...] = (
    ReplayDurationProfile("1mo", "1mo", 21, 5, "outputs/replay_checkpoint_suites/1mo"),
    ReplayDurationProfile("3mo", "3mo", 63, 16, "outputs/replay_checkpoint_suites/3mo"),
    ReplayDurationProfile("6mo", "6mo", 126, 32, "outputs/replay_checkpoint_suites/6mo"),
    ReplayDurationProfile("1y", "1y", 252, 63, "outputs/replay_checkpoint_suites/1y"),
    ReplayDurationProfile("3y", "3y", 756, 189, "outputs/replay_checkpoint_suites/3y"),
    ReplayDurationProfile("5y", "5y", 1260, 315, "outputs/replay_checkpoint_suites/5y"),
)

REPLAY_DURATION_MAP: dict[str, ReplayDurationProfile] = {profile.key: profile for profile in REPLAY_DURATION_PROFILES}
DEFAULT_REPLAY_DURATION_KEY = "1y"


def get_replay_duration_profile(duration_key: str) -> ReplayDurationProfile:
    """Return a known replay duration profile."""
    try:
        return REPLAY_DURATION_MAP[duration_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported replay duration: {duration_key}") from exc


def checkpoint_root_for_duration(duration_key: str) -> Path:
    """Return the output root for a duration-specific checkpoint suite."""
    return Path(get_replay_duration_profile(duration_key).checkpoint_root)


def duration_start_from_end(end_date: str | date | pd.Timestamp, duration_key: str) -> pd.Timestamp:
    """Return the replay start date implied by a duration profile and end date."""
    profile = get_replay_duration_profile(duration_key)
    end_timestamp = pd.Timestamp(end_date)
    return pd.Timestamp(end_timestamp - BDay(max(profile.approximate_business_days - 1, 0)))


def duration_end_from_start(start_date: str | date | pd.Timestamp, duration_key: str) -> pd.Timestamp:
    """Return the replay end date implied by a duration profile and start date."""
    profile = get_replay_duration_profile(duration_key)
    start_timestamp = pd.Timestamp(start_date)
    return pd.Timestamp(start_timestamp + BDay(max(profile.approximate_business_days - 1, 0)))

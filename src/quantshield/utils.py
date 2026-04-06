"""Shared utilities used across QuantShield modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sanitize_ticker_slug(tickers: Iterable[str]) -> str:
    """Create a filesystem-friendly ticker slug."""
    return "_".join(ticker.strip().upper().replace("^", "") for ticker in tickers)


def normalize_datetime_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return a copy with a sorted, timezone-naive DatetimeIndex."""
    result = frame.copy()
    result.index = pd.to_datetime(result.index)
    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)
    result = result.sort_index()
    result = result.loc[~result.index.duplicated(keep="last")]
    return result


def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str = "M") -> pd.DatetimeIndex:
    """Return actual trading dates used as rebalance anchors."""
    if index.empty:
        return pd.DatetimeIndex([])
    normalized = pd.DatetimeIndex(pd.to_datetime(index)).sort_values()
    periods = pd.Series(normalized.to_period(frequency), index=normalized)
    mask = periods != periods.shift(-1)
    return normalized[mask.values]


def infer_periods_per_year(index: pd.DatetimeIndex | None = None, default: int = 252) -> int:
    """Infer a sensible annualization factor."""
    if index is None or len(index) < 2:
        return default
    diffs = pd.Series(index[1:] - index[:-1]).dt.days
    median_gap = float(diffs.median())
    if median_gap <= 2.0:
        return 252
    if median_gap <= 8.0:
        return 52
    if median_gap <= 32.0:
        return 12
    return default


def save_frame(frame: pd.DataFrame | pd.Series, path: str | Path, index_label: str = "Date") -> Path:
    """Save a DataFrame or Series to CSV."""
    destination = Path(path)
    ensure_directory(destination.parent)
    if isinstance(frame, pd.Series):
        frame.to_frame(name=frame.name or "value").to_csv(destination, index_label=index_label)
    else:
        frame.to_csv(destination, index_label=index_label)
    return destination


def format_percent(value: float) -> str:
    """Format a value as a percentage string."""
    return f"{value:.2%}"

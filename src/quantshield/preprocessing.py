"""Data cleaning and return generation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantshield.utils import normalize_datetime_index


def clean_price_data(
    prices: pd.DataFrame,
    *,
    drop_all_nan_assets: bool = True,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """Align a price panel on date and handle missing observations without lookahead."""
    if prices.empty:
        raise ValueError("Price data is empty.")

    cleaned = normalize_datetime_index(prices)
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")

    if drop_all_nan_assets:
        cleaned = cleaned.dropna(axis=1, how="all")
    if cleaned.empty:
        raise ValueError("All assets were dropped after removing all-NaN columns.")

    if forward_fill:
        cleaned = cleaned.ffill()

    cleaned = cleaned.dropna(axis=0, how="any")
    if cleaned.empty:
        raise ValueError("No overlapping price history remains after alignment.")
    return cleaned


def compute_returns(prices: pd.DataFrame, return_type: str = "simple") -> pd.DataFrame:
    """Compute aligned daily return series from prices."""
    if return_type not in {"simple", "log"}:
        raise ValueError("return_type must be either 'simple' or 'log'.")

    if return_type == "simple":
        returns = prices.pct_change()
    else:
        returns = np.log(prices / prices.shift(1))
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty:
        raise ValueError("Return series is empty after computation.")
    return returns


def resample_prices(prices: pd.DataFrame, rule: str = "ME", method: str = "last") -> pd.DataFrame:
    """Resample a price frame using a pandas aggregation rule."""
    if str(rule).strip().upper() in {"M", "1M"}:
        rule = "ME"
    resampler = prices.resample(rule)
    if method == "last":
        return resampler.last().dropna(how="all")
    if method == "first":
        return resampler.first().dropna(how="all")
    raise ValueError("Unsupported resample method. Use 'first' or 'last'.")


def trailing_window(returns: pd.DataFrame, end_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
    """Return the trailing lookback window ending at end_date."""
    history = returns.loc[:end_date]
    if history.empty:
        raise ValueError("No history available up to the requested end date.")
    return history.iloc[-lookback:]

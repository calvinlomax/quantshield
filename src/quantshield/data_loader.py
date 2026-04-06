"""Local market data ingestion backed by yfinance with CSV caching."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
import warnings

import pandas as pd

from quantshield.utils import ensure_directory, normalize_datetime_index, sanitize_ticker_slug

DownloadProvider = Callable[..., pd.DataFrame]

DEFAULT_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "VNQ"]


def _default_provider(**kwargs: object) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(**kwargs)


def extract_adjusted_close(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close prices from a yfinance response."""
    if data.empty:
        raise ValueError("Downloaded yfinance dataset is empty.")

    if isinstance(data.columns, pd.MultiIndex):
        level0 = {str(value) for value in data.columns.get_level_values(0)}
        level1 = {str(value) for value in data.columns.get_level_values(1)}

        if "Adj Close" in level0:
            prices = data["Adj Close"].copy()
        elif "Adj Close" in level1:
            prices = data.xs("Adj Close", axis=1, level=1).copy()
        elif "Close" in level0:
            warnings.warn("Adjusted close not returned by yfinance; using Close instead.", stacklevel=2)
            prices = data["Close"].copy()
        elif "Close" in level1:
            warnings.warn("Adjusted close not returned by yfinance; using Close instead.", stacklevel=2)
            prices = data.xs("Close", axis=1, level=1).copy()
        else:
            raise ValueError("Could not find an adjusted close or close field in the yfinance response.")
    else:
        if "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
            prices.columns = tickers[:1]
        elif "Close" in data.columns:
            warnings.warn("Adjusted close not returned by yfinance; using Close instead.", stacklevel=2)
            prices = data[["Close"]].copy()
            prices.columns = tickers[:1]
        else:
            raise ValueError("Could not find an adjusted close or close field in the yfinance response.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    ordered_columns = [ticker for ticker in tickers if ticker in prices.columns]
    if ordered_columns:
        prices = prices.loc[:, ordered_columns]
    prices = normalize_datetime_index(prices)
    prices.index.name = "Date"
    return prices


class MarketDataLoader:
    """Fetches local market data from yfinance and caches it on disk."""

    def __init__(self, cache_dir: str | Path = "data/raw", provider: DownloadProvider | None = None) -> None:
        self.cache_dir = ensure_directory(cache_dir)
        self.provider = provider or _default_provider

    def cache_path(self, tickers: list[str], start_date: str, end_date: str | None) -> Path:
        """Return the CSV cache path for a download request."""
        end_component = end_date or "latest"
        slug = sanitize_ticker_slug(tickers)
        return self.cache_dir / f"{slug}_{start_date}_{end_component}.csv"

    def load_cached_prices(self, path: str | Path) -> pd.DataFrame:
        """Load cached prices from CSV."""
        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached price file does not exist: {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        prices = normalize_datetime_index(prices)
        prices.index.name = "Date"
        return prices

    def fetch_prices(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str | None = None,
        *,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch adjusted close prices, preferring local cache when available."""
        if not tickers:
            raise ValueError("At least one ticker must be supplied.")

        cache_path = self.cache_path(tickers, start_date, end_date)
        if use_cache and cache_path.exists() and not force_refresh:
            return self.load_cached_prices(cache_path)

        raw = self.provider(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,
            actions=False,
            group_by="column",
            threads=False,
        )
        prices = extract_adjusted_close(raw, tickers)
        if prices.empty:
            raise ValueError("No price data was returned after extraction.")

        prices.index.name = "Date"
        prices.to_csv(cache_path, index_label="Date")
        return prices

"""Market data download and preprocessing for desktop replay inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantshield.data_loader import MarketDataLoader
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield_app.services.checkpoint_service import is_placeholder_ticker


@dataclass(slots=True)
class PreparedMarketData:
    """Prepared market data ready for policy replay."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    replay_returns: pd.DataFrame
    portfolio_tickers: list[str]
    benchmark_ticker: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp | None


class MarketDataService:
    """Download local market data and prepare replay-ready returns."""

    def __init__(self, loader: MarketDataLoader | None = None) -> None:
        self.loader = loader or MarketDataLoader(cache_dir="data/raw")

    def prepare_market_data(
        self,
        *,
        portfolio_tickers: list[str],
        benchmark_ticker: str,
        start_date: str,
        end_date: str | None,
        lookback_window: int,
        return_type: str = "simple",
        force_refresh: bool = False,
    ) -> PreparedMarketData:
        """Download or load cached market data with enough history for the policy lookback."""
        if not portfolio_tickers:
            raise ValueError("At least one portfolio ticker is required.")
        normalized_portfolio = [ticker.strip().upper() for ticker in portfolio_tickers if ticker.strip()]
        normalized_benchmark = benchmark_ticker.strip().upper()
        placeholder_tickers = [ticker for ticker in [*normalized_portfolio, normalized_benchmark] if is_placeholder_ticker(ticker)]
        if placeholder_tickers:
            joined = ", ".join(sorted(set(placeholder_tickers)))
            raise ValueError(
                "Synthetic checkpoint asset slots cannot be downloaded from yfinance. "
                f"Select real ticker symbols instead of: {joined}"
            )
        start_timestamp = pd.Timestamp(start_date)
        end_timestamp = pd.Timestamp(end_date) if end_date else None
        if end_timestamp is not None and end_timestamp < start_timestamp:
            raise ValueError("End date must be on or after the start date.")

        buffered_start = (start_timestamp - pd.tseries.offsets.BDay(max(lookback_window * 4, 252))).date().isoformat()
        fetch_tickers = list(normalized_portfolio)
        if normalized_benchmark not in fetch_tickers:
            fetch_tickers.append(normalized_benchmark)

        raw_prices = self.loader.fetch_prices(
            fetch_tickers,
            buffered_start,
            end_timestamp.date().isoformat() if end_timestamp is not None else None,
            use_cache=True,
            force_refresh=force_refresh,
        )
        prices = clean_price_data(raw_prices, drop_all_nan_assets=True, forward_fill=True)
        returns = compute_returns(prices, return_type=return_type)
        replay_returns = returns.loc[start_timestamp:end_timestamp]
        if replay_returns.empty:
            raise ValueError("No replay return data is available for the requested date range.")
        if normalized_benchmark not in replay_returns.columns:
            raise ValueError(f"Benchmark ticker '{normalized_benchmark}' is not available in the replay data.")
        if len(returns.loc[: replay_returns.index[0]]) < lookback_window:
            raise ValueError(
                "Not enough pre-start history is available to build the model lookback window for the selected start date."
            )
        return PreparedMarketData(
            prices=prices,
            returns=returns,
            replay_returns=replay_returns,
            portfolio_tickers=normalized_portfolio,
            benchmark_ticker=normalized_benchmark,
            start_date=start_timestamp,
            end_date=end_timestamp,
        )

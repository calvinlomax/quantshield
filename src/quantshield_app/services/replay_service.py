"""Inference-time replay backtest service for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from quantshield.metrics import cumulative_returns, performance_summary
from quantshield.rl import LoadedPolicyCheckpoint, predict_policy_weights
from quantshield.utils import get_rebalance_dates, infer_periods_per_year
from quantshield_app.services.market_data_service import PreparedMarketData


@dataclass(slots=True)
class ReplayRequest:
    """User-specified replay request."""

    checkpoint_path: Path
    portfolio_tickers: list[str]
    start_date: str
    end_date: str | None
    rebalance_frequency: str
    benchmark_ticker: str
    starting_capital: float
    force_refresh: bool = False


@dataclass(slots=True)
class ReplayFrame:
    """Single playback frame shown in the UI."""

    index: int
    date: pd.Timestamp
    portfolio_value: float
    benchmark_value: float
    portfolio_return: float
    benchmark_return: float
    excess_return: float
    turnover: float
    rebalanced: bool
    weights: pd.Series


@dataclass(slots=True)
class PolicyReplayResult:
    """Replay result consumed by the desktop UI."""

    checkpoint: LoadedPolicyCheckpoint
    frames: list[ReplayFrame]
    comparison_returns: pd.DataFrame
    cumulative_values: pd.DataFrame
    weights_history: pd.DataFrame
    daily_weights: pd.DataFrame
    summary_table: pd.DataFrame
    metrics: dict[str, float]
    requested_tickers: list[str]
    benchmark_ticker: str
    starting_capital: float


class ReplayService:
    """Simulate inference-driven backtests for desktop playback."""

    def build_replay(
        self,
        *,
        checkpoint: LoadedPolicyCheckpoint,
        market_data: PreparedMarketData,
        rebalance_frequency: str,
        starting_capital: float,
    ) -> PolicyReplayResult:
        """Run deterministic actor-critic inference across a historical replay period."""
        if starting_capital <= 0.0:
            raise ValueError("Starting capital must be positive.")

        portfolio_tickers = self._resolve_ticker_order(market_data.portfolio_tickers, checkpoint.tickers)
        benchmark = market_data.benchmark_ticker
        replay_returns = market_data.replay_returns
        rebalance_dates = get_rebalance_dates(replay_returns.index, frequency=rebalance_frequency)
        rebalance_dates = pd.DatetimeIndex(
            [date for date in rebalance_dates if replay_returns.index.get_loc(date) < len(replay_returns.index) - 1]
        )
        if rebalance_dates.empty:
            raise ValueError("No valid rebalance dates were found for the requested replay period.")

        frames: list[ReplayFrame] = []
        comparison_rows: list[dict[str, object]] = []
        daily_weights_records: list[pd.Series] = []
        rebalance_weight_records: list[pd.Series] = []
        previous_weights: pd.Series | None = None
        portfolio_value = float(starting_capital)
        benchmark_value = float(starting_capital)

        for position, rebalance_date in enumerate(rebalance_dates):
            window = market_data.returns.loc[:rebalance_date, portfolio_tickers].iloc[-checkpoint.training_config.lookback_window :]
            if len(window) < checkpoint.training_config.lookback_window:
                continue

            weights = predict_policy_weights(checkpoint, window)
            turnover = 0.0 if previous_weights is None else float(np.abs(weights - previous_weights).sum())
            rebalance_weight_records.append(weights.rename(rebalance_date))

            start_idx = replay_returns.index.get_loc(rebalance_date) + 1
            if position < len(rebalance_dates) - 1:
                end_idx = replay_returns.index.get_loc(rebalance_dates[position + 1])
            else:
                end_idx = len(replay_returns.index) - 1
            holding_period_returns = replay_returns.iloc[start_idx : end_idx + 1]
            if holding_period_returns.empty:
                continue

            for offset, (date, row) in enumerate(holding_period_returns.iterrows()):
                portfolio_return = float(row[portfolio_tickers].dot(weights))
                benchmark_return = float(row[benchmark])
                portfolio_value *= 1.0 + portfolio_return
                benchmark_value *= 1.0 + benchmark_return

                comparison_rows.append(
                    {
                        "Date": date,
                        "portfolio": portfolio_return,
                        "benchmark": benchmark_return,
                        "excess": portfolio_return - benchmark_return,
                    }
                )
                daily_weights_records.append(weights.rename(date))
                frames.append(
                    ReplayFrame(
                        index=len(frames),
                        date=pd.Timestamp(date),
                        portfolio_value=portfolio_value,
                        benchmark_value=benchmark_value,
                        portfolio_return=portfolio_return,
                        benchmark_return=benchmark_return,
                        excess_return=portfolio_return - benchmark_return,
                        turnover=turnover if offset == 0 else 0.0,
                        rebalanced=offset == 0,
                        weights=weights.copy(),
                    )
                )

            previous_weights = weights

        if not frames:
            raise ValueError("Replay generation produced no frames. Try an earlier start date or different frequency.")

        comparison_returns = pd.DataFrame(comparison_rows).set_index("Date")
        comparison_returns.index = pd.to_datetime(comparison_returns.index)
        comparison_returns.index.name = "Date"
        cumulative_values = cumulative_returns(
            comparison_returns[["portfolio", "benchmark"]],
            start_value=starting_capital,
        ).rename(columns={"portfolio": "Portfolio", "benchmark": "Benchmark"})

        weights_history = pd.DataFrame(rebalance_weight_records)
        weights_history.index.name = "RebalanceDate"
        daily_weights = pd.DataFrame(daily_weights_records)
        daily_weights.index.name = "Date"

        periods_per_year = infer_periods_per_year(comparison_returns.index, default=252)
        summary = performance_summary(comparison_returns[["portfolio", "benchmark"]], periods_per_year=periods_per_year)
        metrics = {
            "annualized_return": float(summary.loc["portfolio", "annualized_return"]),
            "annualized_volatility": float(summary.loc["portfolio", "annualized_volatility"]),
            "sharpe_ratio": float(summary.loc["portfolio", "sharpe_ratio"]),
            "max_drawdown": float(summary.loc["portfolio", "max_drawdown"]),
            "total_return": float(cumulative_values["Portfolio"].iloc[-1] / starting_capital - 1.0),
            "benchmark_total_return": float(cumulative_values["Benchmark"].iloc[-1] / starting_capital - 1.0),
            "benchmark_annualized_return": float(summary.loc["benchmark", "annualized_return"]),
            "excess_total_return": float(
                cumulative_values["Portfolio"].iloc[-1] / starting_capital
                - cumulative_values["Benchmark"].iloc[-1] / starting_capital
            ),
        }

        return PolicyReplayResult(
            checkpoint=checkpoint,
            frames=frames,
            comparison_returns=comparison_returns,
            cumulative_values=cumulative_values,
            weights_history=weights_history,
            daily_weights=daily_weights,
            summary_table=summary,
            metrics=metrics,
            requested_tickers=list(market_data.portfolio_tickers),
            benchmark_ticker=benchmark,
            starting_capital=starting_capital,
        )

    @staticmethod
    def _resolve_ticker_order(requested_tickers: list[str], checkpoint_tickers: list[str]) -> list[str]:
        """Align user-entered tickers to the checkpoint order and reject mismatches."""
        requested_set = set(requested_tickers)
        checkpoint_set = set(checkpoint_tickers)
        if requested_set != checkpoint_set:
            expected = ", ".join(checkpoint_tickers)
            received = ", ".join(requested_tickers)
            raise ValueError(
                "Ticker/model mismatch. "
                f"Selected checkpoint expects [{expected}] but the app received [{received}]."
            )
        return list(checkpoint_tickers)

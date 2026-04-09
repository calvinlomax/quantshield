"""Inference-time replay backtest service for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from quantshield.metrics import cumulative_returns, performance_summary
from quantshield.rl import LoadedPolicyCheckpoint, predict_policy_weights
from quantshield.utils import generate_schedule, infer_periods_per_year
from quantshield_app.services.market_data_service import PreparedMarketData
from quantshield_app.services.treasury_rate_service import TreasuryRateAssumption, TreasuryRateService


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
    shares: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))


@dataclass(slots=True)
class PolicyReplayResult:
    """Replay result consumed by the desktop UI."""

    checkpoint: LoadedPolicyCheckpoint
    frames: list[ReplayFrame]
    prices: pd.DataFrame
    comparison_returns: pd.DataFrame
    cumulative_values: pd.DataFrame
    weights_history: pd.DataFrame
    daily_weights: pd.DataFrame
    asset_returns: pd.DataFrame
    benchmark_returns: pd.Series
    summary_table: pd.DataFrame
    metrics: dict[str, float]
    requested_tickers: list[str]
    benchmark_ticker: str
    starting_capital: float
    rebalance_frequency: str
    rebalance_label: str
    rebalance_mode: str
    estimated_steps: int
    risk_free_assumption: TreasuryRateAssumption | None = None


class ReplayService:
    """Simulate inference-driven backtests for desktop playback."""

    def __init__(self, treasury_rate_service: TreasuryRateService | None = None) -> None:
        self.treasury_rate_service = treasury_rate_service or TreasuryRateService()

    def build_replay(
        self,
        *,
        checkpoint: LoadedPolicyCheckpoint,
        market_data: PreparedMarketData,
        rebalance_frequency: str,
        starting_capital: float,
        rebalance_label: str | None = None,
        rebalance_mode: str = "manual",
        estimated_steps: int | None = None,
    ) -> PolicyReplayResult:
        """Run deterministic actor-critic inference across a historical replay period."""
        if starting_capital <= 0.0:
            raise ValueError("Starting capital must be positive.")

        portfolio_tickers = self._validate_portfolio_tickers(market_data.portfolio_tickers)
        benchmark = market_data.benchmark_ticker
        replay_returns = market_data.replay_returns
        asset_returns = replay_returns.loc[:, portfolio_tickers].copy()
        benchmark_returns = replay_returns.loc[:, benchmark].copy()
        replay_price_columns = list(dict.fromkeys([*portfolio_tickers, benchmark]))
        replay_prices = market_data.prices.loc[:, replay_price_columns].reindex(replay_returns.index).ffill()
        if replay_prices.isna().any().any():
            raise ValueError("Replay price data contains missing values after return-date alignment.")
        rebalance_dates = generate_schedule(replay_returns.index, frequency=rebalance_frequency)
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
        equal_weight_value = float(starting_capital)
        benchmark_value = float(starting_capital)
        benchmark_shares: int | None = None
        benchmark_cash = float(starting_capital)
        equal_weight_target = pd.Series(
            np.full(len(portfolio_tickers), 1.0 / len(portfolio_tickers), dtype=float),
            index=portfolio_tickers,
        )

        for position, rebalance_date in enumerate(rebalance_dates):
            window = market_data.returns.loc[:rebalance_date, portfolio_tickers].iloc[-checkpoint.training_config.lookback_window :]
            if len(window) < checkpoint.training_config.lookback_window:
                continue

            weights = predict_policy_weights(checkpoint, window, tickers=portfolio_tickers)
            turnover = 0.0 if previous_weights is None else float(np.abs(weights - previous_weights).sum())

            rebalance_prices = replay_prices.loc[rebalance_date, portfolio_tickers].astype(float)
            safe_rebalance_prices = rebalance_prices.replace(0.0, np.nan)
            target_notional = portfolio_value * weights
            share_counts = (target_notional / safe_rebalance_prices).fillna(0.0).clip(lower=0.0).apply(np.floor).astype(int)
            invested_value = float((share_counts * rebalance_prices).sum())
            cash_balance = float(portfolio_value - invested_value)
            actual_rebalance_weights = ((share_counts * rebalance_prices) / portfolio_value).fillna(0.0) if portfolio_value > 0.0 else weights * 0.0
            rebalance_weight_records.append(actual_rebalance_weights.rename(rebalance_date))

            equal_target_notional = equal_weight_value * equal_weight_target
            equal_share_counts = (
                (equal_target_notional / safe_rebalance_prices).fillna(0.0).clip(lower=0.0).apply(np.floor).astype(int)
            )
            equal_invested_value = float((equal_share_counts * rebalance_prices).sum())
            equal_cash_balance = float(equal_weight_value - equal_invested_value)

            if benchmark_shares is None:
                benchmark_rebalance_price = float(replay_prices.loc[rebalance_date, benchmark])
                if benchmark_rebalance_price > 0.0:
                    benchmark_shares = int(starting_capital // benchmark_rebalance_price)
                    benchmark_cash = float(starting_capital - benchmark_shares * benchmark_rebalance_price)
                else:
                    benchmark_shares = 0
                    benchmark_cash = float(starting_capital)

            start_idx = replay_returns.index.get_loc(rebalance_date) + 1
            if position < len(rebalance_dates) - 1:
                end_idx = replay_returns.index.get_loc(rebalance_dates[position + 1])
            else:
                end_idx = len(replay_returns.index) - 1
            holding_period_prices = replay_prices.iloc[start_idx : end_idx + 1]
            if holding_period_prices.empty:
                continue

            previous_portfolio_value = float(portfolio_value)
            previous_equal_weight_value = float(equal_weight_value)
            previous_benchmark_value = float(benchmark_value)
            for offset, (date, price_row) in enumerate(holding_period_prices.iterrows()):
                asset_prices = price_row.loc[portfolio_tickers].astype(float)
                asset_values = share_counts.astype(float) * asset_prices
                portfolio_value = float(cash_balance + asset_values.sum())
                current_weights = (asset_values / portfolio_value).fillna(0.0) if portfolio_value > 0.0 else weights * 0.0

                equal_asset_values = equal_share_counts.astype(float) * asset_prices
                equal_weight_value = float(equal_cash_balance + equal_asset_values.sum())

                benchmark_price = float(price_row.loc[benchmark])
                benchmark_value = float(benchmark_cash + (benchmark_shares or 0) * benchmark_price)
                portfolio_return = (portfolio_value / previous_portfolio_value - 1.0) if previous_portfolio_value > 0.0 else 0.0
                equal_weight_return = (
                    equal_weight_value / previous_equal_weight_value - 1.0
                    if previous_equal_weight_value > 0.0
                    else 0.0
                )
                benchmark_return = (benchmark_value / previous_benchmark_value - 1.0) if previous_benchmark_value > 0.0 else 0.0

                comparison_rows.append(
                    {
                        "Date": date,
                        "portfolio": portfolio_return,
                        "equal_weight": equal_weight_return,
                        "benchmark": benchmark_return,
                        "excess": portfolio_return - benchmark_return,
                        "active_vs_equal_weight": portfolio_return - equal_weight_return,
                    }
                )
                daily_weights_records.append(current_weights.rename(date))
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
                        weights=current_weights.copy(),
                        shares=share_counts.copy(),
                    )
                )
                previous_portfolio_value = portfolio_value
                previous_equal_weight_value = equal_weight_value
                previous_benchmark_value = benchmark_value

            previous_weights = weights

        if not frames:
            raise ValueError(
                "Replay generation produced no frames. The selected date range does not contain enough observations "
                f"to build the {checkpoint.training_config.lookback_window}-day model lookback and a forward period. "
                "Choose a longer window or a more recent end date."
            )

        comparison_returns = pd.DataFrame(comparison_rows).set_index("Date")
        comparison_returns.index = pd.to_datetime(comparison_returns.index)
        comparison_returns.index.name = "Date"
        cumulative_values = cumulative_returns(
            comparison_returns[["portfolio", "equal_weight", "benchmark"]],
            start_value=starting_capital,
        ).rename(columns={"portfolio": "Portfolio", "equal_weight": "Equal Weight", "benchmark": "Benchmark"})

        weights_history = pd.DataFrame(rebalance_weight_records)
        weights_history.index.name = "RebalanceDate"
        daily_weights = pd.DataFrame(daily_weights_records)
        daily_weights.index.name = "Date"

        periods_per_year = infer_periods_per_year(comparison_returns.index, default=252)
        risk_free_assumption = self.treasury_rate_service.resolve_for_window(
            business_days=len(replay_returns.index),
            as_of_date=market_data.start_date,
        )
        summary = performance_summary(
            comparison_returns[["portfolio", "equal_weight", "benchmark"]],
            periods_per_year=periods_per_year,
            risk_free_rate=risk_free_assumption.annual_rate,
        )
        metrics = {
            "annualized_return": float(summary.loc["portfolio", "annualized_return"]),
            "annualized_volatility": float(summary.loc["portfolio", "annualized_volatility"]),
            "sharpe_ratio": float(summary.loc["portfolio", "sharpe_ratio"]),
            "max_drawdown": float(summary.loc["portfolio", "max_drawdown"]),
            "risk_free_rate": float(risk_free_assumption.annual_rate),
            "total_return": float(cumulative_values["Portfolio"].iloc[-1] / starting_capital - 1.0),
            "equal_weight_total_return": float(cumulative_values["Equal Weight"].iloc[-1] / starting_capital - 1.0),
            "equal_weight_annualized_return": float(summary.loc["equal_weight", "annualized_return"]),
            "equal_weight_annualized_volatility": float(summary.loc["equal_weight", "annualized_volatility"]),
            "equal_weight_sharpe_ratio": float(summary.loc["equal_weight", "sharpe_ratio"]),
            "equal_weight_max_drawdown": float(summary.loc["equal_weight", "max_drawdown"]),
            "benchmark_total_return": float(cumulative_values["Benchmark"].iloc[-1] / starting_capital - 1.0),
            "benchmark_annualized_return": float(summary.loc["benchmark", "annualized_return"]),
            "excess_total_return": float(
                cumulative_values["Portfolio"].iloc[-1] / starting_capital
                - cumulative_values["Benchmark"].iloc[-1] / starting_capital
            ),
            "active_vs_equal_weight_total_return": float(
                cumulative_values["Portfolio"].iloc[-1] / starting_capital
                - cumulative_values["Equal Weight"].iloc[-1] / starting_capital
            ),
        }

        return PolicyReplayResult(
            checkpoint=checkpoint,
            frames=frames,
            prices=replay_prices.loc[:, portfolio_tickers].copy(),
            comparison_returns=comparison_returns,
            cumulative_values=cumulative_values,
            weights_history=weights_history,
            daily_weights=daily_weights,
            asset_returns=asset_returns,
            benchmark_returns=benchmark_returns,
            summary_table=summary,
            metrics=metrics,
            requested_tickers=list(market_data.portfolio_tickers),
            benchmark_ticker=benchmark,
            starting_capital=starting_capital,
            rebalance_frequency=rebalance_frequency,
            rebalance_label=rebalance_label or rebalance_frequency,
            rebalance_mode=rebalance_mode,
            estimated_steps=int(estimated_steps or len(rebalance_dates)),
            risk_free_assumption=risk_free_assumption,
        )

    @staticmethod
    def _validate_portfolio_tickers(requested_tickers: list[str], minimum_count: int = 5) -> list[str]:
        """Validate the user-selected portfolio tickers for arbitrary-universe inference."""
        normalized = [ticker.strip().upper() for ticker in requested_tickers if ticker.strip()]
        if len(normalized) < minimum_count:
            raise ValueError(f"Select at least {minimum_count} tickers for policy replay.")
        return normalized

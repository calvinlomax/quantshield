"""Rolling portfolio backtest engine."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantshield.config import BacktestConfig as SettingsBacktestConfig
from quantshield.config import OptimizationConfig, RiskConfig
from quantshield.metrics import performance_summary
from quantshield.optimization import OptimizationResult, optimize_portfolio
from quantshield.risk import RiskEstimate, estimate_risk
from quantshield.utils import get_rebalance_dates

BacktestConfig = SettingsBacktestConfig


@dataclass(slots=True)
class BacktestResult:
    """Aggregated result from a rolling backtest."""

    comparison_returns: pd.DataFrame
    weights_history: pd.DataFrame
    turnover: pd.Series
    rebalance_log: pd.DataFrame
    performance_summary: pd.DataFrame
    latest_weights: pd.Series
    latest_risk_estimate: RiskEstimate


def _select_training_window(returns: pd.DataFrame, config: BacktestConfig, rebalance_date: pd.Timestamp) -> pd.DataFrame:
    history = returns.loc[:rebalance_date]
    if config.expanding_window:
        return history
    return history.iloc[-config.lookback_days :]


def run_rolling_backtest(
    returns: pd.DataFrame,
    *,
    risk_config: RiskConfig,
    optimization_config: OptimizationConfig,
    backtest_config: BacktestConfig,
    asset_class_map: dict[str, str] | None = None,
    periods_per_year: int = 252,
) -> BacktestResult:
    """Run a walk-forward backtest using only historical data available at each rebalance."""
    if returns.empty:
        raise ValueError("Cannot backtest an empty return panel.")

    benchmark = backtest_config.benchmark_ticker
    if benchmark not in returns.columns:
        raise ValueError(f"Benchmark ticker '{benchmark}' is not present in the return data.")

    rebalance_dates = get_rebalance_dates(returns.index, frequency=backtest_config.rebalance_frequency)
    rebalance_dates = pd.DatetimeIndex(
        [date for date in rebalance_dates if returns.index.get_loc(date) < len(returns.index) - 1]
    )
    if rebalance_dates.empty:
        raise ValueError("No valid rebalance dates were found in the return history.")

    comparison_segments: list[pd.DataFrame] = []
    weights_records: list[pd.Series] = []
    turnover_records: list[pd.Series] = []
    log_records: list[dict[str, object]] = []

    previous_weights: pd.Series | None = None
    latest_risk_estimate: RiskEstimate | None = None
    latest_weights: pd.Series | None = None

    for position, rebalance_date in enumerate(rebalance_dates):
        training_returns = _select_training_window(returns, backtest_config, rebalance_date)
        if len(training_returns) < max(backtest_config.min_history_days, 2):
            continue

        latest_risk_estimate = estimate_risk(
            training_returns,
            risk_config,
            periods_per_year=periods_per_year,
        )
        optimization_result: OptimizationResult = optimize_portfolio(
            latest_risk_estimate.mean,
            latest_risk_estimate.covariance,
            optimization_config,
            previous_weights=previous_weights,
            asset_class_map=asset_class_map,
        )

        start_idx = returns.index.get_loc(rebalance_date) + 1
        if position < len(rebalance_dates) - 1:
            end_date = rebalance_dates[position + 1]
            end_idx = returns.index.get_loc(end_date)
        else:
            end_idx = len(returns.index) - 1

        holding_period_returns = returns.iloc[start_idx : end_idx + 1]
        if holding_period_returns.empty:
            continue

        portfolio_returns = holding_period_returns.dot(optimization_result.weights)
        segment = pd.DataFrame(
            {
                "portfolio": portfolio_returns,
                "equal_weight": holding_period_returns.mean(axis=1),
                "benchmark": holding_period_returns[benchmark],
            }
        )
        comparison_segments.append(segment)

        weight_record = optimization_result.weights.rename(rebalance_date)
        turnover_record = pd.Series(optimization_result.turnover, index=["turnover"], name=rebalance_date)
        weights_records.append(weight_record)
        turnover_records.append(turnover_record)
        log_records.append(
            {
                "rebalance_date": rebalance_date,
                "train_start": training_returns.index[0],
                "train_end": training_returns.index[-1],
                "sample_size": len(training_returns),
                "objective": optimization_config.objective,
                "success": optimization_result.success,
                "message": optimization_result.message,
                "expected_return": optimization_result.expected_return,
                "expected_volatility": optimization_result.expected_volatility,
                "turnover": optimization_result.turnover,
                "constraint_violations": "; ".join(
                    f"{key}={value:.6f}" for key, value in optimization_result.constraint_violations.items()
                ),
            }
        )

        previous_weights = optimization_result.weights
        latest_weights = optimization_result.weights

    if not comparison_segments or latest_risk_estimate is None or latest_weights is None:
        raise ValueError("Backtest could not generate any out-of-sample periods. Check the lookback configuration.")

    comparison_returns = pd.concat(comparison_segments).sort_index()
    comparison_returns = comparison_returns.loc[~comparison_returns.index.duplicated(keep="last")]
    comparison_returns.index.name = "Date"
    weights_history = pd.DataFrame(weights_records)
    weights_history.index.name = "RebalanceDate"
    turnover = pd.concat(turnover_records, axis=1).T["turnover"]
    turnover.index.name = "RebalanceDate"
    rebalance_log = pd.DataFrame(log_records).set_index("rebalance_date")
    rebalance_log.index.name = "RebalanceDate"

    summary = performance_summary(
        comparison_returns,
        periods_per_year=periods_per_year,
    )

    return BacktestResult(
        comparison_returns=comparison_returns,
        weights_history=weights_history,
        turnover=turnover,
        rebalance_log=rebalance_log,
        performance_summary=summary,
        latest_weights=latest_weights,
        latest_risk_estimate=latest_risk_estimate,
    )

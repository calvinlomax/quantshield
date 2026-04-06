"""End-to-end pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantshield.backtest import BacktestResult, run_rolling_backtest
from quantshield.config import AppConfig
from quantshield.data_loader import MarketDataLoader
from quantshield.plotting import (
    plot_correlation_heatmap,
    plot_cumulative_return_curves,
    plot_drawdown,
    plot_efficient_frontier,
    plot_price_history,
    plot_risk_contribution,
    plot_rolling_volatility,
    plot_weights_over_time,
)
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield.reporting import build_risk_and_stress_reports, build_summary_text, write_summary_text
from quantshield.risk import compare_covariance_estimators
from quantshield.stress_test import StressScenarioResult, run_default_stress_tests
from quantshield.utils import ensure_directory, save_frame


@dataclass(slots=True)
class PipelineResult:
    """Outputs produced by a full pipeline run."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    covariance_summary: pd.DataFrame
    backtest_result: BacktestResult
    risk_attribution: pd.DataFrame
    stress_results: dict[str, StressScenarioResult]
    stress_summary: pd.DataFrame
    summary_text: str


def prepare_market_data(config: AppConfig, loader: MarketDataLoader | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and preprocess local market data."""
    loader = loader or MarketDataLoader(cache_dir=config.data.cache_dir)
    raw_prices = loader.fetch_prices(
        config.data.tickers,
        config.data.start_date,
        config.data.end_date,
        use_cache=config.data.use_cache,
        force_refresh=config.data.force_refresh,
    )
    prices = clean_price_data(
        raw_prices,
        drop_all_nan_assets=config.preprocessing.drop_all_nan_assets,
        forward_fill=config.preprocessing.forward_fill_prices,
    )
    returns = compute_returns(prices, return_type=config.preprocessing.return_type)
    return prices, returns


def run_pipeline_from_data(
    config: AppConfig,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
) -> PipelineResult:
    """Execute the full QuantShield workflow from prepared price and return data."""
    _, covariance_summary = compare_covariance_estimators(
        returns,
        periods_per_year=config.preprocessing.annualization_factor,
        ewma_span=config.risk.ewma_span,
    )
    backtest_result = run_rolling_backtest(
        returns,
        risk_config=config.risk,
        optimization_config=config.optimization,
        backtest_config=config.backtest,
        asset_class_map=config.data.asset_class_map,
        periods_per_year=config.preprocessing.annualization_factor,
    )
    stress_results = run_default_stress_tests(
        backtest_result.latest_weights,
        backtest_result.latest_risk_estimate.covariance,
        config.data.asset_class_map,
    )
    risk_attribution, stress_summary = build_risk_and_stress_reports(
        backtest_result.latest_weights,
        backtest_result.latest_risk_estimate.covariance,
        stress_results,
    )
    summary_text = build_summary_text(
        tickers=config.data.tickers,
        sample_start=str(prices.index.min().date()),
        sample_end=str(prices.index.max().date()),
        lookback_days=config.backtest.lookback_days,
        rebalance_frequency=config.backtest.rebalance_frequency,
        covariance_estimator=config.risk.covariance_estimator,
        objective=config.optimization.objective,
        final_weights=backtest_result.latest_weights,
        performance_summary=backtest_result.performance_summary,
        turnover=backtest_result.turnover,
        risk_attribution=risk_attribution,
        stress_summary=stress_summary,
        latest_constraint_status=str(backtest_result.rebalance_log.iloc[-1]["constraint_violations"]),
    )
    return PipelineResult(
        prices=prices,
        returns=returns,
        covariance_summary=covariance_summary,
        backtest_result=backtest_result,
        risk_attribution=risk_attribution,
        stress_results=stress_results,
        stress_summary=stress_summary,
        summary_text=summary_text,
    )


def run_pipeline(config: AppConfig, loader: MarketDataLoader | None = None) -> PipelineResult:
    """Execute the full QuantShield workflow."""
    prices, returns = prepare_market_data(config, loader=loader)
    return run_pipeline_from_data(config, prices, returns)


def save_pipeline_artifacts(result: PipelineResult, config: AppConfig) -> dict[str, Path]:
    """Persist tables, figures, and text reports to disk."""
    processed_dir = ensure_directory("data/processed")
    figures_dir = ensure_directory(config.reporting.figures_dir)
    tables_dir = ensure_directory(config.reporting.tables_dir)

    clean_prices = result.prices.rename_axis("Date")
    returns = result.returns.rename_axis("Date")
    performance_summary = result.backtest_result.performance_summary.rename_axis("Portfolio")
    comparison_returns = result.backtest_result.comparison_returns.rename_axis("Date")
    weights_history = result.backtest_result.weights_history.rename_axis("RebalanceDate")
    turnover = result.backtest_result.turnover.rename_axis("RebalanceDate").rename("turnover")
    rebalance_log = result.backtest_result.rebalance_log.rename_axis("RebalanceDate")
    final_weights = result.backtest_result.latest_weights.rename_axis("Ticker").rename("weight")
    risk_attribution = result.risk_attribution.rename_axis("Ticker")
    stress_summary = result.stress_summary.rename_axis("Scenario")
    covariance_summary = result.covariance_summary.rename_axis("CovarianceEstimator")

    paths = {
        "clean_prices": save_frame(clean_prices, processed_dir / "clean_prices.csv"),
        "returns": save_frame(returns, processed_dir / "daily_returns.csv"),
        "performance_summary": save_frame(performance_summary, tables_dir / "performance_summary.csv"),
        "comparison_returns": save_frame(comparison_returns, tables_dir / "comparison_returns.csv"),
        "weights_history": save_frame(weights_history, tables_dir / "weights_history.csv"),
        "turnover": save_frame(turnover, tables_dir / "turnover.csv"),
        "rebalance_log": save_frame(rebalance_log, tables_dir / "rebalance_log.csv"),
        "final_weights": save_frame(final_weights, tables_dir / "final_weights.csv"),
        "risk_attribution": save_frame(risk_attribution, tables_dir / "risk_attribution.csv"),
        "stress_summary": save_frame(stress_summary, tables_dir / "stress_summary.csv"),
        "covariance_summary": save_frame(covariance_summary, tables_dir / "covariance_summary.csv"),
        "summary_text": write_summary_text(result.summary_text, tables_dir / "summary_report.txt"),
        "price_history_fig": plot_price_history(result.prices, figures_dir / "price_history.png"),
        "correlation_heatmap_fig": plot_correlation_heatmap(result.returns, figures_dir / "correlation_heatmap.png"),
        "cumulative_returns_fig": plot_cumulative_return_curves(
            result.backtest_result.comparison_returns,
            figures_dir / "cumulative_returns.png",
        ),
        "rolling_volatility_fig": plot_rolling_volatility(
            result.backtest_result.comparison_returns,
            figures_dir / "rolling_volatility.png",
            window=config.reporting.rolling_vol_window,
            periods_per_year=config.preprocessing.annualization_factor,
        ),
        "drawdown_fig": plot_drawdown(result.backtest_result.comparison_returns["portfolio"], figures_dir / "drawdown.png"),
        "weights_fig": plot_weights_over_time(result.backtest_result.weights_history, figures_dir / "weights_over_time.png"),
        "risk_contribution_fig": plot_risk_contribution(
            result.backtest_result.latest_weights,
            result.backtest_result.latest_risk_estimate.covariance,
            figures_dir / "risk_contribution.png",
        ),
        "efficient_frontier_fig": plot_efficient_frontier(
            result.backtest_result.latest_risk_estimate.mean,
            result.backtest_result.latest_risk_estimate.covariance,
            figures_dir / "efficient_frontier.png",
            min_weight=0.0 if isinstance(config.optimization.min_weight, dict) else float(config.optimization.min_weight),
            max_weight=1.0 if isinstance(config.optimization.max_weight, dict) else float(config.optimization.max_weight),
        ),
    }
    return paths

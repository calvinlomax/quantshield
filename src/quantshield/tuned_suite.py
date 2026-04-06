"""Reusable tuned-suite utilities for ML demonstration generation and benchmarking."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quantshield.config import AppConfig
from quantshield.pipeline import prepare_market_data, run_pipeline_from_data, save_pipeline_artifacts
from quantshield.utils import ensure_directory, save_frame

TUNED_PRESETS: dict[str, dict[str, object]] = {
    "min_variance": {
        "tickers": ["SPY", "QQQ", "GLD"],
        "covariance_estimator": "ledoit_wolf",
        "lookback_days": 252,
        "expanding_window": False,
        "max_weight": 0.70,
        "turnover_penalty": 0.0,
    },
    "mean_variance": {
        "tickers": ["SPY", "QQQ", "GLD"],
        "covariance_estimator": "historical",
        "lookback_days": 252,
        "expanding_window": False,
        "max_weight": 1.0,
        "turnover_penalty": 0.0,
        "risk_aversion": 0.1,
    },
    "risk_parity": {
        "tickers": ["SPY", "QQQ", "GLD"],
        "covariance_estimator": "ledoit_wolf",
        "lookback_days": 252,
        "expanding_window": False,
        "max_weight": 0.70,
        "turnover_penalty": 0.0,
    },
    "equal_weight": {
        "tickers": ["SPY", "QQQ", "GLD"],
        "covariance_estimator": "ledoit_wolf",
        "lookback_days": 252,
        "expanding_window": False,
        "max_weight": 1.0,
        "turnover_penalty": 0.0,
    },
}


@dataclass(slots=True)
class TunedSuiteResult:
    """Artifacts returned by the tuned objective suite."""

    prices: pd.DataFrame
    returns: pd.DataFrame
    comparison: pd.DataFrame
    comparison_path: Path
    artifact_paths: dict[str, dict[str, Path]]


def build_objective_config(
    base_config: AppConfig,
    objective: str,
    preset: dict[str, object],
    output_root: str | Path,
) -> AppConfig:
    """Build a per-objective config from the tuned preset table."""
    config = deepcopy(base_config)
    config.data.tickers = list(preset["tickers"])
    config.data.asset_class_map = {
        ticker: base_config.data.asset_class_map[ticker]
        for ticker in config.data.tickers
        if ticker in base_config.data.asset_class_map
    }
    config.optimization.objective = objective
    config.risk.covariance_estimator = str(preset["covariance_estimator"])
    config.backtest.lookback_days = int(preset["lookback_days"])
    config.backtest.expanding_window = bool(preset["expanding_window"])
    config.optimization.max_weight = float(preset["max_weight"])
    config.optimization.turnover_penalty = float(preset["turnover_penalty"])
    if "risk_aversion" in preset:
        config.optimization.risk_aversion = float(preset["risk_aversion"])

    output_path = Path(output_root)
    config.reporting.output_dir = str(output_path / objective)
    config.reporting.figures_dir = str(output_path / objective / "figures")
    config.reporting.tables_dir = str(output_path / objective / "tables")
    return config


def run_tuned_objective_suite(
    base_config: AppConfig,
    *,
    output_root: str | Path = "outputs/tuned_objective_runs",
    force_refresh: bool = False,
) -> TunedSuiteResult:
    """Run the tuned objective suite used as the ML policy's demonstration set."""
    working_config = deepcopy(base_config)
    if force_refresh:
        working_config.data.force_refresh = True

    full_prices, full_returns = prepare_market_data(working_config)
    suite_root = ensure_directory(output_root)
    comparison_rows: list[dict[str, object]] = []
    artifact_paths: dict[str, dict[str, Path]] = {}

    for objective, preset in TUNED_PRESETS.items():
        config = build_objective_config(working_config, objective, preset, suite_root)
        tickers = config.data.tickers
        prices = full_prices[tickers].copy()
        returns = full_returns[tickers].copy()

        result = run_pipeline_from_data(config, prices, returns)
        paths = save_pipeline_artifacts(result, config)
        artifact_paths[objective] = paths

        portfolio_row = result.backtest_result.performance_summary.loc["portfolio"]
        benchmark_row = result.backtest_result.performance_summary.loc["benchmark"]
        comparison_rows.append(
            {
                "objective": objective,
                "tickers": ",".join(tickers),
                "annualized_return": portfolio_row["annualized_return"],
                "benchmark_return": benchmark_row["annualized_return"],
                "excess_return_vs_spy": portfolio_row["annualized_return"] - benchmark_row["annualized_return"],
                "annualized_volatility": portfolio_row["annualized_volatility"],
                "sharpe_ratio": portfolio_row["sharpe_ratio"],
                "max_drawdown": portfolio_row["max_drawdown"],
                "average_turnover": result.backtest_result.turnover.mean(),
                "summary_report": str(paths["summary_text"]),
            }
        )

    comparison = pd.DataFrame(comparison_rows).set_index("objective").sort_values("annualized_return", ascending=False)
    comparison.index.name = "Objective"
    comparison_path = save_frame(comparison, suite_root / "tuned_objective_comparison.csv")

    return TunedSuiteResult(
        prices=full_prices,
        returns=full_returns,
        comparison=comparison,
        comparison_path=comparison_path,
        artifact_paths=artifact_paths,
    )

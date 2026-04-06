"""Run a backtest-tuned objective suite and save separate report bundles."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import AppConfig, load_config
from quantshield.pipeline import prepare_market_data, run_pipeline_from_data, save_pipeline_artifacts
from quantshield.utils import ensure_directory

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and save the backtest-tuned QuantShield objective suite.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--output-root",
        default="outputs/tuned_objective_runs",
        help="Root directory where tuned per-objective reports will be saved.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Refetch data even if cached data exists.")
    return parser.parse_args()


def _build_objective_config(base_config: AppConfig, objective: str, preset: dict[str, object], output_root: Path) -> AppConfig:
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

    config.reporting.output_dir = str(output_root / objective)
    config.reporting.figures_dir = str(output_root / objective / "figures")
    config.reporting.tables_dir = str(output_root / objective / "tables")
    return config


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    if args.force_refresh:
        base_config.data.force_refresh = True

    full_prices, full_returns = prepare_market_data(base_config)
    output_root = ensure_directory(args.output_root)
    comparison_rows: list[dict[str, object]] = []

    for objective, preset in TUNED_PRESETS.items():
        config = _build_objective_config(base_config, objective, preset, output_root)
        tickers = config.data.tickers
        prices = full_prices[tickers].copy()
        returns = full_returns[tickers].copy()

        result = run_pipeline_from_data(config, prices, returns)
        artifact_paths = save_pipeline_artifacts(result, config)
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
                "summary_report": artifact_paths["summary_text"],
            }
        )

        print(f"[{objective}] saved tuned report bundle to {config.reporting.output_dir}")

    comparison = pd.DataFrame(comparison_rows).set_index("objective").sort_values("annualized_return", ascending=False)
    comparison_path = output_root / "tuned_objective_comparison.csv"
    comparison.to_csv(comparison_path, index_label="objective")

    print("")
    print(comparison.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Note: this suite is backtest-tuned on the same historical sample and should be treated as exploratory, not as an out-of-sample claim.")
    print(f"Saved tuned comparison table to {comparison_path}")


if __name__ == "__main__":
    main()

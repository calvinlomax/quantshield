"""Run the classical rolling backtest and save benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.pipeline import prepare_market_data
from quantshield.backtest import run_rolling_backtest
from quantshield.plotting import plot_cumulative_return_curves, plot_drawdown, plot_weights_over_time
from quantshield.utils import ensure_directory, save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuantShield classical rolling backtest.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch data even if cached data exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.force_refresh:
        config.data.force_refresh = True

    prices, returns = prepare_market_data(config)
    result = run_rolling_backtest(
        returns,
        risk_config=config.risk,
        optimization_config=config.optimization,
        backtest_config=config.backtest,
        asset_class_map=config.data.asset_class_map,
        periods_per_year=config.preprocessing.annualization_factor,
    )

    tables_dir = ensure_directory(config.reporting.tables_dir)
    figures_dir = ensure_directory(config.reporting.figures_dir)

    save_frame(result.performance_summary, tables_dir / "performance_summary.csv")
    save_frame(result.comparison_returns, tables_dir / "comparison_returns.csv")
    save_frame(result.weights_history, tables_dir / "weights_history.csv")
    save_frame(result.turnover, tables_dir / "turnover.csv")
    save_frame(result.rebalance_log, tables_dir / "rebalance_log.csv")

    plot_cumulative_return_curves(result.comparison_returns, figures_dir / "cumulative_returns.png")
    plot_drawdown(result.comparison_returns["portfolio"], figures_dir / "drawdown.png")
    plot_weights_over_time(result.weights_history, figures_dir / "weights_over_time.png")

    print("Backtest complete.")
    print(result.performance_summary.to_string(float_format=lambda value: f"{value:0.4f}"))


if __name__ == "__main__":
    main()

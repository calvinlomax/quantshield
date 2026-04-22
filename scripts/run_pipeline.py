"""Run the classical QuantShield benchmark workflow."""

from __future__ import annotations

import argparse

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

bootstrap_project_root(__file__)

from quantshield.config import load_config
from quantshield.pipeline import run_pipeline, save_pipeline_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the classical QuantShield benchmark pipeline.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch data even if cached data exists.")
    parser.add_argument("--objective", help="Override optimization objective.")
    parser.add_argument("--covariance-estimator", help="Override covariance estimator.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.force_refresh:
        config.data.force_refresh = True
    if args.objective:
        config.optimization.objective = args.objective
    if args.covariance_estimator:
        config.risk.covariance_estimator = args.covariance_estimator

    result = run_pipeline(config)
    save_pipeline_artifacts(result, config)

    print(result.summary_text)
    print("")
    print(f"Saved tables to {config.reporting.tables_dir}")
    print(f"Saved figures to {config.reporting.figures_dir}")


if __name__ == "__main__":
    main()

"""Generate the tuned benchmark suite used to train and compare the ML policy."""

from __future__ import annotations

import argparse

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

bootstrap_project_root(__file__)

from quantshield.config import load_config
from quantshield.tuned_suite import run_tuned_objective_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and save the tuned objective suite used as QuantShield's benchmark demonstration set."
    )
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--output-root",
        default="outputs/tuned_objective_runs",
        help="Root directory where tuned per-objective reports will be saved.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Refetch data even if cached data exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    result = run_tuned_objective_suite(
        base_config,
        output_root=args.output_root,
        force_refresh=args.force_refresh,
    )

    for objective in result.comparison.index:
        summary_report = result.artifact_paths[objective]["summary_text"]
        output_dir = summary_report.parent.parent
        print(f"[{objective}] saved tuned report bundle to {output_dir}")

    print("")
    print(result.comparison.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Note: this suite is backtest-tuned on the same historical sample and should be treated as exploratory.")
    print(f"Saved tuned comparison table to {result.comparison_path}")


if __name__ == "__main__":
    main()

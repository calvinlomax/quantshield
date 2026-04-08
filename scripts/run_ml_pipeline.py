"""Run the end-to-end QuantShield ML workflow."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.pipeline import prepare_market_data
from quantshield.tuned_suite import TUNED_PRESETS, run_tuned_objective_suite

try:
    from quantshield.rl import (
        RLTrainingConfig,
        build_offline_rl_dataset,
        load_weight_histories_from_suite,
        save_actor_critic_artifacts,
        train_transformer_actor_critic,
    )
except ImportError as exc:  # pragma: no cover - depends on optional torch install
    raise SystemExit(
        "The ML pipeline requires the optional PyTorch dependency. "
        "Install it with `pip install -r requirements-rl.txt` or `pip install -e .[rl]`."
    ) from exc


DEFAULT_OBJECTIVES = list(TUNED_PRESETS.keys())
DEFAULT_SUITE_ROOT = "outputs/ml_tuned_objective_runs"
DEFAULT_SUITE_REBALANCE_FREQUENCY = "B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QuantShield's end-to-end ML workflow: build demonstrations, train the transformer policy, and save artifacts."
    )
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--suite-root",
        default=DEFAULT_SUITE_ROOT,
        help="Directory where the tuned benchmark suite will be written or loaded from.",
    )
    parser.add_argument(
        "--suite-rebalance-frequency",
        default=DEFAULT_SUITE_REBALANCE_FREQUENCY,
        help="Rebalance frequency used when generating the benchmark demonstration suite.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/rl_policy",
        help="Directory where the trained policy artifacts will be written.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=DEFAULT_OBJECTIVES,
        help="Objectives to include as offline demonstrations.",
    )
    parser.add_argument("--lookback-window", type=int, default=63, help="Trailing return window used to build states.")
    parser.add_argument("--epochs", type=int, default=180, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--hidden-dim", type=int, default=240, help="Transformer hidden dimension.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Number of cross-asset attention heads.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Number of stacked cross-asset attention layers.")
    parser.add_argument("--device", help="Optional torch device override, for example cpu or cuda.")
    parser.add_argument("--skip-suite", action="store_true", help="Reuse existing tuned-suite artifacts instead of regenerating them.")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch data even if cached data exists.")
    return parser.parse_args()


def _build_summary_text(
    *,
    suite_root: Path,
    output_dir: Path,
    objectives: list[str],
    lookback_window: int,
    suite_rebalance_frequency: str,
    suite_comparison,
    benchmark_summary,
    evaluation_summary,
    latest_policy_weights,
) -> str:
    lines = [
        "QuantShield ML Pipeline Summary",
        "==============================",
        "",
        f"Suite root: {suite_root}",
        f"Policy output dir: {output_dir}",
        f"Objectives: {', '.join(objectives)}",
        f"Lookback window: {lookback_window} trading days",
        f"Suite rebalance frequency: {suite_rebalance_frequency}",
        "",
        "Tuned benchmark suite:",
        suite_comparison.to_string(float_format=lambda value: f"{value:0.4f}"),
        "",
        "Policy benchmark comparison:",
        benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
        "",
        "Policy evaluation summary:",
        evaluation_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
        "",
        "Latest policy weights:",
        latest_policy_weights.to_string(float_format=lambda value: f"{value:0.4f}"),
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config)
    if args.force_refresh:
        app_config.data.force_refresh = True

    if args.skip_suite:
        _, returns = prepare_market_data(app_config)
        suite_root = Path(args.suite_root)
        comparison_path = suite_root / "tuned_objective_comparison.csv"
        suite_comparison = pd.read_csv(comparison_path, index_col=0) if comparison_path.exists() else None
    else:
        suite_config = deepcopy(app_config)
        suite_config.backtest.rebalance_frequency = args.suite_rebalance_frequency
        suite_result = run_tuned_objective_suite(
            suite_config,
            output_root=args.suite_root,
            force_refresh=args.force_refresh,
        )
        returns = suite_result.returns
        suite_root = Path(args.suite_root)
        suite_comparison = suite_result.comparison

    weight_histories = load_weight_histories_from_suite(args.suite_root, args.objectives)
    dataset = build_offline_rl_dataset(
        returns,
        weight_histories,
        lookback_window=args.lookback_window,
        benchmark_ticker=app_config.backtest.benchmark_ticker,
    )
    training_config = RLTrainingConfig(
        lookback_window=args.lookback_window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
    )
    result = train_transformer_actor_critic(dataset, training_config, device=args.device)
    artifact_paths = save_actor_critic_artifacts(result, args.output_dir)

    if suite_comparison is None:
        comparison_path = Path(args.suite_root) / "tuned_objective_comparison.csv"
        if not comparison_path.exists():
            raise FileNotFoundError(
                "Tuned suite comparison table not found. Run without `--skip-suite` or generate the suite with "
                "`python scripts/run_tuned_suite.py` first."
            )
        suite_comparison = pd.read_csv(comparison_path, index_col=0)

    summary_text = _build_summary_text(
        suite_root=suite_root,
        output_dir=Path(args.output_dir),
        objectives=list(args.objectives),
        lookback_window=args.lookback_window,
        suite_rebalance_frequency=args.suite_rebalance_frequency,
        suite_comparison=suite_comparison,
        benchmark_summary=result.benchmark_summary,
        evaluation_summary=result.evaluation_summary,
        latest_policy_weights=result.latest_policy_weights,
    )
    summary_path = Path(args.output_dir) / "ml_pipeline_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print("QuantShield ML pipeline complete.")
    print("")
    print("Tuned benchmark suite:")
    print(suite_comparison.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Policy benchmark comparison:")
    print(result.benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Policy evaluation summary:")
    print(result.evaluation_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print(f"Saved policy checkpoint to {artifact_paths['model']}")
    print(f"Saved ML pipeline summary to {summary_path}")
    print(f"Saved ML figures to {Path(artifact_paths['training_diagnostics_fig']).parent}")


if __name__ == "__main__":
    main()

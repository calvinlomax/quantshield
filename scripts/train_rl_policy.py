"""Train the QuantShield transformer actor-critic policy from saved suite weights."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.pipeline import prepare_market_data

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
        "RL training requires the optional PyTorch dependency. "
        "Install it with `pip install -r requirements-rl.txt` or `pip install -e .[rl]`."
    ) from exc


DEFAULT_OBJECTIVES = ["min_variance", "mean_variance", "risk_parity", "equal_weight"]
DEFAULT_SUITE_ROOT = "outputs/ml_tuned_objective_runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the QuantShield transformer actor-critic policy from saved suite weights.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to the base YAML config.")
    parser.add_argument(
        "--suite-root",
        default=DEFAULT_SUITE_ROOT,
        help="Directory containing the saved tuned suite weight histories.",
    )
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=DEFAULT_OBJECTIVES,
        help="Objectives to use as offline demonstrations.",
    )
    parser.add_argument("--output-dir", default="outputs/rl_policy", help="Directory where RL artifacts will be written.")
    parser.add_argument("--lookback-window", type=int, default=63, help="Trailing return window used to build states.")
    parser.add_argument("--epochs", type=int, default=180, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--hidden-dim", type=int, default=240, help="Transformer hidden dimension.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Number of cross-asset attention heads.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Number of stacked cross-asset attention layers.")
    parser.add_argument("--device", help="Optional torch device override, for example cpu or cuda.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config)
    _, returns = prepare_market_data(app_config)

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

    print("Transformer actor-critic training complete.")
    print("Benchmark comparison:")
    print(result.benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Evaluation summary:")
    print(result.evaluation_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print(f"Saved model checkpoint to {artifact_paths['model']}")
    print(f"Saved benchmark summary to {artifact_paths['benchmark_summary']}")
    print(f"Saved RL figures to {Path(artifact_paths['training_diagnostics_fig']).parent}")


if __name__ == "__main__":
    main()

"""Train the QuantShield actor-critic on random 10-stock S&P 500 universes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.rl import RLTrainingConfig, save_actor_critic_artifacts, train_transformer_actor_critic
from quantshield.sp500_random_training import RandomSP500TrainingSpec, build_random_sp500_dataset
from quantshield.utils import save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the actor-critic on 64 random 10-stock portfolios sampled from S&P 500 constituents."
    )
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--output-dir", default="outputs/rl_policy", help="Directory where model artifacts will be written.")
    parser.add_argument("--start-date", default="2018-01-01", help="Historical sample start date for yfinance downloads.")
    parser.add_argument("--end-date", help="Optional historical sample end date.")
    parser.add_argument("--candidate-pool-size", type=int, default=80, help="Random S&P 500 stock pool size used to draw universes.")
    parser.add_argument(
        "--random-universes",
        type=int,
        help="Number of random 10-stock universes to generate. Defaults to the epoch count.",
    )
    parser.add_argument("--portfolio-size", type=int, default=10, help="Stocks per random universe.")
    parser.add_argument("--epochs", type=int, default=64, help="Training epochs for the actor-critic.")
    parser.add_argument("--lookback-window", type=int, default=63, help="Trailing return window used to build states.")
    parser.add_argument("--hidden-dim", type=int, default=240, help="Transformer hidden dimension.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Transformer attention heads.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Transformer attention layers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--rebalance-frequency", default="W-FRI", help="Benchmark demonstration rebalance frequency.")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=["mean_variance"],
        choices=["min_variance", "mean_variance", "risk_parity", "equal_weight"],
        help="Demonstration objectives used when generating random-universe targets.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for universe sampling and training.")
    parser.add_argument("--device", help="Optional torch device override.")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch yfinance prices even if cached.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app_config = load_config(args.config)
    app_config.data.start_date = args.start_date
    app_config.data.end_date = args.end_date
    app_config.data.force_refresh = args.force_refresh
    random_universes = args.random_universes or args.epochs

    print(
        f"Preparing random S&P 500 dataset with {random_universes} universes, "
        f"{args.portfolio_size} stocks per universe, {args.epochs} training epochs.",
        flush=True,
    )

    dataset, summary = build_random_sp500_dataset(
        app_config,
        spec=RandomSP500TrainingSpec(
            start_date=args.start_date,
            end_date=args.end_date,
            candidate_pool_size=args.candidate_pool_size,
            random_universes=random_universes,
            portfolio_size=args.portfolio_size,
            random_seed=args.seed,
            rebalance_frequency=args.rebalance_frequency,
            lookback_window=args.lookback_window,
            force_refresh=args.force_refresh,
            objectives=tuple(args.objectives),
        ),
    )
    print(
        f"Built offline dataset with {len(dataset.states)} samples "
        f"across {summary['universe_summary']['universe_id'].nunique()} random universes.",
        flush=True,
    )

    training_config = RLTrainingConfig(
        lookback_window=args.lookback_window,
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(
        f"Training transformer actor-critic with hidden_dim={args.hidden_dim}, "
        f"heads={args.attention_heads}, layers={args.attention_layers}.",
        flush=True,
    )
    result = train_transformer_actor_critic(dataset, training_config, device=args.device)
    artifact_paths = save_actor_critic_artifacts(result, args.output_dir)

    output_dir = Path(args.output_dir)
    universe_summary = summary["universe_summary"]
    save_frame(universe_summary, output_dir / "random_sp500_universe_summary.csv")
    (output_dir / "random_sp500_training_summary.txt").write_text(
        "\n".join(
            [
                "QuantShield Random S&P 500 Training Summary",
                "==========================================",
                f"Candidate pool size: {args.candidate_pool_size}",
                f"Random universes: {random_universes}",
                f"Portfolio size: {args.portfolio_size}",
                f"Training epochs: {args.epochs}",
                f"Lookback window: {args.lookback_window}",
                f"Rebalance frequency: {args.rebalance_frequency}",
                f"Objectives: {', '.join(args.objectives)}",
                f"Sampled ticker count: {summary['sampled_ticker_count']}",
                "",
                universe_summary.head(10).to_string(index=False),
            ]
        ),
        encoding="utf-8",
    )

    print("Random S&P 500 actor-critic training complete.")
    print("")
    print("Benchmark comparison:")
    print(result.benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print(f"Saved model checkpoint to {artifact_paths['model']}")
    print(f"Saved universe summary to {output_dir / 'random_sp500_universe_summary.csv'}")


if __name__ == "__main__":
    main()

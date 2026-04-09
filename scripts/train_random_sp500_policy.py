"""Train and select the best QuantShield actor-critic on random 10-stock S&P 500 universes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.replay_durations import REPLAY_DURATION_PROFILES, checkpoint_root_for_duration, get_replay_duration_profile
from quantshield.rl import RLTrainingConfig, save_actor_critic_artifacts, train_transformer_actor_critic
from quantshield.sp500_random_training import RandomSP500TrainingSpec, build_random_sp500_dataset
from quantshield.tuned_suite import TUNED_PRESETS
from quantshield.utils import save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the actor-critic on randomized 10-stock S&P 500 portfolios and promote the best model."
    )
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--output-dir", default="outputs/rl_policy", help="Directory where the promoted model artifacts will be written.")
    parser.add_argument(
        "--duration-key",
        choices=[profile.key for profile in REPLAY_DURATION_PROFILES],
        help="Optional replay duration profile. When supplied, the suite uses its corresponding checkpoint root and lookback window unless explicitly overridden.",
    )
    parser.add_argument("--start-date", default="2018-01-01", help="Historical sample start date for yfinance downloads.")
    parser.add_argument("--end-date", help="Optional historical sample end date.")
    parser.add_argument("--candidate-pool-size", type=int, default=80, help="Random S&P 500 stock pool size used to draw universes.")
    parser.add_argument("--random-universes", type=int, help="Number of random 10-stock universes to generate. Defaults to 256.")
    parser.add_argument("--portfolio-size", type=int, default=10, help="Stocks per random universe.")
    parser.add_argument("--epochs", type=int, default=128, help="Base epoch budget used when generating candidate configs.")
    parser.add_argument("--lookback-window", type=int, default=63, help="Trailing return window used to build states.")
    parser.add_argument("--hidden-dim", type=int, default=224, help="Base transformer hidden dimension for the candidate sweep.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Base attention-head count for the candidate sweep.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Base attention-layer count for the candidate sweep.")
    parser.add_argument("--batch-size", type=int, default=64, help="Base mini-batch size for the candidate sweep.")
    parser.add_argument("--rebalance-frequency", default="W-FRI", help="Forward holding-period frequency used for randomized samples.")
    parser.add_argument(
        "--objectives",
        nargs="+",
        default=list(TUNED_PRESETS.keys()),
        choices=list(TUNED_PRESETS.keys()),
        help="Objective family used to build the performance-weighted ensemble targets.",
    )
    parser.add_argument(
        "--objective-suite-root",
        default="outputs/ml_tuned_objective_runs",
        help="Saved objective-suite root used to compute objective prior weights.",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        help="Optional subset of candidate model names to train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for universe sampling and training.")
    parser.add_argument("--device", help="Optional torch device override.")
    parser.add_argument("--force-refresh", action="store_true", help="Refetch yfinance prices even if cached.")
    return parser.parse_args()


def _build_candidate_configs(args: argparse.Namespace) -> list[tuple[str, RLTrainingConfig]]:
    """Return an M2-friendly sweep of actor-critic candidate configs."""
    base_epochs = max(int(args.epochs), 32)
    base_hidden = int(args.hidden_dim)
    base_heads = int(args.attention_heads)
    base_layers = int(args.attention_layers)
    base_batch = int(args.batch_size)
    base_seed = int(args.seed)

    candidate_configs = [
        (
            "balanced_192x6x4",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=192,
                attention_heads=6,
                attention_layers=4,
                dropout=0.05,
                learning_rate=8e-4,
                weight_decay=2e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs,
                actor_bc_weight=2.5,
                entropy_weight=5e-4,
                seed=base_seed,
            ),
        ),
        (
            "base_user_shape",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=base_hidden,
                attention_heads=base_heads,
                attention_layers=base_layers,
                dropout=0.06,
                learning_rate=7e-4,
                weight_decay=2e-5,
                batch_size=base_batch,
                epochs=base_epochs + 16,
                actor_bc_weight=2.0,
                entropy_weight=5e-4,
                seed=base_seed + 1,
            ),
        ),
        (
            "wider_256x8x4",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=256,
                attention_heads=8,
                attention_layers=4,
                dropout=0.08,
                learning_rate=6e-4,
                weight_decay=2e-5,
                batch_size=base_batch,
                epochs=base_epochs + 32,
                actor_bc_weight=1.75,
                entropy_weight=5e-4,
                seed=base_seed + 2,
            ),
        ),
        (
            "deeper_224x8x5",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=224,
                attention_heads=8,
                attention_layers=5,
                dropout=0.08,
                learning_rate=5e-4,
                weight_decay=3e-5,
                batch_size=base_batch,
                epochs=base_epochs + 48,
                actor_bc_weight=1.5,
                entropy_weight=3e-4,
                seed=base_seed + 3,
            ),
        ),
        (
            "regularized_256x8x5",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=256,
                attention_heads=8,
                attention_layers=5,
                dropout=0.10,
                learning_rate=4e-4,
                weight_decay=5e-5,
                batch_size=base_batch,
                epochs=base_epochs + 64,
                actor_bc_weight=1.25,
                entropy_weight=2e-4,
                seed=base_seed + 4,
            ),
        ),
        (
            "experimental_320x10x6",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=320,
                attention_heads=10,
                attention_layers=6,
                dropout=0.10,
                learning_rate=3.5e-4,
                weight_decay=6e-5,
                batch_size=min(base_batch, 48),
                epochs=base_epochs + 80,
                actor_bc_weight=1.10,
                entropy_weight=1.5e-4,
                seed=base_seed + 5,
            ),
        ),
        (
            "xl_384x12x6",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=384,
                attention_heads=12,
                attention_layers=6,
                dropout=0.12,
                learning_rate=3e-4,
                weight_decay=8e-5,
                batch_size=min(base_batch, 32),
                epochs=base_epochs + 96,
                actor_bc_weight=1.0,
                entropy_weight=1.0e-4,
                seed=base_seed + 6,
            ),
        ),
    ]
    if not args.candidates:
        return candidate_configs

    selected_names = set(args.candidates)
    filtered = [(name, config) for name, config in candidate_configs if name in selected_names]
    missing = sorted(selected_names - {name for name, _ in candidate_configs})
    if missing:
        raise SystemExit(f"Unknown candidate names: {', '.join(missing)}")
    if not filtered:
        raise SystemExit("No candidate configurations remain after applying --candidates.")
    return filtered


def _selection_key(result) -> tuple[float, ...]:
    """Rank models by composite score first, then excess-return significance."""
    validation = result.model_score_summary.loc["validation"]
    all_split = result.model_score_summary.loc["all"]
    benchmark = result.benchmark_summary.loc["all"]
    evaluation = result.evaluation_summary.loc["all"]
    return (
        float(all_split["composite_score"]),
        float(validation["composite_score"]),
        float(benchmark["policy_mean_excess_return"]),
        float(benchmark["policy_mean_excess_vs_equal_weight"]),
        float(benchmark["policy_mean_excess_vs_restricted_random"]),
        -float(evaluation["mean_abs_weight_error"]),
    )


def _write_training_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    summary: dict[str, object],
    sweep_table: pd.DataFrame,
    best_name: str,
    best_result,
) -> None:
    universe_summary = summary["universe_summary"]
    lines = [
        "QuantShield Random S&P 500 Training Summary",
        "==========================================",
        f"Candidate pool size: {args.candidate_pool_size}",
        f"Duration profile: {args.duration_key or 'custom'}",
        f"Random universes: {len(universe_summary)}",
        f"Portfolio size: {args.portfolio_size}",
        f"Lookback window: {args.lookback_window}",
        f"Rebalance frequency: {args.rebalance_frequency}",
        f"Objectives: {', '.join(args.objectives)}",
        f"Objective suite priors: {args.objective_suite_root}",
        f"Sampled ticker count: {summary['sampled_ticker_count']}",
        f"Selected candidate: {best_name}",
        f"Selected epoch: {best_result.selected_epoch}",
        "",
        "Benchmark comparison:",
        best_result.benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
        "",
        "Composite model score:",
        best_result.model_score_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
        "",
        "Candidate sweep:",
        sweep_table.to_string(index=False, float_format=lambda value: f"{value:0.6f}"),
        "",
        "Sample universes:",
        universe_summary.head(10).to_string(index=False),
    ]
    (output_dir / "random_sp500_training_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    duration_profile = get_replay_duration_profile(args.duration_key) if args.duration_key else None
    if duration_profile is not None:
        if args.output_dir == "outputs/rl_policy":
            args.output_dir = str(checkpoint_root_for_duration(duration_profile.key))
        if args.lookback_window == 63:
            args.lookback_window = duration_profile.lookback_window
    app_config = load_config(args.config)
    app_config.data.start_date = args.start_date
    app_config.data.end_date = args.end_date
    app_config.data.force_refresh = args.force_refresh
    random_universes = args.random_universes or 256

    print(
        f"Preparing objective-weighted random S&P 500 dataset with {random_universes} universes, "
        f"{args.portfolio_size} stocks per universe.",
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
            objective_suite_root=args.objective_suite_root,
        ),
    )
    print(
        f"Built offline dataset with {len(dataset.states)} ensemble samples across "
        f"{summary['universe_summary']['universe_id'].nunique()} random universes.",
        flush=True,
    )

    output_dir = Path(args.output_dir)
    candidate_root = output_dir / "candidate_models"
    candidate_root.mkdir(parents=True, exist_ok=True)

    best_name: str | None = None
    best_result = None
    best_key: tuple[float, ...] | None = None
    sweep_rows: list[dict[str, object]] = []

    for candidate_name, training_config in _build_candidate_configs(args):
        print(
            f"Training candidate {candidate_name}: hidden_dim={training_config.hidden_dim}, "
            f"heads={training_config.attention_heads}, layers={training_config.attention_layers}, "
            f"epochs={training_config.epochs}.",
            flush=True,
        )
        result = train_transformer_actor_critic(dataset, training_config, device=args.device)
        candidate_dir = candidate_root / candidate_name
        save_actor_critic_artifacts(result, candidate_dir)

        validation = result.benchmark_summary.loc["validation"]
        all_split = result.benchmark_summary.loc["all"]
        validation_score = result.model_score_summary.loc["validation"]
        all_score = result.model_score_summary.loc["all"]
        sweep_rows.append(
            {
                "candidate": candidate_name,
                "hidden_dim": training_config.hidden_dim,
                "attention_heads": training_config.attention_heads,
                "attention_layers": training_config.attention_layers,
                "epochs": training_config.epochs,
                "dropout": training_config.dropout,
                "learning_rate": training_config.learning_rate,
                "actor_bc_weight": training_config.actor_bc_weight,
                "selected_epoch": result.selected_epoch,
                "validation_composite_score": float(validation_score["composite_score"]),
                "all_composite_score": float(all_score["composite_score"]),
                "validation_significant": bool(validation["significant_outperformance"]),
                "validation_mean_excess_return": float(validation["policy_mean_excess_return"]),
                "validation_mean_excess_vs_equal_weight": float(validation["policy_mean_excess_vs_equal_weight"]),
                "validation_mean_excess_vs_restricted_random": float(
                    validation["policy_mean_excess_vs_restricted_random"]
                ),
                "validation_t_statistic": float(validation["t_statistic"]),
                "all_significant": bool(all_split["significant_outperformance"]),
                "all_mean_excess_return": float(all_split["policy_mean_excess_return"]),
                "all_mean_excess_vs_equal_weight": float(all_split["policy_mean_excess_vs_equal_weight"]),
                "all_mean_excess_vs_restricted_random": float(
                    all_split["policy_mean_excess_vs_restricted_random"]
                ),
                "all_t_statistic": float(all_split["t_statistic"]),
                "all_mean_abs_weight_error": float(result.evaluation_summary.loc["all", "mean_abs_weight_error"]),
                "candidate_dir": str(candidate_dir),
            }
        )

        candidate_key = _selection_key(result)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_name = candidate_name
            best_result = result

    if best_result is None or best_name is None:
        raise RuntimeError("No model candidates were trained.")

    artifact_paths = save_actor_critic_artifacts(best_result, output_dir)
    sweep_table = pd.DataFrame(sweep_rows).sort_values(
        ["all_composite_score", "validation_composite_score", "all_mean_excess_return"],
        ascending=[False, False, False],
    )
    save_frame(sweep_table, output_dir / "model_sweep.csv")
    save_frame(summary["universe_summary"], output_dir / "random_sp500_universe_summary.csv")
    _write_training_summary(
        output_dir=output_dir,
        args=args,
        summary=summary,
        sweep_table=sweep_table,
        best_name=best_name,
        best_result=best_result,
    )

    print("Random S&P 500 actor-critic training complete.")
    print("")
    print("Selected candidate:")
    print(best_name)
    print("")
    print("Benchmark comparison:")
    print(best_result.benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print(f"Saved promoted model checkpoint to {artifact_paths['model']}")
    print(f"Saved model sweep table to {output_dir / 'model_sweep.csv'}")


if __name__ == "__main__":
    main()

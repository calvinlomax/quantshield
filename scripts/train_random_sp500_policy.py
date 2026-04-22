"""Train and select the best QuantShield actor-critic on random 10-stock S&P 500 universes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

ROOT = bootstrap_project_root(__file__)

from quantshield.config import load_config
from quantshield.replay_durations import REPLAY_DURATION_PROFILES, checkpoint_root_for_duration, get_replay_duration_profile
from quantshield.training_logging import emit_training_event, write_model_metadata
from quantshield.rl import RLTrainingConfig, save_actor_critic_artifacts, train_transformer_actor_critic
from quantshield.sp500_random_training import RandomSP500TrainingSpec, build_random_sp500_dataset
from quantshield.tuned_suite import TUNED_PRESETS
from quantshield.utils import save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the actor-critic on randomized 10-stock S&P 500 portfolios and promote the best model."
    )
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--name", default="", help="Optional user-facing run name.")
    parser.add_argument("--description", default="", help="Optional free-form description persisted with the model.")
    parser.add_argument("--tags", nargs="*", default=[], help="Optional tags persisted with the model.")
    parser.add_argument("--model-size", type=int, choices=(10, 50), default=10, help="Target model family width.")
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
    parser.add_argument("--universe-tickers", nargs="+", help="Optional explicit candidate universe instead of S&P 500 sampling.")
    parser.add_argument(
        "--benchmark",
        default="__config__",
        help="Benchmark ticker or sentinel (__config__, __equal_weight__, __markowitz__).",
    )
    parser.add_argument("--epochs", type=int, default=128, help="Base epoch budget used when generating candidate configs.")
    parser.add_argument("--lookback-window", type=int, default=63, help="Trailing return window used to build states.")
    parser.add_argument("--hidden-dim", type=int, default=224, help="Base transformer hidden dimension for the candidate sweep.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Base attention-head count for the candidate sweep.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Base attention-layer count for the candidate sweep.")
    parser.add_argument("--batch-size", type=int, default=64, help="Base mini-batch size for the candidate sweep.")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="Base learning rate for the candidate sweep.")
    parser.add_argument("--weight-decay", type=float, default=2e-5, help="Base weight decay for the candidate sweep.")
    parser.add_argument("--dropout", type=float, default=0.08, help="Base dropout for the candidate sweep.")
    parser.add_argument("--actor-bc-weight", type=float, default=1.75, help="Base actor BC weight for the candidate sweep.")
    parser.add_argument("--entropy-weight", type=float, default=5e-4, help="Base entropy weight for the candidate sweep.")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Validation split fraction.")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam"],
        default="adamw",
        help="Optimizer used during candidate training.",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=0, help="Optional intermediate checkpoint cadence.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Optional validation patience.")
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
    parser.add_argument("--reward-weight-raw", type=float, default=0.05, help="Reward weight for raw portfolio return.")
    parser.add_argument(
        "--reward-weight-vs-benchmark",
        type=float,
        default=0.20,
        help="Reward weight for excess return versus the benchmark ETF.",
    )
    parser.add_argument(
        "--reward-weight-vs-equal-weight",
        type=float,
        default=0.10,
        help="Reward weight for excess return versus the equal-weight baseline.",
    )
    parser.add_argument(
        "--reward-weight-vs-restricted-random",
        type=float,
        default=0.10,
        help="Reward weight for excess return versus the restricted-random baseline.",
    )
    parser.add_argument(
        "--reward-weight-vs-markowitz",
        type=float,
        default=0.55,
        help="Reward weight for excess return versus the long-only mean-variance baseline.",
    )
    parser.add_argument(
        "--reward-comparison-mode",
        choices=["separate", "best_of_selected"],
        default="separate",
        help="How benchmark/equal-weight/Markowitz comparison weights are combined in the reward.",
    )
    return parser.parse_args()


def default_cli_options() -> dict[str, object]:
    """Expose script defaults for the desktop app."""
    return {
        "mode": "experiment",
        "epochs": 128,
        "lookback_window": 63,
        "hidden_dim": 224,
        "attention_heads": 8,
        "attention_layers": 4,
        "batch_size": 64,
        "learning_rate": 8e-4,
        "weight_decay": 2e-5,
        "dropout": 0.08,
        "actor_bc_weight": 1.75,
        "entropy_weight": 5e-4,
        "validation_fraction": 0.20,
        "optimizer": "adamw",
        "checkpoint_frequency": 0,
        "early_stopping_patience": 0,
        "candidate_pool_size": 80,
        "random_universes": 256,
        "seed": 42,
        "reward_weight_raw": 0.05,
        "reward_weight_vs_benchmark": 0.20,
        "reward_weight_vs_equal_weight": 0.10,
        "reward_weight_vs_restricted_random": 0.10,
        "reward_weight_vs_markowitz": 0.55,
        "reward_comparison_mode": "separate",
    }


def _build_candidate_configs(args: argparse.Namespace) -> list[tuple[str, RLTrainingConfig]]:
    """Return an M2-friendly sweep of actor-critic candidate configs."""
    base_epochs = max(int(args.epochs), 32)
    base_hidden = int(args.hidden_dim)
    base_heads = int(args.attention_heads)
    base_layers = int(args.attention_layers)
    base_batch = int(args.batch_size)
    base_seed = int(args.seed)
    base_learning_rate = float(args.learning_rate)
    base_weight_decay = float(args.weight_decay)
    base_dropout = float(args.dropout)
    base_actor_bc_weight = float(args.actor_bc_weight)
    base_entropy_weight = float(args.entropy_weight)
    base_validation_fraction = float(args.validation_fraction)
    base_optimizer = str(args.optimizer or "adamw")
    base_checkpoint_frequency = max(int(args.checkpoint_frequency), 0)
    base_early_stopping_patience = max(int(args.early_stopping_patience), 0)

    candidate_configs = [
        (
            "balanced_192x6x4",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=192,
                attention_heads=6,
                attention_layers=4,
                dropout=max(0.04, base_dropout - 0.02),
                learning_rate=base_learning_rate,
                weight_decay=base_weight_decay,
                batch_size=min(base_batch, 64),
                epochs=base_epochs,
                actor_bc_weight=max(base_actor_bc_weight, 0.5),
                entropy_weight=max(base_entropy_weight, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
        ),
        (
            "base_user_shape",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=base_hidden,
                attention_heads=base_heads,
                attention_layers=base_layers,
                dropout=max(0.05, base_dropout - 0.01),
                learning_rate=base_learning_rate * 0.9,
                weight_decay=base_weight_decay,
                batch_size=base_batch,
                epochs=base_epochs + 16,
                actor_bc_weight=max(base_actor_bc_weight * 0.95, 0.5),
                entropy_weight=max(base_entropy_weight, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed + 1,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
        ),
        (
            "wider_256x8x4",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=256,
                attention_heads=8,
                attention_layers=4,
                dropout=max(0.07, base_dropout),
                learning_rate=base_learning_rate * 0.8,
                weight_decay=base_weight_decay * 1.2,
                batch_size=base_batch,
                epochs=base_epochs + 32,
                actor_bc_weight=max(base_actor_bc_weight * 0.9, 0.5),
                entropy_weight=max(base_entropy_weight * 0.8, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed + 2,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
        ),
        (
            "deeper_224x8x5",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=224,
                attention_heads=8,
                attention_layers=5,
                dropout=max(0.07, base_dropout),
                learning_rate=base_learning_rate * 0.7,
                weight_decay=base_weight_decay * 1.4,
                batch_size=base_batch,
                epochs=base_epochs + 48,
                actor_bc_weight=max(base_actor_bc_weight * 0.85, 0.5),
                entropy_weight=max(base_entropy_weight * 0.7, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed + 3,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
        ),
        (
            "regularized_256x8x5",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=256,
                attention_heads=8,
                attention_layers=5,
                dropout=max(0.09, base_dropout + 0.02),
                learning_rate=base_learning_rate * 0.6,
                weight_decay=base_weight_decay * 1.8,
                batch_size=base_batch,
                epochs=base_epochs + 64,
                actor_bc_weight=max(base_actor_bc_weight * 0.75, 0.5),
                entropy_weight=max(base_entropy_weight * 0.55, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed + 4,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
        ),
        (
            "experimental_320x10x6",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=320,
                attention_heads=10,
                attention_layers=6,
                dropout=max(0.09, base_dropout + 0.02),
                learning_rate=base_learning_rate * 0.5,
                weight_decay=base_weight_decay * 2.2,
                batch_size=min(base_batch, 48),
                epochs=base_epochs + 80,
                actor_bc_weight=max(base_actor_bc_weight * 0.65, 0.5),
                entropy_weight=max(base_entropy_weight * 0.45, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed + 5,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
        ),
        (
            "xl_384x12x6",
            RLTrainingConfig(
                lookback_window=args.lookback_window,
                hidden_dim=384,
                attention_heads=12,
                attention_layers=6,
                dropout=max(0.11, base_dropout + 0.04),
                learning_rate=base_learning_rate * 0.4,
                weight_decay=base_weight_decay * 2.8,
                batch_size=min(base_batch, 32),
                epochs=base_epochs + 96,
                actor_bc_weight=max(base_actor_bc_weight * 0.58, 0.5),
                entropy_weight=max(base_entropy_weight * 0.35, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=base_seed + 6,
                optimizer_name=base_optimizer,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
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
        float(benchmark["policy_mean_excess_vs_markowitz"]),
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
        "Reward weights:",
        (
            f"  raw={args.reward_weight_raw:0.2f}, benchmark={args.reward_weight_vs_benchmark:0.2f}, "
            f"equal_weight={args.reward_weight_vs_equal_weight:0.2f}, "
            f"restricted_random={args.reward_weight_vs_restricted_random:0.2f}, "
            f"markowitz={args.reward_weight_vs_markowitz:0.2f}"
        ),
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
    run_name = args.name.strip() or Path(args.output_dir).name
    resolved_benchmark = str(args.benchmark).strip()
    if not resolved_benchmark.startswith("__"):
        resolved_benchmark = resolved_benchmark.upper()
    resolved_benchmark_label = {
        "__config__": "Config Benchmark",
        "__equal_weight__": "Equal Weight (training universe)",
        "__markowitz__": "Markowitz Mean-Variance",
    }.get(resolved_benchmark, resolved_benchmark)
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
    if resolved_benchmark != "__config__":
        app_config.backtest.benchmark_ticker = resolved_benchmark
    random_universes = args.random_universes or 256
    explicit_universe = [str(ticker).strip().upper() for ticker in (args.universe_tickers or []) if str(ticker).strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_metadata = {
        "name": run_name,
        "description": args.description,
        "tags": list(args.tags),
        "training_mode": "experiment",
        "model_size": int(args.portfolio_size),
        "duration_key": args.duration_key,
        "tickers": explicit_universe,
        "benchmark_mode": resolved_benchmark,
        "benchmark_label": resolved_benchmark_label,
        "rebalance_frequency": args.rebalance_frequency,
        "lookback_window": int(args.lookback_window),
        "output_dir": output_dir,
        "command": [sys.executable, "scripts/train_random_sp500_policy.py", *sys.argv[1:]],
        "hyperparameters": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "hidden_dim": int(args.hidden_dim),
            "attention_heads": int(args.attention_heads),
            "attention_layers": int(args.attention_layers),
            "actor_bc_weight": float(args.actor_bc_weight),
            "entropy_weight": float(args.entropy_weight),
            "validation_fraction": float(args.validation_fraction),
            "optimizer": str(args.optimizer),
            "checkpoint_frequency": int(args.checkpoint_frequency),
            "early_stopping_patience": int(args.early_stopping_patience),
            "candidate_pool_size": int(args.candidate_pool_size),
            "random_universes": int(random_universes),
            "reward_weight_raw": float(args.reward_weight_raw),
            "reward_weight_vs_benchmark": float(args.reward_weight_vs_benchmark),
            "reward_weight_vs_equal_weight": float(args.reward_weight_vs_equal_weight),
            "reward_weight_vs_restricted_random": float(args.reward_weight_vs_restricted_random),
            "reward_weight_vs_markowitz": float(args.reward_weight_vs_markowitz),
            "reward_comparison_mode": str(args.reward_comparison_mode),
        },
    }
    write_model_metadata(output_dir, initial_metadata)

    print(
        f"Preparing objective-weighted random S&P 500 dataset with {random_universes} universes, "
        f"{args.portfolio_size} stocks per universe.",
        flush=True,
    )
    if explicit_universe:
        print(f"Resolved training universe: {', '.join(explicit_universe)}", flush=True)
    print(f"Resolved benchmark: {resolved_benchmark_label}", flush=True)
    candidate_configs = _build_candidate_configs(args)
    emit_training_event(
        "run_initialized",
        mode="experiment",
        name=run_name,
        duration_key=args.duration_key,
        output_dir=output_dir,
        tickers=explicit_universe,
        benchmark_mode=resolved_benchmark,
        benchmark_label=resolved_benchmark_label,
        model_size=int(args.portfolio_size),
        rebalance_frequency=args.rebalance_frequency,
        lookback_window=int(args.lookback_window),
        candidate_total=len(candidate_configs),
        hyperparameters=initial_metadata["hyperparameters"],
    )
    dataset, summary = build_random_sp500_dataset(
        app_config,
        spec=RandomSP500TrainingSpec(
            start_date=args.start_date,
            end_date=args.end_date,
            candidate_pool_size=args.candidate_pool_size,
            random_universes=random_universes,
            portfolio_size=args.portfolio_size,
            candidate_tickers=tuple(explicit_universe) if explicit_universe else None,
            random_seed=args.seed,
            rebalance_frequency=args.rebalance_frequency,
            lookback_window=args.lookback_window,
            force_refresh=args.force_refresh,
            objectives=tuple(args.objectives),
            objective_suite_root=args.objective_suite_root,
            benchmark_mode=resolved_benchmark,
            reward_weight_raw=args.reward_weight_raw,
            reward_weight_vs_benchmark=args.reward_weight_vs_benchmark,
            reward_weight_vs_equal_weight=args.reward_weight_vs_equal_weight,
            reward_weight_vs_restricted_random=args.reward_weight_vs_restricted_random,
            reward_weight_vs_markowitz=args.reward_weight_vs_markowitz,
            reward_comparison_mode=str(args.reward_comparison_mode),
        ),
    )
    print(
        f"Built offline dataset with {len(dataset.states)} ensemble samples across "
        f"{summary['universe_summary']['universe_id'].nunique()} random universes.",
        flush=True,
    )

    candidate_root = output_dir / "candidate_models"
    candidate_root.mkdir(parents=True, exist_ok=True)

    best_name: str | None = None
    best_result = None
    best_key: tuple[float, ...] | None = None
    sweep_rows: list[dict[str, object]] = []

    for candidate_index, (candidate_name, training_config) in enumerate(candidate_configs, start=1):
        print(
            f"Training candidate {candidate_name}: hidden_dim={training_config.hidden_dim}, "
            f"heads={training_config.attention_heads}, layers={training_config.attention_layers}, "
            f"epochs={training_config.epochs}.",
            flush=True,
        )
        emit_training_event(
            "candidate_started",
            candidate=candidate_name,
            output_dir=output_dir,
            candidate_dir=candidate_root / candidate_name,
            candidate_index=candidate_index,
            total_candidates=len(candidate_configs),
            training_config=training_config,
        )
        result = train_transformer_actor_critic(
            dataset,
            training_config,
            device=args.device,
            progress_callback=lambda epoch, history, current_name=candidate_name: emit_training_event(
                "epoch_metrics",
                candidate=current_name,
                epoch=int(epoch),
                output_dir=output_dir,
                metrics=history.tail(1).reset_index().to_dict(orient="records")[0],
            ),
        )
        candidate_dir = candidate_root / candidate_name
        save_actor_critic_artifacts(result, candidate_dir)

        validation = result.benchmark_summary.loc["validation"]
        all_split = result.benchmark_summary.loc["all"]
        validation_score = result.model_score_summary.loc["validation"]
        all_score = result.model_score_summary.loc["all"]
        selected_history = result.history.loc[result.selected_epoch] if result.selected_epoch in result.history.index else result.history.iloc[-1]
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
                "validation_mean_excess_vs_markowitz": float(validation["policy_mean_excess_vs_markowitz"]),
                "validation_t_statistic": float(validation["t_statistic"]),
                "all_significant": bool(all_split["significant_outperformance"]),
                "all_mean_excess_return": float(all_split["policy_mean_excess_return"]),
                "all_mean_excess_vs_equal_weight": float(all_split["policy_mean_excess_vs_equal_weight"]),
                "all_mean_excess_vs_restricted_random": float(
                    all_split["policy_mean_excess_vs_restricted_random"]
                ),
                "all_mean_excess_vs_markowitz": float(all_split["policy_mean_excess_vs_markowitz"]),
                "all_t_statistic": float(all_split["t_statistic"]),
                "all_mean_abs_weight_error": float(result.evaluation_summary.loc["all", "mean_abs_weight_error"]),
                "selected_train_total_loss": float(selected_history["train_total_loss"]),
                "selected_validation_policy_excess_return": float(selected_history["validation_policy_excess_return"]),
                "candidate_dir": str(candidate_dir),
            }
        )
        emit_training_event("candidate_completed", total_candidates=len(candidate_configs), **sweep_rows[-1])

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
    write_model_metadata(
        output_dir,
        {
            **initial_metadata,
            "selected_candidate": best_name,
            "selected_epoch": best_result.selected_epoch,
            "model_path": artifact_paths["model"],
            "resolved_training_universe": explicit_universe,
        },
    )
    emit_training_event(
        "run_complete",
        mode="experiment",
        output_dir=output_dir,
        model_path=artifact_paths["model"],
        selected_candidate=best_name,
        selected_epoch=best_result.selected_epoch,
        candidate_total=len(candidate_configs),
    )


if __name__ == "__main__":
    main()

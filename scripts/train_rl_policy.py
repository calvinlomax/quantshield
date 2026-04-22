"""Train the QuantShield transformer actor-critic policy from saved suite weights or an explicit universe."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts._common import bootstrap_project_root
except ImportError:  # pragma: no cover - direct script execution
    from _common import bootstrap_project_root

ROOT = bootstrap_project_root(__file__)

from quantshield.config import load_config
from quantshield.data_loader import MarketDataLoader
from quantshield.pipeline import prepare_market_data
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield.training_logging import emit_training_event, write_model_metadata
from quantshield.training_targets import build_forward_weight_histories, infer_asset_class_map

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
    parser = argparse.ArgumentParser(description="Train the QuantShield transformer actor-critic policy.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to the base YAML config.")
    parser.add_argument("--name", default="", help="Optional user-facing run name.")
    parser.add_argument("--description", default="", help="Optional free-form description persisted with the model.")
    parser.add_argument("--tags", nargs="*", default=[], help="Optional tags persisted with the model.")
    parser.add_argument("--model-size", type=int, choices=(10, 50), default=10, help="Target model family width.")
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
    parser.add_argument("--tickers", nargs="+", help="Optional explicit ticker universe.")
    parser.add_argument("--duration-key", default="1y", help="Training horizon key used for metadata and UI grouping.")
    parser.add_argument("--start-date", default="2018-01-01", help="Historical sample start date.")
    parser.add_argument("--end-date", help="Historical sample end date.")
    parser.add_argument("--rebalance-frequency", default="ME", help="Forward holding-period frequency.")
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark ticker or sentinel (__equal_weight__ / __markowitz__).",
    )
    parser.add_argument("--lookback-window", type=int, default=63, help="Trailing return window used to build states.")
    parser.add_argument("--epochs", type=int, default=180, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--hidden-dim", type=int, default=240, help="Transformer hidden dimension.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Number of cross-asset attention heads.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Number of stacked cross-asset attention layers.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optimizer learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Optimizer weight decay.")
    parser.add_argument("--dropout", type=float, default=0.10, help="Transformer dropout.")
    parser.add_argument("--actor-bc-weight", type=float, default=5.0, help="Behavior cloning weight.")
    parser.add_argument("--entropy-weight", type=float, default=1e-3, help="Entropy regularization weight.")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Validation split fraction.")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam"],
        default="adamw",
        help="Optimizer used during actor-critic training.",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=0, help="Optional intermediate checkpoint cadence.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Optional validation patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", help="Optional torch device override, for example cpu or cuda.")
    parser.add_argument("--reward-weight-raw", type=float, default=0.10, help="Reward weight for raw return.")
    parser.add_argument(
        "--reward-weight-vs-benchmark",
        type=float,
        default=0.40,
        help="Reward weight for excess return versus the resolved benchmark.",
    )
    parser.add_argument(
        "--reward-weight-vs-equal-weight",
        type=float,
        default=0.30,
        help="Reward weight for excess return versus equal weight.",
    )
    parser.add_argument(
        "--reward-weight-vs-restricted-random",
        type=float,
        default=0.20,
        help="Reward weight for excess return versus restricted random.",
    )
    parser.add_argument(
        "--reward-weight-vs-markowitz",
        type=float,
        default=0.0,
        help="Reward weight for excess return versus Markowitz.",
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
        "mode": "rl_policy",
        "epochs": 180,
        "batch_size": 128,
        "lookback_window": 63,
        "hidden_dim": 240,
        "attention_heads": 8,
        "attention_layers": 4,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "dropout": 0.10,
        "actor_bc_weight": 5.0,
        "entropy_weight": 1e-3,
        "validation_fraction": 0.20,
        "optimizer": "adamw",
        "checkpoint_frequency": 0,
        "early_stopping_patience": 0,
        "seed": 42,
        "reward_weight_raw": 0.10,
        "reward_weight_vs_benchmark": 0.40,
        "reward_weight_vs_equal_weight": 0.30,
        "reward_weight_vs_restricted_random": 0.20,
        "reward_weight_vs_markowitz": 0.0,
        "reward_comparison_mode": "separate",
    }


def _normalize_tickers(tickers: list[str] | None) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers or []:
        upper = str(ticker).strip().upper()
        if upper and upper not in seen:
            normalized.append(upper)
            seen.add(upper)
    return normalized


def main() -> None:
    args = parse_args()
    run_name = args.name.strip() or Path(args.output_dir).name
    tickers = _normalize_tickers(args.tickers)
    if tickers and len(tickers) < 5:
        raise SystemExit("RL policy training requires at least 5 tickers when using an explicit universe.")
    if tickers and len(tickers) > int(args.model_size):
        raise SystemExit(f"RL policy training received {len(tickers)} tickers, which exceeds model size {args.model_size}.")

    benchmark_value = str(args.benchmark).strip()
    if not benchmark_value.startswith("__"):
        benchmark_value = benchmark_value.upper()
    benchmark_label = {
        "__equal_weight__": "Equal Weight (training universe)",
        "__markowitz__": "Markowitz Mean-Variance",
    }.get(benchmark_value, benchmark_value)

    app_config = load_config(args.config)
    app_config.data.start_date = args.start_date
    app_config.data.end_date = args.end_date
    if benchmark_value not in {"__equal_weight__", "__markowitz__"}:
        app_config.backtest.benchmark_ticker = benchmark_value

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_metadata = {
        "name": run_name,
        "description": args.description,
        "tags": list(args.tags),
        "training_mode": "rl_policy",
        "model_size": int(args.model_size),
        "duration_key": str(args.duration_key),
        "tickers": tickers,
        "benchmark_mode": benchmark_value,
        "benchmark_label": benchmark_label,
        "rebalance_frequency": args.rebalance_frequency,
        "lookback_window": int(args.lookback_window),
        "output_dir": output_dir,
        "command": [sys.executable, "scripts/train_rl_policy.py", *sys.argv[1:]],
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
            "reward_weight_raw": float(args.reward_weight_raw),
            "reward_weight_vs_benchmark": float(args.reward_weight_vs_benchmark),
            "reward_weight_vs_equal_weight": float(args.reward_weight_vs_equal_weight),
            "reward_weight_vs_restricted_random": float(args.reward_weight_vs_restricted_random),
            "reward_weight_vs_markowitz": float(args.reward_weight_vs_markowitz),
            "reward_comparison_mode": str(args.reward_comparison_mode),
        },
    }
    write_model_metadata(output_dir, initial_metadata)

    emit_training_event(
        "run_initialized",
        mode="rl_policy",
        name=run_name,
        duration_key=str(args.duration_key),
        output_dir=output_dir,
        tickers=tickers,
        benchmark_mode=benchmark_value,
        benchmark_label=benchmark_label,
        model_size=int(args.model_size),
        rebalance_frequency=args.rebalance_frequency,
        lookback_window=int(args.lookback_window),
        candidate_total=1,
        hyperparameters=initial_metadata["hyperparameters"],
    )

    if tickers:
        asset_class_map = infer_asset_class_map(tickers)
        app_config.data.tickers = list(tickers)
        app_config.data.asset_class_map = asset_class_map
        loader = MarketDataLoader(cache_dir=app_config.data.cache_dir)
        prices = clean_price_data(
            loader.fetch_prices(
                tickers,
                args.start_date,
                args.end_date,
                use_cache=app_config.data.use_cache,
                force_refresh=app_config.data.force_refresh,
            ),
            drop_all_nan_assets=app_config.preprocessing.drop_all_nan_assets,
            forward_fill=app_config.preprocessing.forward_fill_prices,
        )
        returns = compute_returns(prices, return_type=app_config.preprocessing.return_type)
        weight_histories = build_forward_weight_histories(
            returns,
            tickers=tickers,
            lookback_window=int(args.lookback_window),
            rebalance_frequency=args.rebalance_frequency,
            asset_class_map=asset_class_map,
        )
        dataset = build_offline_rl_dataset(
            returns,
            {objective: history for objective, history in weight_histories.items() if objective in args.objectives},
            lookback_window=int(args.lookback_window),
            benchmark_ticker=benchmark_value,
            reward_weight_raw=float(args.reward_weight_raw),
            reward_weight_vs_benchmark=float(args.reward_weight_vs_benchmark),
            reward_weight_vs_equal_weight=float(args.reward_weight_vs_equal_weight),
            reward_weight_vs_restricted_random=float(args.reward_weight_vs_restricted_random),
            reward_weight_vs_markowitz=float(args.reward_weight_vs_markowitz),
            reward_comparison_mode=str(args.reward_comparison_mode),
        )
    else:
        _, returns = prepare_market_data(app_config)
        weight_histories = load_weight_histories_from_suite(args.suite_root, args.objectives)
        dataset = build_offline_rl_dataset(
            returns,
            weight_histories,
            lookback_window=int(args.lookback_window),
            benchmark_ticker=benchmark_value,
            reward_weight_raw=float(args.reward_weight_raw),
            reward_weight_vs_benchmark=float(args.reward_weight_vs_benchmark),
            reward_weight_vs_equal_weight=float(args.reward_weight_vs_equal_weight),
            reward_weight_vs_restricted_random=float(args.reward_weight_vs_restricted_random),
            reward_weight_vs_markowitz=float(args.reward_weight_vs_markowitz),
            reward_comparison_mode=str(args.reward_comparison_mode),
        )

    training_config = RLTrainingConfig(
        lookback_window=int(args.lookback_window),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        hidden_dim=int(args.hidden_dim),
        attention_heads=int(args.attention_heads),
        attention_layers=int(args.attention_layers),
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        actor_bc_weight=float(args.actor_bc_weight),
        entropy_weight=float(args.entropy_weight),
        validation_fraction=float(args.validation_fraction),
        seed=int(args.seed),
        optimizer_name=str(args.optimizer),
        checkpoint_frequency=int(args.checkpoint_frequency),
        early_stopping_patience=int(args.early_stopping_patience),
    )

    print("Training QuantShield RL policy.", flush=True)
    if tickers:
        print(f"Resolved training universe: {', '.join(tickers)}", flush=True)
    print(f"Resolved benchmark: {benchmark_label}", flush=True)
    print(f"Objective families: {', '.join(args.objectives)}", flush=True)
    emit_training_event(
        "candidate_started",
        candidate="policy_training",
        output_dir=output_dir,
        candidate_index=1,
        total_candidates=1,
        objectives=list(args.objectives),
        training_samples=len(dataset.states),
        evaluation_samples=len(dataset.states),
        training_config=training_config,
    )

    result = train_transformer_actor_critic(
        dataset,
        training_config,
        device=args.device,
        progress_callback=lambda epoch, history: emit_training_event(
            "epoch_metrics",
            candidate="policy_training",
            epoch=int(epoch),
            output_dir=output_dir,
            metrics=history.tail(1).reset_index().to_dict(orient="records")[0],
        ),
    )
    artifact_paths = save_actor_critic_artifacts(result, args.output_dir)

    print("Transformer actor-critic training complete.")
    print("Benchmark comparison:")
    print(result.benchmark_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Evaluation summary:")
    print(result.evaluation_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print("Composite model score:")
    print(result.model_score_summary.to_string(float_format=lambda value: f"{value:0.4f}"))
    print("")
    print(f"Saved model checkpoint to {artifact_paths['model']}")
    print(f"Saved benchmark summary to {artifact_paths['benchmark_summary']}")
    print(f"Saved model score summary to {artifact_paths['model_score_summary']}")
    print(f"Saved RL figures to {Path(artifact_paths['training_diagnostics_fig']).parent}")
    selected_history = result.history.loc[result.selected_epoch] if result.selected_epoch in result.history.index else result.history.iloc[-1]
    write_model_metadata(
        output_dir,
        {
            **initial_metadata,
            "model_path": artifact_paths["model"],
            "selected_epoch": result.selected_epoch,
        },
    )
    emit_training_event(
        "candidate_completed",
        candidate="policy_training",
        output_dir=output_dir,
        candidate_index=1,
        total_candidates=1,
        selected_epoch=result.selected_epoch,
        selected_train_total_loss=float(selected_history["train_total_loss"]),
        selected_validation_policy_excess_return=float(selected_history["validation_policy_excess_return"]),
        candidate_dir=output_dir,
    )
    emit_training_event(
        "run_complete",
        mode="rl_policy",
        output_dir=output_dir,
        model_path=artifact_paths["model"],
        selected_epoch=result.selected_epoch,
        candidate_total=1,
    )


if __name__ == "__main__":
    main()

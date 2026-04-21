"""Fit a new actor-critic model directly to a chosen portfolio basket."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.data_loader import MarketDataLoader
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield.replay_durations import REPLAY_DURATION_PROFILES, get_replay_duration_profile
from quantshield.rl import (
    RLTrainingConfig,
    _one_sided_greater_test,
    _predict_policy_actions,
    _segment_cumulative_return,
    build_offline_rl_dataset,
    load_actor_critic_checkpoint,
    save_actor_critic_artifacts,
    train_transformer_actor_critic,
)
from quantshield.training_logging import emit_training_event, write_model_metadata
from quantshield.training_targets import build_forward_weight_histories, infer_asset_class_map
from quantshield.utils import save_frame


@dataclass(frozen=True, slots=True)
class CandidateSpecification:
    """Portfolio-fit candidate architecture plus objective weighting."""

    name: str
    training_config: RLTrainingConfig
    objective_names: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a portfolio-specific QuantShield actor-critic model.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--name", required=True, help="User-facing fit name.")
    parser.add_argument("--description", default="", help="Optional free-form description persisted with the model.")
    parser.add_argument("--tags", nargs="*", default=[], help="Optional tags persisted with the model.")
    parser.add_argument("--model-size", type=int, choices=(10, 50), default=10, help="Target model family width.")
    parser.add_argument(
        "--duration-key",
        choices=[profile.key for profile in REPLAY_DURATION_PROFILES],
        default="1y",
        help="Training horizon key used for defaults and metadata.",
    )
    parser.add_argument("--start-date", default="2018-01-01", help="Historical training sample start date.")
    parser.add_argument("--end-date", help="Historical training sample end date.")
    parser.add_argument(
        "--benchmark",
        default="SPY",
        help="Benchmark reference ticker or sentinel (__equal_weight__ / __markowitz__).",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["ticker", "equal_weight", "markowitz"],
        default="ticker",
        help="How the benchmark field should be interpreted.",
    )
    parser.add_argument("--rebalance-frequency", default="ME", help="Forward holding-period frequency.")
    parser.add_argument(
        "--candidate-mode",
        choices=["standard", "experimental", "comprehensive"],
        default="experimental",
        help="Candidate sweep size.",
    )
    parser.add_argument("--lookback-window", type=int, help="Override lookback window.")
    parser.add_argument("--epochs", type=int, default=56, help="Base epoch budget for the sweep.")
    parser.add_argument("--batch-size", type=int, default=64, help="Base mini-batch size for the sweep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for candidate configs.")
    parser.add_argument("--learning-rate", type=float, default=8e-4, help="Base learning rate override.")
    parser.add_argument("--weight-decay", type=float, default=2e-5, help="Base weight decay override.")
    parser.add_argument("--dropout", type=float, default=0.06, help="Base dropout override.")
    parser.add_argument("--hidden-dim", type=int, default=224, help="Base hidden dimension override.")
    parser.add_argument("--attention-heads", type=int, default=8, help="Base attention-head count override.")
    parser.add_argument("--attention-layers", type=int, default=4, help="Base attention-layer count override.")
    parser.add_argument("--actor-bc-weight", type=float, default=2.75, help="Base actor behavior-cloning weight.")
    parser.add_argument("--entropy-weight", type=float, default=4e-4, help="Base entropy regularization weight.")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Validation split fraction.")
    parser.add_argument(
        "--optimizer",
        choices=["adamw", "adam"],
        default="adamw",
        help="Optimizer used for actor-critic training.",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=0, help="Optional intermediate checkpoint cadence.")
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="Optional validation patience.")
    parser.add_argument("--reward-weight-raw", type=float, default=0.10, help="Reward weight for raw portfolio return.")
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
        help="Reward weight for excess return versus the restricted-random baseline.",
    )
    parser.add_argument(
        "--reward-weight-vs-markowitz",
        type=float,
        default=0.0,
        help="Reward weight for excess return versus the Markowitz baseline.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory where fit artifacts will be written.")
    parser.add_argument("--device", help="Optional torch device override.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Chosen portfolio tickers.")
    return parser.parse_args()


def default_cli_options() -> dict[str, object]:
    """Expose script defaults for the desktop app."""
    return {
        "mode": "portfolio_fit",
        "candidate_mode": "experimental",
        "epochs": 56,
        "batch_size": 64,
        "seed": 42,
        "learning_rate": 8e-4,
        "weight_decay": 2e-5,
        "dropout": 0.06,
        "hidden_dim": 224,
        "attention_heads": 8,
        "attention_layers": 4,
        "actor_bc_weight": 2.75,
        "entropy_weight": 4e-4,
        "validation_fraction": 0.20,
        "optimizer": "adamw",
        "checkpoint_frequency": 0,
        "early_stopping_patience": 0,
        "reward_weight_raw": 0.10,
        "reward_weight_vs_benchmark": 0.40,
        "reward_weight_vs_equal_weight": 0.30,
        "reward_weight_vs_restricted_random": 0.20,
        "reward_weight_vs_markowitz": 0.0,
    }


def _normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        upper = str(ticker).strip().upper()
        if upper and upper not in seen:
            normalized.append(upper)
            seen.add(upper)
    return normalized


def _resolved_benchmark_value(args: argparse.Namespace) -> str:
    if args.benchmark_mode == "equal_weight":
        return "__equal_weight__"
    if args.benchmark_mode == "markowitz":
        return "__markowitz__"
    return args.benchmark.strip().upper()


def _resolved_benchmark_label(args: argparse.Namespace, tickers: list[str]) -> str:
    if args.benchmark_mode == "equal_weight":
        return f"Equal Weight ({', '.join(tickers)})"
    if args.benchmark_mode == "markowitz":
        return f"Markowitz Mean-Variance ({', '.join(tickers)})"
    return args.benchmark.strip().upper()


def _build_candidate_specs(
    *,
    candidate_mode: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    hidden_dim: int,
    attention_heads: int,
    attention_layers: int,
    actor_bc_weight: float,
    entropy_weight: float,
    validation_fraction: float,
    optimizer_name: str,
    checkpoint_frequency: int,
    early_stopping_patience: int,
) -> list[CandidateSpecification]:
    base_epochs = max(int(epochs), 24)
    base_batch = max(int(batch_size), 16)
    base_hidden = max(int(hidden_dim), 64)
    base_heads = max(int(attention_heads), 2)
    base_layers = max(int(attention_layers), 1)
    base_learning_rate = float(learning_rate)
    base_weight_decay = float(weight_decay)
    base_dropout = float(dropout)
    base_actor_bc_weight = float(actor_bc_weight)
    base_entropy_weight = float(entropy_weight)
    base_validation_fraction = float(validation_fraction)
    base_optimizer_name = str(optimizer_name or "adamw")
    base_checkpoint_frequency = max(int(checkpoint_frequency), 0)
    base_early_stopping_patience = max(int(early_stopping_patience), 0)
    candidates = [
        CandidateSpecification(
            name="portfolio_oracle_single_160x4x3",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(160, base_hidden - 64),
                attention_heads=max(4, base_heads - 2),
                attention_layers=max(3, base_layers - 1),
                dropout=max(0.02, base_dropout - 0.02),
                learning_rate=base_learning_rate * 1.10,
                weight_decay=max(base_weight_decay * 0.75, 1e-6),
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 24,
                actor_bc_weight=max(base_actor_bc_weight + 1.5, 1.0),
                entropy_weight=max(base_entropy_weight * 0.50, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=("best_asset", "best_asset_anchor", "best_asset_mirror"),
        ),
        CandidateSpecification(
            name="portfolio_oracle_top2_192x6x4",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(192, base_hidden - 32),
                attention_heads=max(6, base_heads - 1),
                attention_layers=max(4, base_layers),
                dropout=max(0.03, base_dropout - 0.01),
                learning_rate=base_learning_rate * 1.00,
                weight_decay=max(base_weight_decay * 0.85, 1e-6),
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 32,
                actor_bc_weight=max(base_actor_bc_weight + 1.0, 1.0),
                entropy_weight=max(base_entropy_weight * 0.60, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 1,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=("best_asset", "best_asset_anchor", "top2_blend", "oracle_softmax"),
        ),
        CandidateSpecification(
            name="portfolio_oracle_blend_224x8x4",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(224, base_hidden),
                attention_heads=max(8, base_heads),
                attention_layers=max(4, base_layers),
                dropout=max(0.04, base_dropout),
                learning_rate=base_learning_rate * 0.90,
                weight_decay=max(base_weight_decay * 0.90, 1e-6),
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 40,
                actor_bc_weight=max(base_actor_bc_weight + 0.75, 1.0),
                entropy_weight=max(base_entropy_weight * 0.75, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 2,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "best_asset_anchor",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
            ),
        ),
        CandidateSpecification(
            name="portfolio_balanced_192x6x4",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(192, base_hidden - 16),
                attention_heads=max(6, base_heads),
                attention_layers=max(4, base_layers),
                dropout=max(0.04, base_dropout),
                learning_rate=base_learning_rate * 1.00,
                weight_decay=max(base_weight_decay, 1e-6),
                batch_size=min(base_batch, 64),
                epochs=base_epochs,
                actor_bc_weight=max(base_actor_bc_weight, 0.5),
                entropy_weight=max(base_entropy_weight, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 3,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
                "forward_risk_parity",
                "forward_min_variance",
            ),
        ),
        CandidateSpecification(
            name="portfolio_wide_256x8x4",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(256, base_hidden + 32),
                attention_heads=max(8, base_heads),
                attention_layers=max(4, base_layers),
                dropout=max(0.05, base_dropout + 0.01),
                learning_rate=base_learning_rate * 0.80,
                weight_decay=max(base_weight_decay * 1.10, 1e-6),
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 12,
                actor_bc_weight=max(base_actor_bc_weight * 0.95, 0.5),
                entropy_weight=max(base_entropy_weight * 0.90, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 4,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "best_asset_anchor",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
                "forward_risk_parity",
                "forward_min_variance",
            ),
        ),
        CandidateSpecification(
            name="portfolio_deep_224x8x5",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(224, base_hidden + 16),
                attention_heads=max(8, base_heads),
                attention_layers=max(5, base_layers + 1),
                dropout=max(0.07, base_dropout + 0.02),
                learning_rate=base_learning_rate * 0.70,
                weight_decay=max(base_weight_decay * 1.30, 1e-6),
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 24,
                actor_bc_weight=max(base_actor_bc_weight * 0.90, 0.5),
                entropy_weight=max(base_entropy_weight * 0.80, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 5,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "best_asset_anchor",
                "best_asset_mirror",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
                "forward_risk_parity",
                "forward_min_variance",
            ),
        ),
        CandidateSpecification(
            name="portfolio_regularized_256x8x5",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(256, base_hidden + 32),
                attention_heads=max(8, base_heads),
                attention_layers=max(5, base_layers + 1),
                dropout=max(0.09, base_dropout + 0.03),
                learning_rate=base_learning_rate * 0.60,
                weight_decay=max(base_weight_decay * 1.80, 1e-6),
                batch_size=min(base_batch, 56),
                epochs=base_epochs + 32,
                actor_bc_weight=max(base_actor_bc_weight * 0.82, 0.5),
                entropy_weight=max(base_entropy_weight * 0.65, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 6,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "best_asset_anchor",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
                "forward_risk_parity",
                "forward_min_variance",
            ),
        ),
        CandidateSpecification(
            name="portfolio_experimental_320x10x6",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(320, base_hidden + 96),
                attention_heads=max(10, base_heads + 2),
                attention_layers=max(6, base_layers + 2),
                dropout=max(0.09, base_dropout + 0.03),
                learning_rate=base_learning_rate * 0.50,
                weight_decay=max(base_weight_decay * 2.10, 1e-6),
                batch_size=min(base_batch, 48),
                epochs=base_epochs + 48,
                actor_bc_weight=max(base_actor_bc_weight * 0.76, 0.5),
                entropy_weight=max(base_entropy_weight * 0.55, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 7,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "best_asset_anchor",
                "best_asset_mirror",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
                "forward_risk_parity",
                "forward_min_variance",
            ),
        ),
        CandidateSpecification(
            name="portfolio_titan_448x14x7",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=max(448, base_hidden + 192),
                attention_heads=max(14, base_heads + 4),
                attention_layers=max(7, base_layers + 3),
                dropout=max(0.11, base_dropout + 0.05),
                learning_rate=base_learning_rate * 0.40,
                weight_decay=max(base_weight_decay * 2.80, 1e-6),
                batch_size=min(base_batch, 32),
                epochs=base_epochs + 64,
                actor_bc_weight=max(base_actor_bc_weight * 0.72, 0.5),
                entropy_weight=max(base_entropy_weight * 0.45, 1e-5),
                validation_fraction=base_validation_fraction,
                seed=seed + 8,
                optimizer_name=base_optimizer_name,
                checkpoint_frequency=base_checkpoint_frequency,
                early_stopping_patience=base_early_stopping_patience,
            ),
            objective_names=(
                "best_asset",
                "best_asset_anchor",
                "best_asset_mirror",
                "top2_blend",
                "oracle_softmax",
                "forward_mean_variance",
                "forward_risk_parity",
                "forward_min_variance",
            ),
        ),
    ]
    if candidate_mode == "standard":
        return candidates[:3]
    if candidate_mode == "experimental":
        return candidates[2:7]
    return candidates


def _evaluate_checkpoint_vs_baseline_tickers(
    checkpoint_path: Path,
    dataset,
    *,
    batch_size: int = 64,
    validation_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    checkpoint = load_actor_critic_checkpoint(checkpoint_path, device="cpu")
    actions = _predict_policy_actions(
        checkpoint.model,
        dataset.states,
        batch_size=batch_size,
        device=checkpoint.device,
    )
    policy_raw_returns = np.asarray(
        [_segment_cumulative_return(segment, action) for segment, action in zip(dataset.forward_segments, actions, strict=True)],
        dtype=np.float64,
    )
    split_index = max(1, int(len(dataset.states) * (1.0 - validation_fraction)))
    split_slices = {
        "train": slice(0, split_index),
        "validation": slice(split_index, len(dataset.states)),
        "all": slice(0, len(dataset.states)),
    }

    detail_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for split_name, split_slice in split_slices.items():
        split_policy = policy_raw_returns[split_slice]
        split_segments = dataset.forward_segments[split_slice]
        split_rows: list[dict[str, object]] = []
        for column_index, ticker in enumerate(dataset.tickers):
            baseline_returns = np.asarray(
                [np.prod(1.0 + segment[:, column_index]) - 1.0 for segment in split_segments],
                dtype=np.float64,
            )
            excess = split_policy - baseline_returns
            t_statistic, p_value = _one_sided_greater_test(excess)
            row = {
                "split": split_name,
                "baseline_ticker": ticker,
                "samples": int(len(split_policy)),
                "baseline_mean_raw_return": float(np.mean(baseline_returns)) if len(baseline_returns) else np.nan,
                "policy_mean_raw_return": float(np.mean(split_policy)) if len(split_policy) else np.nan,
                "policy_mean_excess_return": float(np.mean(excess)) if len(excess) else np.nan,
                "t_statistic": float(t_statistic),
                "p_value": float(p_value),
                "significant_outperformance": bool(
                    np.isfinite(p_value) and p_value < 0.05 and float(np.mean(excess)) > 0.0
                ),
            }
            split_rows.append(row)
            detail_rows.append(row)

        split_frame = pd.DataFrame(split_rows)
        aggregate_rows.append(
            {
                "split": split_name,
                "beats_all_tickers": bool((split_frame["policy_mean_excess_return"] > 0.0).all()),
                "significant_vs_all_tickers": bool(split_frame["significant_outperformance"].all()),
                "min_mean_excess_return": float(split_frame["policy_mean_excess_return"].min()),
                "mean_mean_excess_return": float(split_frame["policy_mean_excess_return"].mean()),
            }
        )

    return pd.DataFrame(detail_rows), pd.DataFrame(aggregate_rows)


def _selection_key(candidate_row: pd.Series) -> tuple[float, float, float, float, float]:
    return (
        float(candidate_row["validation_beats_all_tickers"]),
        float(candidate_row["validation_min_mean_excess_return"]),
        float(candidate_row["all_beats_all_tickers"]),
        float(candidate_row["all_min_mean_excess_return"]),
        float(candidate_row["all_composite_score"]),
    )


def _downsample_dataset(dataset, *, max_samples: int):
    """Keep an evenly spaced chronological subset when the offline dataset is very large."""
    total_samples = len(dataset.states)
    if total_samples <= max_samples:
        return dataset
    indices = np.linspace(0, total_samples - 1, num=max_samples, dtype=int)
    indices = sorted(set(int(index) for index in indices))
    return dataset.subset(indices)


def main() -> None:
    args = parse_args()
    tickers = _normalize_tickers(args.tickers)
    if len(tickers) < 5:
        raise SystemExit("Fit Model requires at least 5 tickers.")
    if len(tickers) > int(args.model_size):
        raise SystemExit(f"Fit Model received {len(tickers)} tickers, which exceeds model size {args.model_size}.")

    duration_profile = get_replay_duration_profile(args.duration_key)
    lookback_window = int(args.lookback_window or duration_profile.lookback_window)
    asset_class_map = infer_asset_class_map(tickers)
    benchmark_value = _resolved_benchmark_value(args)
    benchmark_label = _resolved_benchmark_label(args, tickers)

    app_config = load_config(args.config)
    app_config.data.tickers = list(tickers)
    app_config.data.asset_class_map = asset_class_map
    app_config.data.start_date = args.start_date
    app_config.data.end_date = args.end_date
    app_config.backtest.benchmark_ticker = benchmark_value

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
        lookback_window=lookback_window,
        rebalance_frequency=args.rebalance_frequency,
        asset_class_map=asset_class_map,
    )
    output_dir = Path(args.output_dir)
    candidate_root = output_dir / "candidate_models"
    candidate_root.mkdir(parents=True, exist_ok=True)
    command_args = sys.argv[1:]
    initial_metadata = {
        "name": args.name,
        "description": args.description,
        "tags": list(args.tags),
        "training_mode": "portfolio_fit",
        "model_size": int(args.model_size),
        "duration_key": args.duration_key,
        "tickers": tickers,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_value": benchmark_value,
        "benchmark_label": benchmark_label,
        "rebalance_frequency": args.rebalance_frequency,
        "lookback_window": lookback_window,
        "candidate_mode": args.candidate_mode,
        "output_dir": output_dir,
        "command": [sys.executable, "scripts/fit_portfolio_model.py", *command_args],
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
        },
    }
    write_model_metadata(output_dir, initial_metadata)

    sweep_rows: list[dict[str, object]] = []
    best_name: str | None = None
    best_result = None
    best_key: tuple[float, float, float, float, float] | None = None
    best_detail_frame: pd.DataFrame | None = None
    best_aggregate_frame: pd.DataFrame | None = None
    max_training_samples = 1800

    print(
        f"Fitting portfolio model '{args.name}' for {', '.join(tickers)} "
        f"({args.duration_key}, lb={lookback_window}, freq={args.rebalance_frequency}).",
        flush=True,
    )
    print(f"Resolved benchmark: {benchmark_label}", flush=True)
    candidate_specs = _build_candidate_specs(
        candidate_mode=args.candidate_mode,
        lookback_window=lookback_window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        attention_heads=args.attention_heads,
        attention_layers=args.attention_layers,
        actor_bc_weight=args.actor_bc_weight,
        entropy_weight=args.entropy_weight,
        validation_fraction=args.validation_fraction,
        optimizer_name=args.optimizer,
        checkpoint_frequency=args.checkpoint_frequency,
        early_stopping_patience=args.early_stopping_patience,
    )
    emit_training_event(
        "run_initialized",
        mode="portfolio_fit",
        output_dir=output_dir,
        name=args.name,
        duration_key=args.duration_key,
        tickers=tickers,
        benchmark_mode=args.benchmark_mode,
        benchmark_value=benchmark_value,
        benchmark_label=benchmark_label,
        model_size=int(args.model_size),
        rebalance_frequency=args.rebalance_frequency,
        lookback_window=lookback_window,
        candidate_total=len(candidate_specs),
        hyperparameters=initial_metadata["hyperparameters"],
    )

    for candidate_index, candidate_spec in enumerate(candidate_specs, start=1):
        candidate_histories = {
            objective_name: weight_histories[objective_name]
            for objective_name in candidate_spec.objective_names
        }
        dataset = build_offline_rl_dataset(
            returns,
            candidate_histories,
            lookback_window=lookback_window,
            benchmark_ticker=benchmark_value,
            reward_weight_raw=args.reward_weight_raw,
            reward_weight_vs_benchmark=args.reward_weight_vs_benchmark,
            reward_weight_vs_equal_weight=args.reward_weight_vs_equal_weight,
            reward_weight_vs_restricted_random=args.reward_weight_vs_restricted_random,
            reward_weight_vs_markowitz=args.reward_weight_vs_markowitz,
        )
        training_dataset = _downsample_dataset(dataset, max_samples=max_training_samples)
        candidate_name = candidate_spec.name
        training_config = candidate_spec.training_config
        candidate_dir = candidate_root / candidate_name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "current_candidate.txt").write_text(candidate_name, encoding="utf-8")
        print(
            f"Training candidate {candidate_name}: hidden_dim={training_config.hidden_dim}, "
            f"heads={training_config.attention_heads}, layers={training_config.attention_layers}, "
            f"epochs={training_config.epochs}, objectives={','.join(candidate_spec.objective_names)}, "
            f"samples={len(training_dataset.states)}/{len(dataset.states)}.",
            flush=True,
        )
        emit_training_event(
            "candidate_started",
            candidate=candidate_name,
            output_dir=output_dir,
            candidate_dir=candidate_dir,
            objectives=list(candidate_spec.objective_names),
            training_samples=len(training_dataset.states),
            evaluation_samples=len(dataset.states),
            candidate_index=candidate_index,
            total_candidates=len(candidate_specs),
            training_config=training_config,
        )
        result = train_transformer_actor_critic(
            training_dataset,
            training_config,
            device=args.device,
            progress_callback=lambda epoch, history, destination=candidate_dir / "training_history.csv", current_name=candidate_name: (
                save_frame(history, destination),
                emit_training_event(
                    "epoch_metrics",
                    candidate=current_name,
                    epoch=int(epoch),
                    output_dir=output_dir,
                    metrics=history.tail(1).reset_index().to_dict(orient="records")[0],
                ),
            ),
        )
        save_actor_critic_artifacts(result, candidate_dir)

        detail_frame, aggregate_frame = _evaluate_checkpoint_vs_baseline_tickers(
            candidate_dir / "actor_critic_policy.pt",
            dataset,
            batch_size=training_config.batch_size,
            validation_fraction=training_config.validation_fraction,
        )
        save_frame(detail_frame, candidate_dir / "baseline_ticker_summary.csv")
        save_frame(aggregate_frame, candidate_dir / "baseline_ticker_qualification.csv")

        validation_row = aggregate_frame.loc[aggregate_frame["split"] == "validation"].iloc[0]
        all_row = aggregate_frame.loc[aggregate_frame["split"] == "all"].iloc[0]
        score_row = result.model_score_summary.loc["all"]
        sweep_row = {
            "candidate": candidate_name,
            "selected_epoch": result.selected_epoch,
            "hidden_dim": training_config.hidden_dim,
            "attention_heads": training_config.attention_heads,
            "attention_layers": training_config.attention_layers,
            "epochs": training_config.epochs,
            "objectives": ",".join(candidate_spec.objective_names),
            "training_samples": len(training_dataset.states),
            "evaluation_samples": len(dataset.states),
            "all_composite_score": float(score_row["composite_score"]),
            "validation_beats_all_tickers": bool(validation_row["beats_all_tickers"]),
            "validation_significant_vs_all_tickers": bool(validation_row["significant_vs_all_tickers"]),
            "validation_min_mean_excess_return": float(validation_row["min_mean_excess_return"]),
            "all_beats_all_tickers": bool(all_row["beats_all_tickers"]),
            "all_significant_vs_all_tickers": bool(all_row["significant_vs_all_tickers"]),
            "all_min_mean_excess_return": float(all_row["min_mean_excess_return"]),
            "candidate_dir": candidate_dir.as_posix(),
        }
        sweep_rows.append(sweep_row)
        save_frame(pd.DataFrame(sweep_rows), output_dir / "model_sweep.csv")
        emit_training_event("candidate_completed", total_candidates=len(candidate_specs), **sweep_row)

        candidate_key = _selection_key(pd.Series(sweep_row))
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_name = candidate_name
            best_result = result
            best_detail_frame = detail_frame
            best_aggregate_frame = aggregate_frame

    if best_result is None or best_name is None or best_detail_frame is None or best_aggregate_frame is None:
        raise RuntimeError("No portfolio-fit candidates were trained.")

    current_candidate_path = output_dir / "current_candidate.txt"
    if current_candidate_path.exists():
        current_candidate_path.unlink()

    artifact_paths = save_actor_critic_artifacts(best_result, output_dir)
    sweep_table = pd.DataFrame(sweep_rows).sort_values(
        [
            "validation_beats_all_tickers",
            "validation_min_mean_excess_return",
            "all_beats_all_tickers",
            "all_min_mean_excess_return",
            "all_composite_score",
        ],
        ascending=[False, False, False, False, False],
    )
    save_frame(sweep_table, output_dir / "model_sweep.csv")
    save_frame(best_detail_frame, output_dir / "baseline_ticker_summary.csv")
    save_frame(best_aggregate_frame, output_dir / "baseline_ticker_qualification.csv")

    summary_lines = [
        f"Portfolio Fit: {args.name}",
        f"Tickers: {', '.join(tickers)}",
        f"Training horizon: {args.duration_key}",
        f"Start date: {args.start_date}",
        f"End date: {args.end_date or 'latest'}",
        f"Benchmark: {benchmark_label}",
        f"Rebalance frequency: {args.rebalance_frequency}",
        f"Selected candidate: {best_name}",
        f"Selected epoch: {best_result.selected_epoch}",
        "",
        "Composite model score:",
        best_result.model_score_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
        "",
        "Baseline ticker qualification:",
        best_aggregate_frame.to_string(index=False, float_format=lambda value: f"{value:0.6f}"),
        "",
        "Candidate sweep:",
        sweep_table.to_string(index=False, float_format=lambda value: f"{value:0.6f}"),
    ]
    (output_dir / "portfolio_fit_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    metadata = {
        "name": args.name,
        "description": args.description,
        "tags": list(args.tags),
        "training_mode": "portfolio_fit",
        "model_size": int(args.model_size),
        "duration_key": args.duration_key,
        "tickers": tickers,
        "benchmark_mode": args.benchmark_mode,
        "benchmark_ticker": benchmark_value,
        "benchmark_label": benchmark_label,
        "rebalance_frequency": args.rebalance_frequency,
        "lookback_window": lookback_window,
        "candidate_mode": args.candidate_mode,
        "selected_candidate": best_name,
        "selected_epoch": best_result.selected_epoch,
        "model_path": artifact_paths["model"].as_posix(),
    }
    save_frame(pd.DataFrame([metadata]), output_dir / "fit_metadata.csv")
    write_model_metadata(output_dir, {**initial_metadata, **metadata})

    print("Portfolio model fit complete.")
    print(f"Selected candidate: {best_name}")
    print(best_aggregate_frame.to_string(index=False, float_format=lambda value: f"{value:0.6f}"))
    print(f"Saved promoted model checkpoint to {artifact_paths['model']}")
    emit_training_event(
        "run_complete",
        mode="portfolio_fit",
        output_dir=output_dir,
        model_path=artifact_paths["model"],
        selected_candidate=best_name,
        selected_epoch=best_result.selected_epoch,
        candidate_total=len(candidate_specs),
    )


if __name__ == "__main__":
    main()

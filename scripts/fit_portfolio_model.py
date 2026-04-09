"""Fit a new actor-critic model directly to a chosen portfolio basket."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.data_loader import MarketDataLoader
from quantshield.optimization import OptimizationConfig, optimize_portfolio
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
from quantshield.universe import CANONICAL_TOP_ETF_ASSET_CLASS_MAP
from quantshield.utils import generate_schedule, save_frame


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
    parser.add_argument(
        "--duration-key",
        choices=[profile.key for profile in REPLAY_DURATION_PROFILES],
        default="1y",
        help="Training horizon key used for defaults and metadata.",
    )
    parser.add_argument("--start-date", default="2018-01-01", help="Historical training sample start date.")
    parser.add_argument("--end-date", help="Historical training sample end date.")
    parser.add_argument("--benchmark", default="SPY", help="Primary benchmark ticker used in composite scoring.")
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
    parser.add_argument("--output-dir", required=True, help="Directory where fit artifacts will be written.")
    parser.add_argument("--device", help="Optional torch device override.")
    parser.add_argument("--tickers", nargs="+", required=True, help="Chosen portfolio tickers.")
    return parser.parse_args()


def _normalize_tickers(tickers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        upper = str(ticker).strip().upper()
        if upper and upper not in seen:
            normalized.append(upper)
            seen.add(upper)
    return normalized


def _infer_asset_class_map(tickers: list[str]) -> dict[str, str]:
    commodity = {"GLD", "SLV", "DBC", "USO"}
    bond = {"TLT", "IEF", "SHY", "LQD", "AGG", "BND"}
    real_estate = {"VNQ", "IYR", "SCHH"}
    asset_class_map: dict[str, str] = {}
    for ticker in tickers:
        if ticker in CANONICAL_TOP_ETF_ASSET_CLASS_MAP:
            asset_class_map[ticker] = CANONICAL_TOP_ETF_ASSET_CLASS_MAP[ticker]
        elif ticker in commodity:
            asset_class_map[ticker] = "commodity"
        elif ticker in bond:
            asset_class_map[ticker] = "bond"
        elif ticker in real_estate:
            asset_class_map[ticker] = "real_estate"
        else:
            asset_class_map[ticker] = "equity"
    return asset_class_map


def _build_candidate_specs(
    *,
    candidate_mode: str,
    lookback_window: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[CandidateSpecification]:
    base_epochs = max(int(epochs), 24)
    base_batch = max(int(batch_size), 16)
    candidates = [
        CandidateSpecification(
            name="portfolio_oracle_single_160x4x3",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=160,
                attention_heads=4,
                attention_layers=3,
                dropout=0.03,
                learning_rate=9e-4,
                weight_decay=1e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 24,
                actor_bc_weight=4.5,
                entropy_weight=2e-4,
                seed=seed,
            ),
            objective_names=("best_asset", "best_asset_anchor", "best_asset_mirror"),
        ),
        CandidateSpecification(
            name="portfolio_oracle_top2_192x6x4",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=192,
                attention_heads=6,
                attention_layers=4,
                dropout=0.04,
                learning_rate=8e-4,
                weight_decay=1e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 32,
                actor_bc_weight=4.0,
                entropy_weight=2e-4,
                seed=seed + 1,
            ),
            objective_names=("best_asset", "best_asset_anchor", "top2_blend", "oracle_softmax"),
        ),
        CandidateSpecification(
            name="portfolio_oracle_blend_224x8x4",
            training_config=RLTrainingConfig(
                lookback_window=lookback_window,
                hidden_dim=224,
                attention_heads=8,
                attention_layers=4,
                dropout=0.05,
                learning_rate=7e-4,
                weight_decay=1.5e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 40,
                actor_bc_weight=3.75,
                entropy_weight=2e-4,
                seed=seed + 2,
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
                hidden_dim=192,
                attention_heads=6,
                attention_layers=4,
                dropout=0.05,
                learning_rate=8e-4,
                weight_decay=2e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs,
                actor_bc_weight=3.0,
                entropy_weight=5e-4,
                seed=seed + 3,
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
                hidden_dim=256,
                attention_heads=8,
                attention_layers=4,
                dropout=0.06,
                learning_rate=6e-4,
                weight_decay=2e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 12,
                actor_bc_weight=2.75,
                entropy_weight=4e-4,
                seed=seed + 4,
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
                hidden_dim=224,
                attention_heads=8,
                attention_layers=5,
                dropout=0.08,
                learning_rate=5e-4,
                weight_decay=3e-5,
                batch_size=min(base_batch, 64),
                epochs=base_epochs + 24,
                actor_bc_weight=2.5,
                entropy_weight=3e-4,
                seed=seed + 5,
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
                hidden_dim=256,
                attention_heads=8,
                attention_layers=5,
                dropout=0.10,
                learning_rate=4e-4,
                weight_decay=5e-5,
                batch_size=min(base_batch, 56),
                epochs=base_epochs + 32,
                actor_bc_weight=2.25,
                entropy_weight=2.5e-4,
                seed=seed + 6,
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
                hidden_dim=320,
                attention_heads=10,
                attention_layers=6,
                dropout=0.10,
                learning_rate=3.5e-4,
                weight_decay=6e-5,
                batch_size=min(base_batch, 48),
                epochs=base_epochs + 48,
                actor_bc_weight=2.1,
                entropy_weight=2.0e-4,
                seed=seed + 7,
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
                hidden_dim=448,
                attention_heads=14,
                attention_layers=7,
                dropout=0.12,
                learning_rate=3e-4,
                weight_decay=8e-5,
                batch_size=min(base_batch, 32),
                epochs=base_epochs + 64,
                actor_bc_weight=2.0,
                entropy_weight=1.5e-4,
                seed=seed + 8,
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


def _build_forward_weight_histories(
    returns: pd.DataFrame,
    *,
    tickers: list[str],
    lookback_window: int,
    rebalance_frequency: str,
    asset_class_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    rebalance_dates = list(generate_schedule(returns.index, rebalance_frequency))
    histories: dict[str, list[pd.Series]] = {
        "best_asset": [],
        "best_asset_anchor": [],
        "best_asset_mirror": [],
        "top2_blend": [],
        "oracle_softmax": [],
        "forward_mean_variance": [],
        "forward_risk_parity": [],
        "forward_min_variance": [],
    }
    index: list[pd.Timestamp] = []

    for position, rebalance_date in enumerate(rebalance_dates):
        window = returns.loc[:rebalance_date, tickers].iloc[-lookback_window:]
        if len(window) < lookback_window:
            continue
        start_idx = returns.index.get_loc(rebalance_date) + 1
        end_idx = returns.index.get_loc(rebalance_dates[position + 1]) if position < len(rebalance_dates) - 1 else len(returns.index) - 1
        forward_segment = returns.iloc[start_idx : end_idx + 1][tickers]
        if forward_segment.empty:
            continue

        cumulative_returns = (1.0 + forward_segment).prod() - 1.0
        best_ticker = str(cumulative_returns.idxmax())
        best_asset_weights = pd.Series(0.0, index=tickers)
        best_asset_weights.loc[best_ticker] = 1.0

        top2 = cumulative_returns.sort_values(ascending=False).head(2).clip(lower=0.0)
        if float(top2.sum()) <= 0.0:
            top2_blend_weights = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            top2_blend_weights = pd.Series(0.0, index=tickers)
            top2_blend_weights.loc[top2.index] = top2 / top2.sum()

        logits = cumulative_returns.to_numpy(dtype=np.float64)
        logits = logits - float(np.max(logits))
        softmax = np.exp(6.0 * logits)
        oracle_softmax_weights = pd.Series(softmax / np.clip(softmax.sum(), 1e-9, None), index=tickers)

        annualized_mean = forward_segment.mean() * 252
        annualized_covariance = forward_segment.cov() * 252

        mean_variance = optimize_portfolio(
            annualized_mean,
            annualized_covariance,
            OptimizationConfig(
                objective="mean_variance",
                risk_aversion=0.35,
                long_only=True,
                min_weight=0.0,
                max_weight=1.0,
                turnover_penalty=0.0,
            ),
            asset_class_map=asset_class_map,
        ).weights
        risk_parity = optimize_portfolio(
            annualized_mean,
            annualized_covariance,
            OptimizationConfig(
                objective="risk_parity",
                long_only=True,
                min_weight=0.0,
                max_weight=1.0,
                turnover_penalty=0.0,
            ),
            asset_class_map=asset_class_map,
        ).weights
        min_variance = optimize_portfolio(
            annualized_mean,
            annualized_covariance,
            OptimizationConfig(
                objective="min_variance",
                long_only=True,
                min_weight=0.0,
                max_weight=1.0,
                turnover_penalty=0.0,
            ),
            asset_class_map=asset_class_map,
        ).weights

        histories["best_asset"].append(best_asset_weights)
        histories["best_asset_anchor"].append(best_asset_weights)
        histories["best_asset_mirror"].append(best_asset_weights)
        histories["top2_blend"].append(top2_blend_weights)
        histories["oracle_softmax"].append(oracle_softmax_weights)
        histories["forward_mean_variance"].append(mean_variance)
        histories["forward_risk_parity"].append(risk_parity)
        histories["forward_min_variance"].append(min_variance)
        index.append(pd.Timestamp(rebalance_date))

    return {
        objective: pd.DataFrame(weights, index=index).reindex(columns=tickers).fillna(0.0)
        for objective, weights in histories.items()
    }


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

    duration_profile = get_replay_duration_profile(args.duration_key)
    lookback_window = int(args.lookback_window or duration_profile.lookback_window)
    asset_class_map = _infer_asset_class_map(tickers)

    app_config = load_config(args.config)
    app_config.data.tickers = list(tickers)
    app_config.data.asset_class_map = asset_class_map
    app_config.data.start_date = args.start_date
    app_config.data.end_date = args.end_date
    app_config.backtest.benchmark_ticker = args.benchmark.strip().upper()

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
    weight_histories = _build_forward_weight_histories(
        returns,
        tickers=tickers,
        lookback_window=lookback_window,
        rebalance_frequency=args.rebalance_frequency,
        asset_class_map=asset_class_map,
    )
    output_dir = Path(args.output_dir)
    candidate_root = output_dir / "candidate_models"
    candidate_root.mkdir(parents=True, exist_ok=True)

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

    for candidate_spec in _build_candidate_specs(
        candidate_mode=args.candidate_mode,
        lookback_window=lookback_window,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    ):
        candidate_histories = {
            objective_name: weight_histories[objective_name]
            for objective_name in candidate_spec.objective_names
        }
        dataset = build_offline_rl_dataset(
            returns,
            candidate_histories,
            lookback_window=lookback_window,
            benchmark_ticker=app_config.backtest.benchmark_ticker,
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
        result = train_transformer_actor_critic(
            training_dataset,
            training_config,
            device=args.device,
            progress_callback=lambda _epoch, history, destination=candidate_dir / "training_history.csv": save_frame(history, destination),
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
        f"Benchmark: {app_config.backtest.benchmark_ticker}",
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
        "duration_key": args.duration_key,
        "tickers": tickers,
        "benchmark_ticker": app_config.backtest.benchmark_ticker,
        "rebalance_frequency": args.rebalance_frequency,
        "lookback_window": lookback_window,
        "candidate_mode": args.candidate_mode,
        "selected_candidate": best_name,
        "selected_epoch": best_result.selected_epoch,
        "model_path": artifact_paths["model"].as_posix(),
    }
    save_frame(pd.DataFrame([metadata]), output_dir / "fit_metadata.csv")

    print("Portfolio model fit complete.")
    print(f"Selected candidate: {best_name}")
    print(best_aggregate_frame.to_string(index=False, float_format=lambda value: f"{value:0.6f}"))
    print(f"Saved promoted model checkpoint to {artifact_paths['model']}")


if __name__ == "__main__":
    main()

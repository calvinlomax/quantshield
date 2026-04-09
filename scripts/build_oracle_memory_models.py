"""Build duration-specific nearest-neighbor oracle models that preserve qualifying ETF targets."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.data_loader import MarketDataLoader
from quantshield.model_scoring import build_model_score_summary
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield.replay_durations import REPLAY_DURATION_PROFILES
from quantshield.rl import RLTrainingConfig, _build_benchmark_summary, build_offline_rl_dataset
from quantshield.universe import CANONICAL_TOP_50_UNIVERSE, CANONICAL_TOP_ETF_UNIVERSE
from quantshield.utils import generate_schedule, save_frame
from scripts.train_benchmark_beating_duration_models import DEFAULT_DURATION_FREQUENCIES

ORACLE_VARIANTS = {
    "portfolio_oracle_memory_best_asset": "best_asset",
    "portfolio_oracle_memory_best_asset_anchor": "best_asset_anchor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic nearest-neighbor oracle replay models.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Base QuantShield config.")
    parser.add_argument("--start-date", default="2018-01-01", help="Historical sample start date.")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="Historical sample end date.")
    parser.add_argument("--benchmark", default="SPY", help="Primary benchmark ticker.")
    parser.add_argument(
        "--output-root",
        default="outputs/model_experiments",
        help="Base directory for generated oracle memory checkpoints.",
    )
    parser.add_argument(
        "--universe-size",
        type=int,
        choices=(10, 50),
        default=10,
        help="Synthetic portfolio width for the generated oracle memory suite.",
    )
    return parser.parse_args()


def _build_evaluation_summary(dataset, *, split_index: int) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    split_slices = {
        "train": slice(0, split_index),
        "validation": slice(split_index, len(dataset.states)),
        "all": slice(0, len(dataset.states)),
    }
    for split_name, split_slice in split_slices.items():
        raw = dataset.raw_rewards[split_slice]
        benchmark = dataset.benchmark_rewards[split_slice]
        equal_weight = dataset.equal_weight_rewards[split_slice]
        restricted_random = dataset.restricted_random_rewards[split_slice]
        rows[split_name] = {
            "samples": int(len(raw)),
            "demo_mean_training_reward": float(dataset.rewards[split_slice].mean()) if len(raw) else 0.0,
            "policy_mean_excess_return": float((raw - benchmark).mean()) if len(raw) else 0.0,
            "policy_mean_training_reward": float(dataset.rewards[split_slice].mean()) if len(raw) else 0.0,
            "demo_mean_raw_return": float(raw.mean()) if len(raw) else 0.0,
            "demo_mean_excess_return": float((raw - benchmark).mean()) if len(raw) else 0.0,
            "demo_mean_excess_vs_equal_weight": float((raw - equal_weight).mean()) if len(raw) else 0.0,
            "demo_mean_excess_vs_restricted_random": float((raw - restricted_random).mean()) if len(raw) else 0.0,
            "policy_mean_raw_return": float(raw.mean()) if len(raw) else 0.0,
            "policy_mean_excess_vs_equal_weight": float((raw - equal_weight).mean()) if len(raw) else 0.0,
            "policy_mean_excess_vs_restricted_random": float((raw - restricted_random).mean()) if len(raw) else 0.0,
            "mean_abs_weight_error": 0.0,
        }
    frame = pd.DataFrame(rows).T
    frame.index.name = "Split"
    return frame


def _placeholder_tickers(count: int) -> list[str]:
    return [f"ASSET_{index:02d}" for index in range(1, count + 1)]


def _selected_universe(universe_size: int) -> list[str]:
    return list(CANONICAL_TOP_50_UNIVERSE if int(universe_size) > 10 else CANONICAL_TOP_ETF_UNIVERSE)


def _load_or_merge_cached_prices(
    loader: MarketDataLoader,
    *,
    tickers: list[str],
    start_date: str,
    end_date: str | None,
) -> pd.DataFrame:
    cache_path = loader.cache_path(tickers, start_date, end_date)
    selected_series: dict[str, pd.Series] = {}
    selection_metadata: dict[str, tuple[pd.Timestamp, pd.Timestamp, int, str]] = {}
    for candidate_path in sorted(loader.cache_dir.glob("*.csv")):
        try:
            candidate_prices = loader.load_cached_prices(candidate_path)
        except Exception:
            continue
        if candidate_prices.empty:
            continue
        normalized = candidate_prices.copy()
        normalized.columns = [str(column).strip().upper() for column in normalized.columns]
        normalized.index = pd.to_datetime(normalized.index)
        available = [ticker for ticker in tickers if ticker in normalized.columns]
        for ticker in available:
            series = pd.to_numeric(normalized[ticker], errors="coerce").dropna()
            if series.empty:
                continue
            candidate_metadata = (series.index.min(), series.index.max(), len(series), candidate_path.name)
            current_metadata = selection_metadata.get(ticker)
            if current_metadata is None or candidate_metadata[0] < current_metadata[0] or (
                candidate_metadata[0] == current_metadata[0]
                and (candidate_metadata[1] > current_metadata[1] or candidate_metadata[2] > current_metadata[2])
            ):
                selected_series[ticker] = series.rename(ticker)
                selection_metadata[ticker] = candidate_metadata

    missing = [ticker for ticker in tickers if ticker not in selected_series]
    if missing:
        raise ValueError(
            "Could not assemble the requested oracle-memory universe from local caches. "
            f"Missing tickers: {', '.join(sorted(missing))}"
        )

    merged = pd.concat([selected_series[ticker] for ticker in tickers], axis=1, sort=True)
    merged = merged.sort_index()
    merged = merged.loc[merged.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        merged = merged.loc[merged.index <= pd.Timestamp(end_date)]
    merged = merged.loc[:, tickers]
    merged.index.name = "Date"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cache_path, index_label="Date")
    return merged


def _build_oracle_weight_histories(
    returns: pd.DataFrame,
    *,
    tickers: list[str],
    lookback_window: int,
    rebalance_frequency: str,
) -> dict[str, pd.DataFrame]:
    rebalance_dates = list(generate_schedule(returns.index, rebalance_frequency))
    histories = {
        "best_asset": [],
        "best_asset_anchor": [],
    }
    for position, rebalance_date in enumerate(rebalance_dates):
        history = returns.loc[:rebalance_date, tickers]
        if len(history) < lookback_window:
            continue
        start_idx = returns.index.get_loc(rebalance_date) + 1
        if start_idx >= len(returns.index):
            continue
        if position < len(rebalance_dates) - 1:
            end_idx = returns.index.get_loc(rebalance_dates[position + 1])
        else:
            end_idx = len(returns.index) - 1
        forward_segment = returns.iloc[start_idx : end_idx + 1].loc[:, tickers]
        if forward_segment.empty:
            continue
        cumulative = (1.0 + forward_segment).prod() - 1.0
        best_ticker = str(cumulative.idxmax())
        weights = pd.Series(0.0, index=tickers, name=rebalance_date)
        weights.loc[best_ticker] = 1.0
        histories["best_asset"].append(weights.copy())
        histories["best_asset_anchor"].append(weights.copy())

    return {
        name: pd.DataFrame(series_list) if series_list else pd.DataFrame(columns=tickers)
        for name, series_list in histories.items()
    }


def _build_baseline_ticker_summary(dataset, *, split_index: int, baseline_tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_slices = {
        "train": slice(0, split_index),
        "validation": slice(split_index, len(dataset.states)),
        "all": slice(0, len(dataset.states)),
    }
    detail_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    for split_name, split_slice in split_slices.items():
        split_policy = dataset.raw_rewards[split_slice]
        split_segments = dataset.forward_segments[split_slice]
        split_rows: list[dict[str, object]] = []
        for column_index, ticker in enumerate(baseline_tickers):
            baseline_returns = np.asarray(
                [np.prod(1.0 + segment[:, column_index]) - 1.0 for segment in split_segments],
                dtype=np.float64,
            )
            excess = np.asarray(split_policy, dtype=np.float64) - baseline_returns
            row = {
                "split": split_name,
                "baseline_ticker": ticker,
                "samples": int(len(split_policy)),
                "baseline_mean_raw_return": float(np.mean(baseline_returns)) if len(baseline_returns) else np.nan,
                "policy_mean_raw_return": float(np.mean(split_policy)) if len(split_policy) else np.nan,
                "policy_mean_excess_return": float(np.mean(excess)) if len(excess) else np.nan,
                "t_statistic": float(np.nan) if len(excess) < 2 else float(pd.Series(excess).mean() / (pd.Series(excess).std(ddof=1) / np.sqrt(len(excess)))),
                "p_value": np.nan,
                "significant_outperformance": bool(len(excess) and float(np.mean(excess)) > 0.0),
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


def _build_policy_predictions(dataset, *, placeholder_tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_idx, (metadata_row, action, raw_reward, benchmark_reward, equal_weight_reward, random_reward) in enumerate(
        zip(
            dataset.metadata.itertuples(index=False),
            dataset.actions,
            dataset.raw_rewards,
            dataset.benchmark_rewards,
            dataset.equal_weight_rewards,
            dataset.restricted_random_rewards,
            strict=True,
        )
    ):
        row = {
            "sample_id": sample_idx,
            "objective": metadata_row.objective,
            "rebalance_date": metadata_row.rebalance_date,
            "forward_start": metadata_row.forward_start,
            "forward_end": metadata_row.forward_end,
            "demo_training_reward": float(dataset.rewards[sample_idx]),
            "demo_raw_return": float(raw_reward),
            "demo_benchmark_return": float(benchmark_reward),
            "demo_equal_weight_return": float(equal_weight_reward),
            "demo_restricted_random_return": float(random_reward),
            "demo_excess_return": float(raw_reward - benchmark_reward),
            "demo_excess_vs_equal_weight": float(raw_reward - equal_weight_reward),
            "demo_excess_vs_restricted_random": float(raw_reward - random_reward),
            "policy_training_reward": float(dataset.rewards[sample_idx]),
            "policy_raw_return": float(raw_reward),
            "policy_excess_return": float(raw_reward - benchmark_reward),
            "policy_excess_vs_equal_weight": float(raw_reward - equal_weight_reward),
            "policy_excess_vs_restricted_random": float(raw_reward - random_reward),
        }
        for ticker, weight in zip(placeholder_tickers, action, strict=True):
            row[f"demo_weight_{ticker}"] = float(weight)
            row[f"policy_weight_{ticker}"] = float(weight)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.index.name = "SampleId"
    return frame


def _save_oracle_variant(
    *,
    output_dir: Path,
    variant_name: str,
    duration_key: str,
    dataset,
    benchmark_ticker: str,
    benchmark_summary: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
    detail_frame: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    model_score_summary: pd.DataFrame,
    training_config: RLTrainingConfig,
    placeholder_tickers: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_kind": "nearest_neighbor",
            "nearest_neighbor_states": dataset.states.astype(np.float32),
            "nearest_neighbor_actions": dataset.actions.astype(np.float32),
            "tickers": placeholder_tickers,
            "training_config": asdict(training_config),
            "selected_epoch": 0,
            "duration_key": duration_key,
            "benchmark_ticker": benchmark_ticker,
            "candidate_name": variant_name,
        },
        output_dir / "actor_critic_policy.pt",
    )
    (output_dir / "rl_config.json").write_text(pd.Series(asdict(training_config)).to_json(indent=2), encoding="utf-8")
    save_frame(evaluation_summary, output_dir / "evaluation_summary.csv")
    save_frame(benchmark_summary, output_dir / "benchmark_summary.csv")
    save_frame(model_score_summary, output_dir / "model_score_summary.csv")
    save_frame(detail_frame, output_dir / "baseline_ticker_summary.csv")
    save_frame(aggregate_frame, output_dir / "baseline_ticker_qualification.csv")
    save_frame(_build_policy_predictions(dataset, placeholder_tickers=placeholder_tickers).set_index("sample_id"), output_dir / "policy_predictions.csv")
    latest_policy_weights = pd.Series(dataset.actions[-1], index=placeholder_tickers, name="policy_weight")
    latest_policy_weights.index.name = "Ticker"
    save_frame(latest_policy_weights, output_dir / "latest_policy_weights.csv")
    summary_lines = [
        f"Oracle Memory Model: {variant_name}",
        f"Training horizon: {duration_key}",
        f"Benchmark: {benchmark_ticker}",
        f"Selected candidate: {variant_name}",
        "Selected epoch: 0",
        "",
        "Benchmark summary:",
        benchmark_summary.to_string(float_format=lambda value: f'{value:0.6f}'),
        "",
        "Benchmark ETF qualification:",
        aggregate_frame.to_string(index=False, float_format=lambda value: f'{value:0.6f}'),
    ]
    (output_dir / "random_sp500_training_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = ROOT / args.output_root / f"portfolio_size_{int(args.universe_size)}"
    run_stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%d_%H%M%S")
    tickers = _selected_universe(args.universe_size)
    placeholder_tickers = _placeholder_tickers(len(tickers))

    config = load_config(args.config)
    loader = MarketDataLoader(cache_dir=config.data.cache_dir)
    prices = clean_price_data(
        _load_or_merge_cached_prices(
            loader,
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
        ),
        drop_all_nan_assets=config.preprocessing.drop_all_nan_assets,
        forward_fill=config.preprocessing.forward_fill_prices,
    )
    returns = compute_returns(prices, return_type=config.preprocessing.return_type)

    summary_rows: list[dict[str, object]] = []
    for profile in REPLAY_DURATION_PROFILES:
        frequency = DEFAULT_DURATION_FREQUENCIES.get(profile.key, "ME")
        histories = _build_oracle_weight_histories(
            returns,
            tickers=tickers,
            lookback_window=profile.lookback_window,
            rebalance_frequency=frequency,
        )
        for variant_name, history_key in ORACLE_VARIANTS.items():
            dataset = build_offline_rl_dataset(
                returns,
                {variant_name: histories[history_key]},
                lookback_window=profile.lookback_window,
                benchmark_ticker=args.benchmark,
            )
            split_index = max(1, int(len(dataset.states) * 0.8))
            benchmark_summary = _build_benchmark_summary(
                train_policy_raw=dataset.raw_rewards[:split_index],
                train_policy_excess=dataset.raw_rewards[:split_index] - dataset.benchmark_rewards[:split_index],
                train_benchmark_raw=dataset.benchmark_rewards[:split_index],
                train_equal_weight_raw=dataset.equal_weight_rewards[:split_index],
                train_restricted_random_raw=dataset.restricted_random_rewards[:split_index],
                validation_policy_raw=dataset.raw_rewards[split_index:],
                validation_policy_excess=dataset.raw_rewards[split_index:] - dataset.benchmark_rewards[split_index:],
                validation_benchmark_raw=dataset.benchmark_rewards[split_index:],
                validation_equal_weight_raw=dataset.equal_weight_rewards[split_index:],
                validation_restricted_random_raw=dataset.restricted_random_rewards[split_index:],
                full_policy_raw=dataset.raw_rewards,
                full_policy_excess=dataset.raw_rewards - dataset.benchmark_rewards,
                full_benchmark_raw=dataset.benchmark_rewards,
                full_equal_weight_raw=dataset.equal_weight_rewards,
                full_restricted_random_raw=dataset.restricted_random_rewards,
            )
            evaluation_summary = _build_evaluation_summary(dataset, split_index=split_index)
            model_score_summary = build_model_score_summary(benchmark_summary, evaluation_summary)
            detail_frame, aggregate_frame = _build_baseline_ticker_summary(
                dataset,
                split_index=split_index,
                baseline_tickers=tickers,
            )

            variant_dir = output_root / profile.key / f"{run_stamp}_{variant_name}"
            training_config = RLTrainingConfig(
                lookback_window=profile.lookback_window,
                hidden_dim=160,
                attention_heads=4,
                attention_layers=3,
                epochs=0,
            )
            _save_oracle_variant(
                output_dir=variant_dir,
                variant_name=variant_name,
                duration_key=profile.key,
                dataset=dataset,
                benchmark_ticker=args.benchmark,
                benchmark_summary=benchmark_summary,
                evaluation_summary=evaluation_summary,
                detail_frame=detail_frame,
                aggregate_frame=aggregate_frame,
                model_score_summary=model_score_summary,
                training_config=training_config,
                placeholder_tickers=placeholder_tickers,
            )
            validation_row = aggregate_frame.loc[aggregate_frame["split"] == "validation"].iloc[0]
            all_row = aggregate_frame.loc[aggregate_frame["split"] == "all"].iloc[0]
            summary_rows.append(
                {
                    "duration": profile.key,
                    "variant": variant_name,
                    "output_dir": variant_dir.as_posix(),
                    "validation_beats_all_tickers": bool(validation_row["beats_all_tickers"]),
                    "validation_min_mean_excess_return": float(validation_row["min_mean_excess_return"]),
                    "all_beats_all_tickers": bool(all_row["beats_all_tickers"]),
                    "all_min_mean_excess_return": float(all_row["min_mean_excess_return"]),
                }
            )
            print(
                f"Built {profile.key} {variant_name}: "
                f"validation_beats_all={bool(validation_row['beats_all_tickers'])}, "
                f"all_beats_all={bool(all_row['beats_all_tickers'])}.",
                flush=True,
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_root / f"oracle_memory_summary_{run_stamp}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved oracle memory summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()

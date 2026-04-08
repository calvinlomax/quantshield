"""Random S&P 500 universe training utilities for the actor-critic policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from quantshield.config import AppConfig
from quantshield.data_loader import MarketDataLoader
from quantshield.optimization import optimize_portfolio
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield.risk import RiskConfig as EstimationRiskConfig, estimate_risk
from quantshield.rl import OfflinePortfolioDataset, _state_features
from quantshield.tuned_suite import TUNED_PRESETS

SP500_CONSTITUENTS_CSV_URL = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


@dataclass(slots=True)
class RandomSP500TrainingSpec:
    """Configuration for random S&P 500 universe generation."""

    start_date: str = "2018-01-01"
    end_date: str | None = None
    candidate_pool_size: int = 80
    random_universes: int = 64
    portfolio_size: int = 10
    random_seed: int = 42
    rebalance_frequency: str = "W-FRI"
    lookback_window: int = 63
    benchmark_mode: str = "__equal_weight__"
    batch_download_size: int = 40
    force_refresh: bool = False
    objectives: tuple[str, ...] = ("mean_variance",)


def fetch_sp500_constituents() -> list[str]:
    """Fetch the current S&P 500 constituent list."""
    try:
        table = pd.read_csv(SP500_CONSTITUENTS_CSV_URL)
    except Exception:
        table = pd.read_html(SP500_WIKIPEDIA_URL, match="Symbol")[0]

    if "Symbol" not in table.columns:
        raise ValueError("The constituent source does not contain a Symbol column.")
    symbols = table["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    deduped: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        upper = symbol.strip().upper()
        if upper and upper not in seen:
            deduped.append(upper)
            seen.add(upper)
    if not deduped:
        raise ValueError("No S&P 500 symbols were parsed from the constituent source.")
    return deduped


def sample_random_universes(
    constituents: Iterable[str],
    *,
    candidate_pool_size: int,
    random_universes: int,
    portfolio_size: int,
    random_seed: int,
) -> tuple[list[str], list[list[str]]]:
    """Sample a candidate stock pool and per-epoch random 10-stock universes."""
    rng = np.random.default_rng(random_seed)
    ordered = list(dict.fromkeys(ticker.strip().upper() for ticker in constituents if ticker.strip()))
    if len(ordered) < candidate_pool_size:
        raise ValueError("Candidate pool size exceeds the available constituent count.")
    if portfolio_size > candidate_pool_size:
        raise ValueError("Portfolio size cannot exceed the sampled candidate pool size.")

    candidate_pool = sorted(rng.choice(ordered, size=candidate_pool_size, replace=False).tolist())
    universes = [
        sorted(rng.choice(candidate_pool, size=portfolio_size, replace=False).tolist())
        for _ in range(random_universes)
    ]
    return candidate_pool, universes


def fetch_price_panel(
    *,
    tickers: list[str],
    loader: MarketDataLoader,
    start_date: str,
    end_date: str | None,
    batch_size: int = 40,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch a wide price panel in manageable yfinance batches."""
    frames: list[pd.DataFrame] = []
    for start in range(0, len(tickers), batch_size):
        chunk = tickers[start : start + batch_size]
        prices = loader.fetch_prices(
            chunk,
            start_date,
            end_date,
            use_cache=True,
            force_refresh=force_refresh,
        )
        frames.append(prices)
    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.loc[:, ~combined.columns.duplicated(keep="last")]
    return combined.reindex(columns=tickers)


def _objective_config_for_universe(
    base_config: AppConfig,
    *,
    objective: str,
    tickers: list[str],
    rebalance_frequency: str,
) -> AppConfig:
    preset = deepcopy(TUNED_PRESETS[objective])
    config = deepcopy(base_config)
    config.data.tickers = list(tickers)
    config.data.asset_class_map = {ticker: "equity" for ticker in tickers}
    config.optimization.objective = objective
    config.risk.covariance_estimator = str(preset["covariance_estimator"])
    config.backtest.lookback_days = int(preset["lookback_days"])
    config.backtest.expanding_window = bool(preset["expanding_window"])
    config.backtest.rebalance_frequency = rebalance_frequency
    config.backtest.benchmark_ticker = tickers[0]
    config.optimization.max_weight = float(preset["max_weight"])
    config.optimization.turnover_penalty = float(preset["turnover_penalty"])
    if "risk_aversion" in preset:
        config.optimization.risk_aversion = float(preset["risk_aversion"])
    return config


def _rebalance_schedule(index: pd.DatetimeIndex, frequency: str) -> list[pd.Timestamp]:
    """Return realized rebalance dates that exist in the source index."""
    anchor = pd.Series(index=index, data=index)
    schedule = anchor.groupby(pd.Grouper(freq=frequency)).last().dropna()
    return [pd.Timestamp(value) for value in schedule.tolist()]


def combine_offline_datasets(datasets: list[OfflinePortfolioDataset]) -> OfflinePortfolioDataset:
    """Concatenate random-universe offline datasets into a single training panel."""
    if not datasets:
        raise ValueError("At least one offline dataset is required.")

    action_dim = datasets[0].actions.shape[1]
    lookback_window = datasets[0].lookback_window
    feature_names = list(datasets[0].feature_names)

    for dataset in datasets:
        if dataset.actions.shape[1] != action_dim:
            raise ValueError("All random-universe datasets must have the same action dimension.")
        if dataset.lookback_window != lookback_window:
            raise ValueError("All random-universe datasets must share the same lookback window.")

    metadata = pd.concat([dataset.metadata for dataset in datasets], ignore_index=True)
    return OfflinePortfolioDataset(
        states=np.concatenate([dataset.states for dataset in datasets], axis=0),
        actions=np.concatenate([dataset.actions for dataset in datasets], axis=0),
        rewards=np.concatenate([dataset.rewards for dataset in datasets], axis=0),
        raw_rewards=np.concatenate([dataset.raw_rewards for dataset in datasets], axis=0),
        benchmark_rewards=np.concatenate([dataset.benchmark_rewards for dataset in datasets], axis=0),
        forward_segments=[segment for dataset in datasets for segment in dataset.forward_segments],
        metadata=metadata,
        tickers=[f"ASSET_{index + 1:02d}" for index in range(action_dim)],
        feature_names=feature_names,
        lookback_window=lookback_window,
    )


def build_random_sp500_dataset(
    base_config: AppConfig,
    *,
    loader: MarketDataLoader | None = None,
    spec: RandomSP500TrainingSpec,
) -> tuple[OfflinePortfolioDataset, dict[str, object]]:
    """Build a per-epoch offline dataset from random S&P 500 10-stock universes."""
    loader = loader or MarketDataLoader(cache_dir=base_config.data.cache_dir)
    constituents = fetch_sp500_constituents()
    candidate_pool, universes = sample_random_universes(
        constituents,
        candidate_pool_size=spec.candidate_pool_size,
        random_universes=spec.random_universes,
        portfolio_size=spec.portfolio_size,
        random_seed=spec.random_seed,
    )

    rng = np.random.default_rng(spec.random_seed)
    all_tickers = candidate_pool
    prices = fetch_price_panel(
        tickers=all_tickers,
        loader=loader,
        start_date=spec.start_date,
        end_date=spec.end_date,
        batch_size=spec.batch_download_size,
        force_refresh=spec.force_refresh,
    )
    clean_prices = clean_price_data(
        prices,
        drop_all_nan_assets=base_config.preprocessing.drop_all_nan_assets,
        forward_fill=base_config.preprocessing.forward_fill_prices,
    )
    returns = compute_returns(clean_prices, return_type=base_config.preprocessing.return_type)
    schedule = _rebalance_schedule(returns.index, spec.rebalance_frequency)
    eligible_dates = [
        rebalance_date
        for date_index, rebalance_date in enumerate(schedule[:-1])
        if returns.index.get_loc(rebalance_date) + 1 >= spec.lookback_window and date_index < len(schedule) - 1
    ]
    if not eligible_dates:
        raise ValueError("No eligible rebalance dates were available for the requested random S&P 500 training sample.")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    benchmark_rewards: list[float] = []
    forward_segments: list[np.ndarray] = []
    universe_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    tickers = [f"ASSET_{index + 1:02d}" for index in range(spec.portfolio_size)]

    schedule_positions = {rebalance_date: position for position, rebalance_date in enumerate(schedule)}

    for universe_id, sampled_tickers in enumerate(universes):
        available_tickers = [ticker for ticker in sampled_tickers if ticker in returns.columns]
        if len(available_tickers) < spec.portfolio_size:
            continue

        rebalance_date = pd.Timestamp(rng.choice(eligible_dates))
        rebalance_position = schedule_positions[rebalance_date]
        next_rebalance_date = schedule[rebalance_position + 1]

        universe_returns = returns.loc[:, available_tickers]
        window = universe_returns.loc[:rebalance_date].iloc[-spec.lookback_window:]
        forward_segment = universe_returns.loc[(universe_returns.index > rebalance_date) & (universe_returns.index <= next_rebalance_date)]
        if len(window) < spec.lookback_window or forward_segment.empty:
            continue

        for objective in spec.objectives:
            config = _objective_config_for_universe(
                base_config,
                objective=objective,
                tickers=available_tickers,
                rebalance_frequency=spec.rebalance_frequency,
            )
            risk_estimate = estimate_risk(
                window,
                EstimationRiskConfig(
                    mean_estimator=config.risk.mean_estimator,
                    covariance_estimator=config.risk.covariance_estimator,
                    ewma_span=config.risk.ewma_span,
                    annualize=config.risk.annualize,
                ),
                periods_per_year=base_config.preprocessing.annualization_factor,
            )
            optimization_result = optimize_portfolio(
                risk_estimate.mean,
                risk_estimate.covariance,
                config.optimization,
                asset_class_map=config.data.asset_class_map,
            )
            action = optimization_result.weights.reindex(available_tickers).to_numpy(dtype=np.float32)
            action = action / np.clip(action.sum(), 1e-6, None)
            segment_array = forward_segment.to_numpy(dtype=np.float32)
            raw_reward = float(np.prod(1.0 + segment_array @ action) - 1.0)
            benchmark_segment = segment_array.mean(axis=1)
            benchmark_reward = float(np.prod(1.0 + benchmark_segment) - 1.0)
            excess_reward = raw_reward - benchmark_reward

            states.append(_state_features(window))
            actions.append(action)
            rewards.append(excess_reward)
            raw_rewards.append(raw_reward)
            benchmark_rewards.append(benchmark_reward)
            forward_segments.append(segment_array)
            metadata_rows.append(
                {
                    "objective": objective,
                    "rebalance_date": rebalance_date,
                    "forward_start": forward_segment.index[0],
                    "forward_end": forward_segment.index[-1],
                    "raw_reward": raw_reward,
                    "benchmark_reward": benchmark_reward,
                    "excess_reward": excess_reward,
                    "universe_id": universe_id,
                    "universe_tickers": ",".join(available_tickers),
                }
            )

        universe_rows.append(
            {
                "universe_id": universe_id,
                "tickers": ",".join(available_tickers),
                "rebalance_date": rebalance_date,
                "forward_end": next_rebalance_date,
                "samples": len(spec.objectives),
            }
        )

    if not states:
        raise ValueError("No random-universe offline RL samples were created.")

    combined = OfflinePortfolioDataset(
        states=np.stack(states, axis=0),
        actions=np.stack(actions, axis=0),
        rewards=np.asarray(rewards, dtype=np.float32),
        raw_rewards=np.asarray(raw_rewards, dtype=np.float32),
        benchmark_rewards=np.asarray(benchmark_rewards, dtype=np.float32),
        forward_segments=forward_segments,
        metadata=pd.DataFrame(metadata_rows),
        tickers=tickers,
        feature_names=["return", "z_score", "cumulative_return"],
        lookback_window=spec.lookback_window,
    )
    summary = {
        "candidate_pool": candidate_pool,
        "universes": universes,
        "universe_summary": pd.DataFrame(universe_rows),
        "constituent_count": len(constituents),
        "sampled_ticker_count": len(candidate_pool),
    }
    return combined, summary

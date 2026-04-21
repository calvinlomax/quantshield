"""Random S&P 500 universe training utilities for the actor-critic policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from quantshield.config import AppConfig
from quantshield.data_loader import MarketDataLoader
from quantshield.optimization import optimize_portfolio
from quantshield.preprocessing import clean_price_data, compute_returns
from quantshield.risk import RiskConfig as EstimationRiskConfig, estimate_risk
from quantshield.rl import (
    OfflinePortfolioDataset,
    _compose_training_reward,
    _mean_variance_baseline_reward,
    _restricted_random_weights,
    _stable_seed,
    _state_features,
)
from quantshield.tuned_suite import TUNED_PRESETS
from quantshield.universe import CANONICAL_TOP_ETF_UNIVERSE, SEARCH_SEED_TICKERS

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
    candidate_tickers: tuple[str, ...] | None = None
    random_seed: int = 42
    rebalance_frequency: str = "W-FRI"
    lookback_window: int = 63
    benchmark_mode: str = "__config__"
    batch_download_size: int = 40
    force_refresh: bool = False
    objectives: tuple[str, ...] = tuple(TUNED_PRESETS.keys())
    objective_suite_root: str = "outputs/ml_tuned_objective_runs"
    prior_weight: float = 0.35
    mixture_temperature: float = 0.18
    restricted_random_min_weight: float = 0.0
    restricted_random_max_weight: float = 0.35
    reward_weight_raw: float = 0.10
    reward_weight_vs_benchmark: float = 0.40
    reward_weight_vs_equal_weight: float = 0.30
    reward_weight_vs_restricted_random: float = 0.20
    reward_weight_vs_markowitz: float = 0.0
    markowitz_risk_aversion: float = 3.0
    markowitz_max_weight: float = 0.35


@dataclass(slots=True)
class ObjectiveRunPriors:
    """Objective-run priors loaded from the saved tuned-suite artifacts."""

    global_excess_scores: dict[str, float]
    daily_excess_returns: dict[str, pd.Series]


def fetch_sp500_constituents() -> list[str]:
    """Fetch the current S&P 500 constituent list."""
    try:
        table = pd.read_csv(SP500_CONSTITUENTS_CSV_URL)
    except Exception:
        try:
            table = pd.read_html(SP500_WIKIPEDIA_URL, match="Symbol")[0]
        except Exception:
            return _cached_constituent_fallback()

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


def _cached_constituent_fallback(cache_dir: str | Path = "data/raw") -> list[str]:
    """Recover a broad ticker universe from locally cached price panels."""
    root = Path(cache_dir)
    candidate_symbols: set[str] = set()
    excluded = {ticker.upper() for ticker in (*CANONICAL_TOP_ETF_UNIVERSE, *SEARCH_SEED_TICKERS)}
    for path in root.glob("*.csv"):
        if path.name.startswith("."):
            continue
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if columns and columns[0] == "Date":
            columns = columns[1:]
        for column in columns:
            symbol = str(column).strip().upper()
            if not symbol:
                continue
            if symbol in excluded or symbol.startswith("ASSET_"):
                continue
            if re.fullmatch(r"\d+", symbol):
                continue
            candidate_symbols.add(symbol)
    if len(candidate_symbols) < 50:
        raise ValueError("Unable to recover a sufficiently broad local constituent universe.")
    return sorted(candidate_symbols)


def _cached_training_universe(
    cache_dir: str | Path,
    *,
    start_date: str,
    end_date: str | None,
) -> list[str]:
    """Return the largest cached ticker universe available for the requested date range."""
    root = Path(cache_dir)
    end_component = end_date or "latest"
    excluded = {ticker.upper() for ticker in (*CANONICAL_TOP_ETF_UNIVERSE, *SEARCH_SEED_TICKERS)}
    best_symbols: list[str] = []
    pattern = f"*_{start_date}_{end_component}.csv"
    for path in root.glob(pattern):
        if path.name.startswith("."):
            continue
        try:
            columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if columns and columns[0] == "Date":
            columns = columns[1:]
        symbols = [
            str(column).strip().upper()
            for column in columns
            if str(column).strip()
            and str(column).strip().upper() not in excluded
            and not str(column).strip().upper().startswith("ASSET_")
            and not re.fullmatch(r"\d+", str(column).strip().upper())
        ]
        if len(symbols) > len(best_symbols):
            best_symbols = symbols
    return sorted(best_symbols)


def _load_largest_cached_price_panel(
    loader: MarketDataLoader,
    *,
    start_date: str,
    end_date: str | None,
    benchmark_ticker: str | None,
) -> pd.DataFrame | None:
    """Return the broadest cached price panel that covers the requested sample window."""
    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date) if end_date is not None else None
    excluded = {ticker.upper() for ticker in (*CANONICAL_TOP_ETF_UNIVERSE, *SEARCH_SEED_TICKERS)}
    best_prices: pd.DataFrame | None = None
    best_rank: tuple[int, int] | None = None

    for candidate_path in Path(loader.cache_dir).glob("*.csv"):
        try:
            cached_prices = loader.load_cached_prices(candidate_path)
        except Exception:
            continue
        if cached_prices.empty:
            continue
        if cached_prices.index.min() > requested_start:
            continue
        if requested_end is not None and cached_prices.index.max() < requested_end:
            continue

        subset = cached_prices.loc[cached_prices.index >= requested_start]
        if requested_end is not None:
            subset = subset.loc[subset.index <= requested_end]
        if subset.empty:
            continue

        available_symbols = [
            str(column).strip().upper()
            for column in subset.columns
            if str(column).strip()
            and str(column).strip().upper() not in excluded
            and not str(column).strip().upper().startswith("ASSET_")
            and not re.fullmatch(r"\d+", str(column).strip().upper())
        ]
        if benchmark_ticker is not None and benchmark_ticker not in subset.columns:
            continue

        rank = (len(available_symbols), len(subset))
        if best_rank is not None and rank <= best_rank:
            continue
        ordered_columns = available_symbols + ([benchmark_ticker] if benchmark_ticker and benchmark_ticker not in available_symbols else [])
        best_prices = subset.loc[:, [column for column in ordered_columns if column in subset.columns]].copy()
        best_rank = rank

    return best_prices


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


def load_objective_run_priors(
    suite_root: str | Path,
    objectives: Iterable[str],
) -> ObjectiveRunPriors:
    """Load global and daily excess-return priors from saved objective-suite outputs."""
    root = Path(suite_root)
    comparison_path = root / "tuned_objective_comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError(f"Objective comparison table not found: {comparison_path}")

    comparison = pd.read_csv(comparison_path, index_col=0)
    global_scores = (
        comparison.get("excess_return_vs_spy", pd.Series(dtype=float))
        .reindex(list(objectives))
        .fillna(0.0)
        .astype(float)
        .to_dict()
    )

    daily_excess_returns: dict[str, pd.Series] = {}
    for objective in objectives:
        comparison_returns_path = root / objective / "tables" / "comparison_returns.csv"
        if not comparison_returns_path.exists():
            raise FileNotFoundError(f"Objective comparison returns not found: {comparison_returns_path}")
        frame = pd.read_csv(comparison_returns_path, parse_dates=["Date"])
        frame["Date"] = pd.to_datetime(frame["Date"])
        daily_excess = pd.Series(
            frame["portfolio"].astype(float).to_numpy() - frame["benchmark"].astype(float).to_numpy(),
            index=frame["Date"],
            name="daily_excess_return",
        ).sort_index()
        daily_excess_returns[objective] = daily_excess

    return ObjectiveRunPriors(
        global_excess_scores=global_scores,
        daily_excess_returns=daily_excess_returns,
    )


def _annualized_mean_return(series: pd.Series, periods_per_year: int) -> float:
    if series.empty:
        return 0.0
    return float(series.mean()) * periods_per_year


def _softmax_weights(values: dict[str, float], temperature: float) -> dict[str, float]:
    keys = list(values)
    if not keys:
        return {}
    vector = np.asarray([float(values[key]) for key in keys], dtype=np.float64)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    safe_temperature = max(float(temperature), 1e-3)
    stabilized = (vector - vector.max()) / safe_temperature
    exponentials = np.exp(stabilized)
    weights = exponentials / np.clip(exponentials.sum(), 1e-12, None)
    return {key: float(weight) for key, weight in zip(keys, weights, strict=True)}


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
        equal_weight_rewards=np.concatenate([dataset.equal_weight_rewards for dataset in datasets], axis=0),
        restricted_random_rewards=np.concatenate([dataset.restricted_random_rewards for dataset in datasets], axis=0),
        markowitz_rewards=np.concatenate([dataset.markowitz_rewards for dataset in datasets], axis=0),
        forward_segments=[segment for dataset in datasets for segment in dataset.forward_segments],
        metadata=metadata,
        tickers=[f"ASSET_{index + 1:02d}" for index in range(action_dim)],
        feature_names=feature_names,
        lookback_window=lookback_window,
        reward_weight_raw=datasets[0].reward_weight_raw,
        reward_weight_vs_benchmark=datasets[0].reward_weight_vs_benchmark,
        reward_weight_vs_equal_weight=datasets[0].reward_weight_vs_equal_weight,
        reward_weight_vs_restricted_random=datasets[0].reward_weight_vs_restricted_random,
        reward_weight_vs_markowitz=datasets[0].reward_weight_vs_markowitz,
    )


def _build_objective_weighted_target(
    *,
    objective_actions: dict[str, np.ndarray],
    objective_excess_returns: dict[str, float],
    priors: ObjectiveRunPriors,
    forward_index: pd.DatetimeIndex,
    periods_per_year: int,
    prior_weight: float,
    mixture_temperature: float,
) -> tuple[np.ndarray, float, dict[str, float], dict[str, float]]:
    """Blend objective actions into one target using suite priors and realized segment performance."""
    objective_scores: dict[str, float] = {}
    for objective, excess_return in objective_excess_returns.items():
        prior_series = priors.daily_excess_returns.get(objective, pd.Series(dtype=float))
        local_prior_score = _annualized_mean_return(prior_series.reindex(forward_index).dropna(), periods_per_year)
        global_prior_score = float(priors.global_excess_scores.get(objective, 0.0))
        realized_score = float(excess_return) * (periods_per_year / max(len(forward_index), 1))
        objective_scores[objective] = (
            float(prior_weight) * (0.5 * global_prior_score + 0.5 * local_prior_score)
            + (1.0 - float(prior_weight)) * realized_score
        )

    mixture_weights = _softmax_weights(objective_scores, mixture_temperature)
    first_action = next(iter(objective_actions.values()))
    blended_action = np.zeros_like(first_action, dtype=np.float64)
    weighted_reward = 0.0
    for objective, action in objective_actions.items():
        mixture_weight = float(mixture_weights[objective])
        blended_action += mixture_weight * np.asarray(action, dtype=np.float64)
        weighted_reward += mixture_weight * float(objective_excess_returns[objective])

    blended_action = np.clip(blended_action, 1e-6, None)
    blended_action = blended_action / np.clip(blended_action.sum(), 1e-6, None)
    return blended_action.astype(np.float32), float(weighted_reward), objective_scores, mixture_weights


def build_random_sp500_dataset(
    base_config: AppConfig,
    *,
    loader: MarketDataLoader | None = None,
    spec: RandomSP500TrainingSpec,
) -> tuple[OfflinePortfolioDataset, dict[str, object]]:
    """Build a randomized offline dataset from S&P 500 universes with objective-weighted targets."""
    loader = loader or MarketDataLoader(cache_dir=base_config.data.cache_dir)
    priors = load_objective_run_priors(spec.objective_suite_root, spec.objectives)
    benchmark_ticker = (
        base_config.backtest.benchmark_ticker
        if spec.benchmark_mode in {"__config__", "__market__"}
        else spec.benchmark_mode
    )
    use_equal_weight_benchmark = benchmark_ticker == "__equal_weight__"
    use_markowitz_benchmark = benchmark_ticker == "__markowitz__"
    cached_prices = _load_largest_cached_price_panel(
        loader,
        start_date=spec.start_date,
        end_date=spec.end_date,
        benchmark_ticker=None if use_equal_weight_benchmark or use_markowitz_benchmark else benchmark_ticker,
    )
    if cached_prices is not None:
        cached_constituents = [
            str(column).strip().upper()
            for column in cached_prices.columns
            if str(column).strip()
            and str(column).strip().upper() != benchmark_ticker
        ]
    else:
        cached_constituents = _cached_training_universe(
            base_config.data.cache_dir,
            start_date=spec.start_date,
            end_date=spec.end_date,
        )
    if spec.candidate_tickers:
        constituents = [str(ticker).strip().upper() for ticker in spec.candidate_tickers if str(ticker).strip()]
    else:
        constituents = cached_constituents if len(cached_constituents) >= spec.candidate_pool_size else fetch_sp500_constituents()
    candidate_pool, universes = sample_random_universes(
        constituents,
        candidate_pool_size=min(spec.candidate_pool_size, len(constituents)),
        random_universes=spec.random_universes,
        portfolio_size=spec.portfolio_size,
        random_seed=spec.random_seed,
    )

    rng = np.random.default_rng(spec.random_seed)
    all_tickers = list(candidate_pool)
    if not use_equal_weight_benchmark and not use_markowitz_benchmark and benchmark_ticker not in all_tickers:
        all_tickers.append(benchmark_ticker)
    if cached_prices is not None and set(all_tickers).issubset(cached_prices.columns):
        prices = cached_prices.loc[:, all_tickers].copy()
    else:
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
    benchmark_returns = (
        returns[benchmark_ticker]
        if not use_equal_weight_benchmark and not use_markowitz_benchmark and benchmark_ticker in returns.columns
        else None
    )
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
    equal_weight_rewards: list[float] = []
    restricted_random_rewards: list[float] = []
    markowitz_rewards: list[float] = []
    forward_segments: list[np.ndarray] = []
    universe_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    tickers = [f"ASSET_{index + 1:02d}" for index in range(spec.portfolio_size)]

    schedule_positions = {rebalance_date: position for position, rebalance_date in enumerate(schedule)}
    periods_per_year = base_config.preprocessing.annualization_factor

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

        segment_array = forward_segment.to_numpy(dtype=np.float32)
        equal_weight_segment = segment_array.mean(axis=1)
        equal_weight_reward = float(np.prod(1.0 + equal_weight_segment) - 1.0)
        if benchmark_returns is None:
            benchmark_reward = equal_weight_reward
        else:
            benchmark_segment = benchmark_returns.loc[
                (benchmark_returns.index > rebalance_date) & (benchmark_returns.index <= next_rebalance_date)
            ].to_numpy(dtype=np.float32)
            benchmark_reward = float(np.prod(1.0 + benchmark_segment) - 1.0) if len(benchmark_segment) else equal_weight_reward
        restricted_random_action = _restricted_random_weights(
            len(available_tickers),
            seed=_stable_seed("restricted_random", universe_id, rebalance_date.isoformat(), next_rebalance_date.isoformat()),
            min_weight=spec.restricted_random_min_weight,
            max_weight=spec.restricted_random_max_weight,
        )
        restricted_random_reward = float(np.prod(1.0 + segment_array @ restricted_random_action) - 1.0)
        markowitz_reward = _mean_variance_baseline_reward(
            window,
            forward_segment,
            periods_per_year=periods_per_year,
            risk_aversion=spec.markowitz_risk_aversion,
            max_weight=spec.markowitz_max_weight,
        )
        if use_markowitz_benchmark:
            benchmark_reward = markowitz_reward
        objective_actions: dict[str, np.ndarray] = {}
        objective_raw_rewards: dict[str, float] = {}
        objective_excess_rewards: dict[str, float] = {}

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
            raw_reward = float(np.prod(1.0 + segment_array @ action) - 1.0)
            excess_reward = raw_reward - benchmark_reward
            objective_actions[objective] = action
            objective_raw_rewards[objective] = raw_reward
            objective_excess_rewards[objective] = excess_reward

        blended_action, weighted_reward, objective_scores, mixture_weights = _build_objective_weighted_target(
            objective_actions=objective_actions,
            objective_excess_returns=objective_excess_rewards,
            priors=priors,
            forward_index=forward_segment.index,
            periods_per_year=periods_per_year,
            prior_weight=spec.prior_weight,
            mixture_temperature=spec.mixture_temperature,
        )
        blended_raw_reward = float(np.prod(1.0 + segment_array @ blended_action) - 1.0)
        blended_excess_reward = blended_raw_reward - benchmark_reward
        composite_reward = _compose_training_reward(
            blended_raw_reward,
            benchmark_reward=benchmark_reward,
            equal_weight_reward=equal_weight_reward,
            restricted_random_reward=restricted_random_reward,
            markowitz_reward=markowitz_reward,
            reward_weight_raw=spec.reward_weight_raw,
            reward_weight_vs_benchmark=spec.reward_weight_vs_benchmark,
            reward_weight_vs_equal_weight=spec.reward_weight_vs_equal_weight,
            reward_weight_vs_restricted_random=spec.reward_weight_vs_restricted_random,
            reward_weight_vs_markowitz=spec.reward_weight_vs_markowitz,
        )
        selected_objective = max(mixture_weights, key=mixture_weights.get)

        states.append(_state_features(window))
        actions.append(blended_action)
        rewards.append(composite_reward)
        raw_rewards.append(blended_raw_reward)
        benchmark_rewards.append(benchmark_reward)
        equal_weight_rewards.append(equal_weight_reward)
        restricted_random_rewards.append(restricted_random_reward)
        markowitz_rewards.append(markowitz_reward)
        forward_segments.append(segment_array)
        metadata_row = {
            "objective": "objective_weighted_ensemble",
            "selected_objective": selected_objective,
            "rebalance_date": rebalance_date,
            "forward_start": forward_segment.index[0],
            "forward_end": forward_segment.index[-1],
            "raw_reward": blended_raw_reward,
            "benchmark_reward": benchmark_reward,
            "equal_weight_reward": equal_weight_reward,
            "restricted_random_reward": restricted_random_reward,
            "markowitz_reward": markowitz_reward,
            "excess_reward": blended_excess_reward,
            "weighted_reward": weighted_reward,
            "composite_reward": composite_reward,
            "universe_id": universe_id,
            "universe_tickers": ",".join(available_tickers),
        }
        for objective in spec.objectives:
            metadata_row[f"objective_score_{objective}"] = float(objective_scores[objective])
            metadata_row[f"objective_weight_{objective}"] = float(mixture_weights[objective])
            metadata_row[f"objective_excess_{objective}"] = float(objective_excess_rewards[objective])
            metadata_row[f"objective_raw_{objective}"] = float(objective_raw_rewards[objective])
        metadata_rows.append(metadata_row)

        universe_rows.append(
            {
                "universe_id": universe_id,
                "tickers": ",".join(available_tickers),
                "rebalance_date": rebalance_date,
                "forward_end": next_rebalance_date,
                "samples": 1,
                "selected_objective": selected_objective,
                "weighted_reward": weighted_reward,
                "composite_reward": composite_reward,
                "blended_excess_reward": blended_excess_reward,
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
        equal_weight_rewards=np.asarray(equal_weight_rewards, dtype=np.float32),
        restricted_random_rewards=np.asarray(restricted_random_rewards, dtype=np.float32),
        markowitz_rewards=np.asarray(markowitz_rewards, dtype=np.float32),
        forward_segments=forward_segments,
        metadata=pd.DataFrame(metadata_rows),
        tickers=tickers,
        feature_names=["return", "z_score", "cumulative_return"],
        lookback_window=spec.lookback_window,
        reward_weight_raw=spec.reward_weight_raw,
        reward_weight_vs_benchmark=spec.reward_weight_vs_benchmark,
        reward_weight_vs_equal_weight=spec.reward_weight_vs_equal_weight,
        reward_weight_vs_restricted_random=spec.reward_weight_vs_restricted_random,
        reward_weight_vs_markowitz=spec.reward_weight_vs_markowitz,
    )
    summary = {
        "candidate_pool": candidate_pool,
        "universes": universes,
        "universe_summary": pd.DataFrame(universe_rows),
        "constituent_count": len(constituents),
        "sampled_ticker_count": len(candidate_pool),
    }
    return combined, summary

from __future__ import annotations

import numpy as np
import pandas as pd

from quantshield.config import AppConfig, DataConfig
from quantshield.sp500_random_training import (
    ObjectiveRunPriors,
    RandomSP500TrainingSpec,
    _build_objective_weighted_target,
    build_random_sp500_dataset,
    sample_random_universes,
)


def test_sample_random_universes_shapes() -> None:
    constituents = [f"TICK{index:03d}" for index in range(50)]
    candidate_pool, universes = sample_random_universes(
        constituents,
        candidate_pool_size=20,
        random_universes=8,
        portfolio_size=10,
        random_seed=7,
    )

    assert len(candidate_pool) == 20
    assert len(universes) == 8
    assert all(len(universe) == 10 for universe in universes)
    assert all(set(universe).issubset(candidate_pool) for universe in universes)


def test_build_random_sp500_dataset_from_mock_panel(monkeypatch) -> None:
    tickers = [f"TICK{index:03d}" for index in range(20)]
    index = pd.date_range("2024-01-01", periods=120, freq="B")
    prices = pd.DataFrame(
        {
            ticker: 100.0 + np.linspace(0.0, 10.0 + offset, len(index))
            for offset, ticker in enumerate(tickers)
        },
        index=index,
    )

    monkeypatch.setattr("quantshield.sp500_random_training.fetch_sp500_constituents", lambda: tickers)
    monkeypatch.setattr("quantshield.sp500_random_training.fetch_price_panel", lambda **_: prices)
    priors = ObjectiveRunPriors(
        global_excess_scores={"min_variance": -0.01, "mean_variance": 0.05, "risk_parity": 0.01, "equal_weight": -0.02},
        daily_excess_returns={
            "min_variance": pd.Series(-0.0002, index=index),
            "mean_variance": pd.Series(0.0008, index=index),
            "risk_parity": pd.Series(0.0003, index=index),
            "equal_weight": pd.Series(-0.0001, index=index),
        },
    )
    monkeypatch.setattr("quantshield.sp500_random_training.load_objective_run_priors", lambda *args, **kwargs: priors)

    config = AppConfig(data=DataConfig(tickers=tickers))
    dataset, summary = build_random_sp500_dataset(
        config,
        spec=RandomSP500TrainingSpec(
            start_date="2024-01-01",
            end_date="2024-06-30",
            candidate_pool_size=12,
            random_universes=6,
            portfolio_size=10,
            random_seed=11,
            lookback_window=20,
            rebalance_frequency="W-FRI",
            objectives=("min_variance", "mean_variance", "risk_parity", "equal_weight"),
        ),
    )

    assert dataset.states.shape == (6, 10, 20, 3)
    assert dataset.actions.shape == (6, 10)
    assert len(dataset.metadata) == 6
    assert set(["universe_id", "universe_tickers", "rebalance_date", "selected_objective", "weighted_reward"]).issubset(dataset.metadata.columns)
    assert len(dataset.equal_weight_rewards) == len(dataset.states)
    assert len(dataset.restricted_random_rewards) == len(dataset.states)
    assert len(dataset.markowitz_rewards) == len(dataset.states)
    assert len(summary["candidate_pool"]) == 12


def test_objective_weighted_target_prefers_stronger_objectives() -> None:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    action_mv = np.array([0.60, 0.30, 0.10], dtype=np.float32)
    action_eq = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
    priors = ObjectiveRunPriors(
        global_excess_scores={"mean_variance": 0.08, "equal_weight": -0.02},
        daily_excess_returns={
            "mean_variance": pd.Series(0.001, index=index),
            "equal_weight": pd.Series(-0.001, index=index),
        },
    )

    blended_action, weighted_reward, scores, weights = _build_objective_weighted_target(
        objective_actions={"mean_variance": action_mv, "equal_weight": action_eq},
        objective_excess_returns={"mean_variance": 0.03, "equal_weight": -0.01},
        priors=priors,
        forward_index=index,
        periods_per_year=252,
        prior_weight=0.35,
        mixture_temperature=0.15,
    )

    assert np.isclose(blended_action.sum(), 1.0)
    assert weights["mean_variance"] > weights["equal_weight"]
    assert scores["mean_variance"] > scores["equal_weight"]
    assert weighted_reward > 0.0

from __future__ import annotations

import numpy as np
import pandas as pd

from quantshield.config import AppConfig, DataConfig
from quantshield.sp500_random_training import (
    RandomSP500TrainingSpec,
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
            objectives=("mean_variance",),
        ),
    )

    assert dataset.states.shape == (6, 10, 20, 3)
    assert dataset.actions.shape == (6, 10)
    assert len(dataset.metadata) == 6
    assert set(["universe_id", "universe_tickers", "rebalance_date"]).issubset(dataset.metadata.columns)
    assert len(summary["candidate_pool"]) == 12

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quantshield.rl import (  # noqa: E402
    CrossAssetAttentionActorCritic,
    build_offline_rl_dataset,
)


def test_build_offline_rl_dataset_shapes() -> None:
    index = pd.date_range("2024-01-01", periods=90, freq="B")
    returns = pd.DataFrame(
        {
            "SPY": np.linspace(0.0010, 0.0020, len(index)),
            "QQQ": np.linspace(0.0015, 0.0025, len(index)),
            "GLD": np.linspace(0.0005, 0.0015, len(index)),
        },
        index=index,
    )
    weights = pd.DataFrame(
        {
            "SPY": [0.4, 0.3],
            "QQQ": [0.4, 0.5],
            "GLD": [0.2, 0.2],
        },
        index=pd.to_datetime(["2024-03-29", "2024-04-30"]),
    )

    dataset = build_offline_rl_dataset(
        returns,
        {"risk_parity": weights},
        lookback_window=20,
        benchmark_ticker="SPY",
    )

    assert dataset.states.shape[1:] == (3, 20, 3)
    assert dataset.actions.shape[1] == 3
    assert len(dataset.metadata) == len(dataset.states)


def test_actor_outputs_simplex_weights() -> None:
    model = CrossAssetAttentionActorCritic(
        num_assets=3,
        lookback_window=20,
        feature_dim=3,
        hidden_dim=32,
        attention_heads=4,
        attention_layers=1,
    )
    states = torch.randn(5, 3, 20, 3)
    _, policy_mean, _ = model.policy_distribution(states)

    sums = policy_mean.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert torch.all(policy_mean >= 0.0)

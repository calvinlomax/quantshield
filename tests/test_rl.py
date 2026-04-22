from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from quantshield.rl import (  # noqa: E402
    _compose_training_reward,
    RLTrainingConfig,
    CrossAssetAttentionActorCritic,
    build_offline_rl_dataset,
    normalize_training_config,
    save_actor_critic_artifacts,
    train_transformer_actor_critic,
)
from quantshield.training_targets import build_forward_weight_histories  # noqa: E402


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
    assert len(dataset.equal_weight_rewards) == len(dataset.states)
    assert len(dataset.restricted_random_rewards) == len(dataset.states)
    assert len(dataset.markowitz_rewards) == len(dataset.states)


def test_build_offline_rl_dataset_accepts_external_benchmark_series() -> None:
    index = pd.date_range("2024-01-01", periods=90, freq="B")
    returns = pd.DataFrame(
        {
            "ACGL": np.linspace(0.0010, 0.0020, len(index)),
            "MAR": np.linspace(0.0015, 0.0025, len(index)),
            "MSCI": np.linspace(0.0005, 0.0015, len(index)),
            "SPY": np.linspace(0.0008, 0.0018, len(index)),
        },
        index=index,
    )
    weights = pd.DataFrame(
        {
            "ACGL": [0.4, 0.3],
            "MAR": [0.4, 0.5],
            "MSCI": [0.2, 0.2],
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
    assert len(dataset.benchmark_rewards) == len(dataset.states)
    assert np.isfinite(dataset.benchmark_rewards).all()


def test_compose_training_reward_can_compare_against_best_selected_baseline() -> None:
    separate = _compose_training_reward(
        0.08,
        benchmark_reward=0.03,
        equal_weight_reward=0.06,
        restricted_random_reward=0.04,
        markowitz_reward=0.05,
        reward_weight_raw=0.0,
        reward_weight_vs_benchmark=0.4,
        reward_weight_vs_equal_weight=0.3,
        reward_weight_vs_restricted_random=0.2,
        reward_weight_vs_markowitz=0.1,
        reward_comparison_mode="separate",
    )
    best_of = _compose_training_reward(
        0.08,
        benchmark_reward=0.03,
        equal_weight_reward=0.06,
        restricted_random_reward=0.04,
        markowitz_reward=0.05,
        reward_weight_raw=0.0,
        reward_weight_vs_benchmark=0.4,
        reward_weight_vs_equal_weight=0.3,
        reward_weight_vs_restricted_random=0.2,
        reward_weight_vs_markowitz=0.1,
        reward_comparison_mode="best_of_selected",
    )

    assert separate == pytest.approx(0.037)
    assert best_of == pytest.approx(0.024)


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


def test_normalize_training_config_makes_attention_heads_divide_hidden_dim() -> None:
    config = RLTrainingConfig(
        lookback_window=5,
        hidden_dim=160,
        attention_heads=6,
        attention_layers=3,
    )

    normalized = normalize_training_config(config)

    assert normalized.hidden_dim == 160
    assert normalized.attention_heads in {1, 2, 4, 5, 8, 10, 16}
    assert normalized.hidden_dim % normalized.attention_heads == 0


def test_build_forward_weight_histories_handles_single_step_segments_without_covariance_warning() -> None:
    index = pd.date_range("2024-01-01", periods=12, freq="B")
    returns = pd.DataFrame(
        {
            "SPY": np.linspace(0.0010, 0.0020, len(index)),
            "QQQ": np.linspace(0.0015, 0.0025, len(index)),
            "GLD": np.linspace(0.0005, 0.0012, len(index)),
            "IVV": np.linspace(0.0011, 0.0019, len(index)),
            "VOO": np.linspace(0.0009, 0.0017, len(index)),
        },
        index=index,
    )
    asset_class_map = {ticker: "equity" for ticker in returns.columns}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        histories = build_forward_weight_histories(
            returns,
            tickers=list(returns.columns),
            lookback_window=5,
            rebalance_frequency="B",
            asset_class_map=asset_class_map,
        )

    assert histories["forward_mean_variance"].empty is False
    assert caught == []


def test_training_produces_benchmark_summary(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=160, freq="B")
    returns = pd.DataFrame(
        {
            "SPY": np.linspace(0.0008, 0.0018, len(index)),
            "QQQ": np.linspace(0.0010, 0.0022, len(index)),
            "GLD": np.linspace(0.0004, 0.0011, len(index)),
        },
        index=index,
    )
    weights = pd.DataFrame(
        {
            "SPY": [0.50, 0.45, 0.35, 0.30],
            "QQQ": [0.30, 0.35, 0.45, 0.50],
            "GLD": [0.20, 0.20, 0.20, 0.20],
        },
        index=pd.to_datetime(["2024-03-29", "2024-04-30", "2024-05-31", "2024-06-28"]),
    )
    dataset = build_offline_rl_dataset(
        returns,
        {"mean_variance": weights},
        lookback_window=20,
        benchmark_ticker="SPY",
    )
    config = RLTrainingConfig(
        lookback_window=20,
        hidden_dim=16,
        attention_heads=4,
        attention_layers=1,
        epochs=1,
        batch_size=2,
    )

    result = train_transformer_actor_critic(dataset, config, device="cpu")

    assert {
        "benchmark_mean_raw_return",
        "policy_mean_raw_return",
        "policy_mean_excess_return",
        "policy_mean_excess_vs_markowitz",
        "t_statistic",
        "p_value",
    }.issubset(
        result.benchmark_summary.columns
    )
    assert list(result.benchmark_summary.index) == ["train", "validation", "all"]
    assert result.selected_epoch >= 1
    assert "selected_checkpoint" in result.history.columns
    assert result.history["selected_checkpoint"].sum() == 1
    assert "composite_score" in result.model_score_summary.columns

    artifact_paths = save_actor_critic_artifacts(result, tmp_path / "rl_artifacts")
    assert artifact_paths["training_diagnostics_fig"].exists()
    assert artifact_paths["benchmark_comparison_fig"].exists()
    assert artifact_paths["policy_cumulative_returns_fig"].exists()
    assert artifact_paths["latest_policy_weights_fig"].exists()
    assert artifact_paths["model_score_summary"].exists()

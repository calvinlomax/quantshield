"""Core transformer actor-critic policy training on saved portfolio weight demonstrations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader, TensorDataset

from quantshield.config import OptimizationConfig
from quantshield.model_scoring import build_model_score_summary
from quantshield.optimization import optimize_portfolio
from quantshield.plotting import (
    plot_rl_benchmark_comparison,
    plot_rl_latest_weights,
    plot_rl_policy_cumulative_returns,
    plot_rl_training_diagnostics,
)
from quantshield.risk import RiskConfig as EstimationRiskConfig, estimate_risk
from quantshield.utils import ensure_directory, save_frame


def _normalize_simplex(weights: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(weights, eps, None)
    return clipped / clipped.sum()


def _segment_cumulative_return(segment_returns: np.ndarray, weights: np.ndarray) -> float:
    daily_returns = segment_returns @ weights
    return float(np.prod(1.0 + daily_returns) - 1.0)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _restricted_random_weights(
    num_assets: int,
    *,
    seed: int,
    min_weight: float = 0.0,
    max_weight: float = 0.35,
    max_attempts: int = 1024,
) -> np.ndarray:
    if num_assets <= 0:
        raise ValueError("Restricted random weights require at least one asset.")
    if min_weight < 0.0:
        raise ValueError("Restricted random minimum weight must be non-negative.")
    if min_weight * num_assets > 1.0:
        raise ValueError("Restricted random minimum weight is infeasible for the asset count.")
    if max_weight < 1.0 / num_assets:
        raise ValueError("Restricted random maximum weight is too small to produce a simplex allocation.")

    rng = np.random.default_rng(seed)
    base = np.full(num_assets, float(min_weight), dtype=np.float64)
    remaining = max(1.0 - float(base.sum()), 0.0)
    best_weights: np.ndarray | None = None
    best_overflow = float("inf")
    for _ in range(max_attempts):
        sample = rng.dirichlet(np.ones(num_assets, dtype=np.float64))
        candidate = base + remaining * sample
        overflow = float(np.maximum(candidate - max_weight, 0.0).sum())
        if overflow < best_overflow:
            best_overflow = overflow
            best_weights = candidate
        if overflow <= 1e-9:
            return candidate.astype(np.float32)

    assert best_weights is not None
    clipped = np.minimum(best_weights, max_weight)
    deficit = 1.0 - float(clipped.sum())
    if deficit > 0.0:
        room = np.maximum(max_weight - clipped, 0.0)
        room_sum = float(room.sum())
        if room_sum > 0.0:
            clipped += deficit * (room / room_sum)
    clipped = np.clip(clipped, min_weight, max_weight)
    clipped /= np.clip(clipped.sum(), 1e-9, None)
    return clipped.astype(np.float32)


def _mean_variance_baseline_reward(
    lookback_window: pd.DataFrame,
    forward_segment: pd.DataFrame | np.ndarray,
    *,
    periods_per_year: int = 252,
    risk_aversion: float = 3.0,
    max_weight: float = 0.35,
) -> float:
    """Compute a long-only mean-variance baseline reward for a realized forward segment."""
    if isinstance(forward_segment, pd.DataFrame):
        segment_frame = forward_segment.copy()
    else:
        segment_array = np.asarray(forward_segment, dtype=np.float32)
        segment_frame = pd.DataFrame(segment_array, columns=list(lookback_window.columns))

    risk_estimate = estimate_risk(
        lookback_window,
        EstimationRiskConfig(
            mean_estimator="historical",
            covariance_estimator="ledoit_wolf",
            annualize=True,
        ),
        periods_per_year=periods_per_year,
    )
    optimization_result = optimize_portfolio(
        risk_estimate.mean,
        risk_estimate.covariance,
        OptimizationConfig(
            objective="mean_variance",
            risk_aversion=float(risk_aversion),
            long_only=True,
            min_weight=0.0,
            max_weight=float(max_weight),
            turnover_penalty=0.0,
        ),
    )
    markowitz_action = _normalize_simplex(
        optimization_result.weights.reindex(segment_frame.columns).fillna(0.0).to_numpy(dtype=np.float32)
    )
    return _segment_cumulative_return(segment_frame.to_numpy(dtype=np.float32), markowitz_action)


def _compose_training_reward(
    raw_reward: float,
    *,
    benchmark_reward: float,
    equal_weight_reward: float,
    restricted_random_reward: float,
    markowitz_reward: float,
    reward_weight_raw: float = 0.10,
    reward_weight_vs_benchmark: float = 0.40,
    reward_weight_vs_equal_weight: float = 0.30,
    reward_weight_vs_restricted_random: float = 0.20,
    reward_weight_vs_markowitz: float = 0.0,
) -> float:
    weights = np.asarray(
        [
            float(reward_weight_raw),
            float(reward_weight_vs_benchmark),
            float(reward_weight_vs_equal_weight),
            float(reward_weight_vs_restricted_random),
            float(reward_weight_vs_markowitz),
        ],
        dtype=np.float64,
    )
    total_weight = float(np.abs(weights).sum())
    if total_weight <= 1e-12:
        return float(raw_reward)
    weights /= total_weight
    reward_components = np.asarray(
        [
            float(raw_reward),
            float(raw_reward - benchmark_reward),
            float(raw_reward - equal_weight_reward),
            float(raw_reward - restricted_random_reward),
            float(raw_reward - markowitz_reward),
        ],
        dtype=np.float64,
    )
    return float(weights @ reward_components)


def _state_features(window: pd.DataFrame) -> np.ndarray:
    values = window.to_numpy(dtype=np.float32)
    volatility = values.std(axis=0, keepdims=True) + 1e-6
    z_scores = values / volatility
    cumulative = np.cumsum(values, axis=0)
    stacked = np.stack([values, z_scores, cumulative], axis=-1)
    return np.transpose(stacked, (1, 0, 2)).astype(np.float32)


@dataclass(slots=True)
class RLTrainingConfig:
    """Configuration for offline actor-critic training."""

    lookback_window: int = 63
    hidden_dim: int = 240
    attention_heads: int = 8
    attention_layers: int = 4
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 180
    actor_bc_weight: float = 5.0
    entropy_weight: float = 1e-3
    validation_fraction: float = 0.20
    gradient_clip_norm: float = 1.0
    seed: int = 42


@dataclass(slots=True)
class OfflinePortfolioDataset:
    """Offline RL dataset built from saved portfolio weights and realized returns."""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    raw_rewards: np.ndarray
    benchmark_rewards: np.ndarray
    equal_weight_rewards: np.ndarray
    restricted_random_rewards: np.ndarray
    markowitz_rewards: np.ndarray
    forward_segments: list[np.ndarray]
    metadata: pd.DataFrame
    tickers: list[str]
    feature_names: list[str]
    lookback_window: int
    reward_weight_raw: float = 0.10
    reward_weight_vs_benchmark: float = 0.40
    reward_weight_vs_equal_weight: float = 0.30
    reward_weight_vs_restricted_random: float = 0.20
    reward_weight_vs_markowitz: float = 0.0

    def subset(self, indices: Sequence[int]) -> "OfflinePortfolioDataset":
        subset_indices = list(indices)
        return OfflinePortfolioDataset(
            states=self.states[subset_indices],
            actions=self.actions[subset_indices],
            rewards=self.rewards[subset_indices],
            raw_rewards=self.raw_rewards[subset_indices],
            benchmark_rewards=self.benchmark_rewards[subset_indices],
            equal_weight_rewards=self.equal_weight_rewards[subset_indices],
            restricted_random_rewards=self.restricted_random_rewards[subset_indices],
            markowitz_rewards=self.markowitz_rewards[subset_indices],
            forward_segments=[self.forward_segments[index] for index in subset_indices],
            metadata=self.metadata.iloc[subset_indices].reset_index(drop=True),
            tickers=self.tickers,
            feature_names=self.feature_names,
            lookback_window=self.lookback_window,
            reward_weight_raw=self.reward_weight_raw,
            reward_weight_vs_benchmark=self.reward_weight_vs_benchmark,
            reward_weight_vs_equal_weight=self.reward_weight_vs_equal_weight,
            reward_weight_vs_restricted_random=self.reward_weight_vs_restricted_random,
            reward_weight_vs_markowitz=self.reward_weight_vs_markowitz,
        )


@dataclass(slots=True)
class RLTrainingResult:
    """Artifacts returned after model training."""

    model: "CrossAssetAttentionActorCritic"
    config: RLTrainingConfig
    history: pd.DataFrame
    evaluation_summary: pd.DataFrame
    benchmark_summary: pd.DataFrame
    policy_predictions: pd.DataFrame
    latest_policy_weights: pd.Series
    tickers: list[str]
    selected_epoch: int
    model_score_summary: pd.DataFrame


@dataclass(slots=True)
class LoadedPolicyCheckpoint:
    """Loaded actor-critic checkpoint ready for inference."""

    path: Path
    model: nn.Module
    tickers: list[str]
    training_config: RLTrainingConfig
    device: torch.device


def load_weight_histories_from_suite(
    suite_root: str | Path,
    objectives: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Load per-objective weight history CSVs from a saved suite directory."""
    root = Path(suite_root)
    histories: dict[str, pd.DataFrame] = {}
    for objective in objectives:
        path = root / objective / "tables" / "weights_history.csv"
        if not path.exists():
            raise FileNotFoundError(f"Weight history not found for objective '{objective}': {path}")
        history = pd.read_csv(path, index_col=0, parse_dates=True)
        history.index = pd.to_datetime(history.index)
        history = history.sort_index()
        histories[objective] = history
    return histories


def build_offline_rl_dataset(
    returns: pd.DataFrame,
    weight_histories: dict[str, pd.DataFrame],
    *,
    lookback_window: int = 63,
    benchmark_ticker: str = "SPY",
    restricted_random_min_weight: float = 0.0,
    restricted_random_max_weight: float = 0.35,
    reward_weight_raw: float = 0.10,
    reward_weight_vs_benchmark: float = 0.40,
    reward_weight_vs_equal_weight: float = 0.30,
    reward_weight_vs_restricted_random: float = 0.20,
    reward_weight_vs_markowitz: float = 0.0,
    markowitz_risk_aversion: float = 3.0,
    markowitz_max_weight: float = 0.35,
) -> OfflinePortfolioDataset:
    """Create an offline RL dataset from saved objective-suite weight histories."""
    if not weight_histories:
        raise ValueError("At least one weight history is required to build the RL dataset.")

    tickers = list(next(iter(weight_histories.values())).columns)
    feature_names = ["return", "z_score", "cumulative_return"]

    records: list[dict[str, object]] = []
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    raw_rewards: list[float] = []
    benchmark_rewards: list[float] = []
    equal_weight_rewards: list[float] = []
    restricted_random_rewards: list[float] = []
    markowitz_rewards: list[float] = []
    forward_segments: list[np.ndarray] = []

    record_position = 0
    for objective, history in weight_histories.items():
        aligned_history = history.reindex(columns=tickers).fillna(0.0).sort_index()
        for position, rebalance_date in enumerate(aligned_history.index):
            window = returns.loc[:rebalance_date, tickers].iloc[-lookback_window:]
            if len(window) < lookback_window:
                continue

            start_idx = returns.index.get_loc(rebalance_date) + 1
            if position < len(aligned_history.index) - 1:
                end_idx = returns.index.get_loc(aligned_history.index[position + 1])
            else:
                end_idx = len(returns.index) - 1
            segment = returns.iloc[start_idx : end_idx + 1][tickers]
            if segment.empty:
                continue

            action = _normalize_simplex(aligned_history.loc[rebalance_date].to_numpy(dtype=np.float32))
            raw_reward = _segment_cumulative_return(segment.to_numpy(dtype=np.float32), action)
            equal_weight_reward = float(np.prod(1.0 + segment.to_numpy(dtype=np.float32).mean(axis=1)) - 1.0)

            if benchmark_ticker == "__equal_weight__":
                benchmark_reward = equal_weight_reward
            else:
                if benchmark_ticker not in returns.columns:
                    raise ValueError(f"Benchmark ticker '{benchmark_ticker}' is not available in the return panel.")
                benchmark_segment = returns.iloc[start_idx : end_idx + 1][benchmark_ticker].to_numpy(dtype=np.float32)
                benchmark_reward = float(np.prod(1.0 + benchmark_segment) - 1.0)
            restricted_random_action = _restricted_random_weights(
                len(tickers),
                seed=_stable_seed(objective, pd.Timestamp(rebalance_date).isoformat(), len(segment)),
                min_weight=restricted_random_min_weight,
                max_weight=restricted_random_max_weight,
            )
            restricted_random_reward = _segment_cumulative_return(segment.to_numpy(dtype=np.float32), restricted_random_action)
            markowitz_reward = _mean_variance_baseline_reward(
                window,
                segment,
                periods_per_year=252,
                risk_aversion=markowitz_risk_aversion,
                max_weight=markowitz_max_weight,
            )
            excess_reward = raw_reward - benchmark_reward
            composite_reward = _compose_training_reward(
                raw_reward,
                benchmark_reward=benchmark_reward,
                equal_weight_reward=equal_weight_reward,
                restricted_random_reward=restricted_random_reward,
                markowitz_reward=markowitz_reward,
                reward_weight_raw=reward_weight_raw,
                reward_weight_vs_benchmark=reward_weight_vs_benchmark,
                reward_weight_vs_equal_weight=reward_weight_vs_equal_weight,
                reward_weight_vs_restricted_random=reward_weight_vs_restricted_random,
                reward_weight_vs_markowitz=reward_weight_vs_markowitz,
            )

            states.append(_state_features(window))
            actions.append(action.astype(np.float32))
            rewards.append(float(composite_reward))
            raw_rewards.append(float(raw_reward))
            benchmark_rewards.append(float(benchmark_reward))
            equal_weight_rewards.append(float(equal_weight_reward))
            restricted_random_rewards.append(float(restricted_random_reward))
            markowitz_rewards.append(float(markowitz_reward))
            forward_segments.append(segment.to_numpy(dtype=np.float32))
            records.append(
                {
                    "record_position": record_position,
                    "objective": objective,
                    "rebalance_date": pd.Timestamp(rebalance_date),
                    "forward_start": segment.index[0],
                    "forward_end": segment.index[-1],
                    "raw_reward": raw_reward,
                    "benchmark_reward": benchmark_reward,
                    "equal_weight_reward": equal_weight_reward,
                    "restricted_random_reward": restricted_random_reward,
                    "markowitz_reward": markowitz_reward,
                    "excess_reward": excess_reward,
                    "composite_reward": composite_reward,
                }
            )
            record_position += 1

    if not states:
        raise ValueError("No offline RL samples were created. Check lookback length and available weight histories.")

    metadata = pd.DataFrame(records).sort_values(["rebalance_date", "objective"]).reset_index(drop=True)
    order = metadata["record_position"].to_numpy(dtype=int)
    metadata = metadata.drop(columns=["record_position"])

    return OfflinePortfolioDataset(
        states=np.stack(states, axis=0)[order],
        actions=np.stack(actions, axis=0)[order],
        rewards=np.asarray(rewards, dtype=np.float32)[order],
        raw_rewards=np.asarray(raw_rewards, dtype=np.float32)[order],
        benchmark_rewards=np.asarray(benchmark_rewards, dtype=np.float32)[order],
        equal_weight_rewards=np.asarray(equal_weight_rewards, dtype=np.float32)[order],
        restricted_random_rewards=np.asarray(restricted_random_rewards, dtype=np.float32)[order],
        markowitz_rewards=np.asarray(markowitz_rewards, dtype=np.float32)[order],
        forward_segments=[forward_segments[index] for index in order],
        metadata=metadata,
        tickers=tickers,
        feature_names=feature_names,
        lookback_window=lookback_window,
        reward_weight_raw=reward_weight_raw,
        reward_weight_vs_benchmark=reward_weight_vs_benchmark,
        reward_weight_vs_equal_weight=reward_weight_vs_equal_weight,
        reward_weight_vs_restricted_random=reward_weight_vs_restricted_random,
        reward_weight_vs_markowitz=reward_weight_vs_markowitz,
    )


class CrossAssetAttentionActorCritic(nn.Module):
    """Transformer-style actor-critic with cross-asset attention and Dirichlet policy head."""

    def __init__(
        self,
        *,
        num_assets: int,
        lookback_window: int,
        feature_dim: int,
        hidden_dim: int = 64,
        attention_heads: int = 4,
        attention_layers: int = 2,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.num_assets = num_assets
        self.lookback_window = lookback_window
        self.feature_dim = feature_dim

        self.input_projection = nn.Sequential(
            nn.Linear(lookback_window * feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.asset_embedding = nn.Parameter(torch.zeros(1, num_assets, hidden_dim))
        nn.init.normal_(self.asset_embedding, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.cross_asset_attention = nn.TransformerEncoder(encoder_layer, num_layers=attention_layers)

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim + num_assets, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        batch_size, num_assets, lookback, features = states.shape
        flattened = states.reshape(batch_size, num_assets, lookback * features)
        if num_assets <= self.num_assets:
            asset_embedding = self.asset_embedding[:, :num_assets, :]
        else:
            repeated_tail = self.asset_embedding[:, -1:, :].expand(1, num_assets - self.num_assets, -1)
            asset_embedding = torch.cat([self.asset_embedding, repeated_tail], dim=1)
        embeddings = self.input_projection(flattened) + asset_embedding
        return self.cross_asset_attention(embeddings)

    def policy_distribution(self, states: torch.Tensor) -> tuple[Dirichlet, torch.Tensor, torch.Tensor]:
        context = self.encode(states)
        logits = self.actor_head(context).squeeze(-1)
        concentration = F.softplus(logits) + 1.0
        distribution = Dirichlet(concentration)
        policy_mean = concentration / concentration.sum(dim=-1, keepdim=True)
        return distribution, policy_mean, context

    def q_values(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        pooled = context.mean(dim=1)
        critic_input = torch.cat([pooled, actions], dim=-1)
        return self.critic_head(critic_input).squeeze(-1)


class NearestNeighborPolicy(nn.Module):
    """Non-parametric policy that reuses the closest fitted state/action example."""

    def __init__(self, stored_states: np.ndarray, stored_actions: np.ndarray) -> None:
        super().__init__()
        flattened_states = np.asarray(stored_states, dtype=np.float32)
        if flattened_states.ndim != 4:
            raise ValueError("Nearest-neighbor policy expects states with shape [samples, assets, lookback, features].")
        action_array = np.asarray(stored_actions, dtype=np.float32)
        if action_array.ndim != 2:
            raise ValueError("Nearest-neighbor policy expects actions with shape [samples, assets].")
        if len(flattened_states) != len(action_array):
            raise ValueError("Nearest-neighbor policy requires the same number of states and actions.")
        self.max_assets = int(flattened_states.shape[1])
        self.lookback_window = int(flattened_states.shape[2])
        self.feature_dim = int(flattened_states.shape[3])
        self.register_buffer(
            "stored_states",
            torch.as_tensor(flattened_states.reshape(len(flattened_states), -1), dtype=torch.float32),
        )
        self.register_buffer("stored_actions", torch.as_tensor(action_array, dtype=torch.float32))

    def _align_states(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 4:
            raise ValueError("Nearest-neighbor policy expects batched states with shape [batch, assets, lookback, features].")
        batch_size, asset_count, lookback_window, feature_dim = states.shape
        if lookback_window != self.lookback_window or feature_dim != self.feature_dim:
            raise ValueError(
                "Nearest-neighbor policy received an incompatible state tensor "
                f"({asset_count} assets, lb={lookback_window}, feat={feature_dim}); "
                f"expected lb={self.lookback_window}, feat={self.feature_dim}."
            )
        if asset_count < self.max_assets:
            padding = torch.zeros(
                batch_size,
                self.max_assets - asset_count,
                lookback_window,
                feature_dim,
                dtype=states.dtype,
                device=states.device,
            )
            states = torch.cat([states, padding], dim=1)
        elif asset_count > self.max_assets:
            states = states[:, : self.max_assets]
        return states

    def policy_distribution(self, states: torch.Tensor) -> tuple[Dirichlet, torch.Tensor, torch.Tensor]:
        aligned_states = self._align_states(states)
        batch_size, asset_count = states.shape[:2]
        flattened = aligned_states.reshape(len(aligned_states), -1)
        distances = torch.cdist(flattened, self.stored_states)
        nearest_indices = torch.argmin(distances, dim=1)
        full_actions = self.stored_actions.index_select(0, nearest_indices)
        policy_mean = full_actions[:, :asset_count]
        policy_mean = policy_mean / torch.clamp(policy_mean.sum(dim=-1, keepdim=True), min=1e-6)
        concentration = torch.clamp(policy_mean, min=1e-4) * 500.0 + 1.0
        distribution = Dirichlet(concentration)
        return distribution, policy_mean, policy_mean.unsqueeze(-1)

    def q_values(self, context: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(len(actions), dtype=actions.dtype, device=actions.device)


def _evaluate_policy_returns(
    actions: np.ndarray,
    forward_segments: Sequence[np.ndarray],
    benchmark_rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_policy_returns = np.asarray(
        [_segment_cumulative_return(segment, action) for segment, action in zip(forward_segments, actions, strict=True)],
        dtype=np.float32,
    )
    policy_excess_returns = raw_policy_returns - benchmark_rewards
    return raw_policy_returns, policy_excess_returns


def _compose_reward_vector(
    policy_raw_returns: np.ndarray,
    *,
    benchmark_rewards: np.ndarray,
    equal_weight_rewards: np.ndarray,
    restricted_random_rewards: np.ndarray,
    markowitz_rewards: np.ndarray,
    reward_weight_raw: float = 0.10,
    reward_weight_vs_benchmark: float = 0.40,
    reward_weight_vs_equal_weight: float = 0.30,
    reward_weight_vs_restricted_random: float = 0.20,
    reward_weight_vs_markowitz: float = 0.0,
) -> np.ndarray:
    rewards = [
        _compose_training_reward(
            float(raw_return),
            benchmark_reward=float(benchmark_reward),
            equal_weight_reward=float(equal_weight_reward),
            restricted_random_reward=float(restricted_random_reward),
            markowitz_reward=float(markowitz_reward),
            reward_weight_raw=reward_weight_raw,
            reward_weight_vs_benchmark=reward_weight_vs_benchmark,
            reward_weight_vs_equal_weight=reward_weight_vs_equal_weight,
            reward_weight_vs_restricted_random=reward_weight_vs_restricted_random,
            reward_weight_vs_markowitz=reward_weight_vs_markowitz,
        )
        for raw_return, benchmark_reward, equal_weight_reward, restricted_random_reward, markowitz_reward in zip(
            policy_raw_returns,
            benchmark_rewards,
            equal_weight_rewards,
            restricted_random_rewards,
            markowitz_rewards,
            strict=True,
        )
    ]
    return np.asarray(rewards, dtype=np.float32)


def _predict_policy_actions(
    model: CrossAssetAttentionActorCritic,
    states: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch_states = torch.as_tensor(states[start : start + batch_size], dtype=torch.float32, device=device)
            _, policy_mean, _ = model.policy_distribution(batch_states)
            outputs.append(policy_mean.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _training_config_from_dict(raw_config: dict[str, object] | None) -> RLTrainingConfig:
    """Hydrate a training config from a serialized checkpoint payload."""
    if not raw_config:
        return RLTrainingConfig()
    return RLTrainingConfig(**raw_config)


def load_actor_critic_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> LoadedPolicyCheckpoint:
    """Load a serialized actor-critic checkpoint for deterministic inference."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    payload = torch.load(path, map_location=target_device, weights_only=False)
    tickers = list(payload.get("tickers", []))
    if not tickers:
        raise ValueError(f"Checkpoint {path} does not contain any ticker metadata.")

    training_config = _training_config_from_dict(payload.get("training_config"))
    if payload.get("policy_kind") == "nearest_neighbor":
        stored_states = payload.get("nearest_neighbor_states")
        stored_actions = payload.get("nearest_neighbor_actions")
        if stored_states is None or stored_actions is None:
            raise ValueError(f"Checkpoint {path} does not contain nearest-neighbor state/action payloads.")
        model = NearestNeighborPolicy(stored_states, stored_actions).to(target_device)
        model.eval()
    else:
        model = CrossAssetAttentionActorCritic(
            num_assets=len(tickers),
            lookback_window=training_config.lookback_window,
            feature_dim=3,
            hidden_dim=training_config.hidden_dim,
            attention_heads=training_config.attention_heads,
            attention_layers=training_config.attention_layers,
            dropout=training_config.dropout,
        ).to(target_device)
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError(f"Checkpoint {path} does not contain a valid model state dict.")
        model.load_state_dict(state_dict)
        model.eval()

    return LoadedPolicyCheckpoint(
        path=path,
        model=model,
        tickers=tickers,
        training_config=training_config,
        device=target_device,
    )


def build_policy_state(
    returns_window: pd.DataFrame,
    *,
    tickers: Sequence[str] | None = None,
    lookback_window: int,
) -> np.ndarray:
    """Build a single policy state tensor from a trailing return window."""
    ordered_window = returns_window.reindex(columns=list(tickers)) if tickers is not None else returns_window.copy()
    if ordered_window.isna().any().any():
        raise ValueError("Return window contains missing values after ticker alignment.")
    if len(ordered_window) != lookback_window:
        raise ValueError(
            f"Expected a return window with {lookback_window} rows but received {len(ordered_window)}."
        )
    return _state_features(ordered_window)


def predict_policy_weights(
    checkpoint: LoadedPolicyCheckpoint,
    returns_window: pd.DataFrame,
    *,
    tickers: Sequence[str] | None = None,
) -> pd.Series:
    """Predict deterministic portfolio weights from a trailing return window."""
    output_tickers = list(tickers) if tickers is not None else list(returns_window.columns)
    state = build_policy_state(
        returns_window,
        tickers=output_tickers,
        lookback_window=checkpoint.training_config.lookback_window,
    )
    batch = torch.as_tensor(state[None, ...], dtype=torch.float32, device=checkpoint.device)
    checkpoint.model.eval()
    with torch.no_grad():
        _, policy_mean, _ = checkpoint.model.policy_distribution(batch)
    weights = policy_mean.squeeze(0).detach().cpu().numpy()
    series = pd.Series(weights, index=output_tickers, name="policy_weight")
    series.index.name = "Ticker"
    return series


def _one_sided_greater_test(values: np.ndarray) -> tuple[float, float]:
    """Return a one-sided t-statistic and p-value for mean(values) > 0."""
    sample = np.asarray(values, dtype=np.float64)
    if len(sample) < 2:
        return float("nan"), float("nan")
    result = stats.ttest_1samp(sample, 0.0, alternative="greater")
    return float(result.statistic), float(result.pvalue)


def _build_benchmark_summary(
    *,
    train_policy_raw: np.ndarray,
    train_policy_excess: np.ndarray,
    train_benchmark_raw: np.ndarray,
    train_equal_weight_raw: np.ndarray,
    train_restricted_random_raw: np.ndarray,
    train_markowitz_raw: np.ndarray,
    validation_policy_raw: np.ndarray,
    validation_policy_excess: np.ndarray,
    validation_benchmark_raw: np.ndarray,
    validation_equal_weight_raw: np.ndarray,
    validation_restricted_random_raw: np.ndarray,
    validation_markowitz_raw: np.ndarray,
    full_policy_raw: np.ndarray,
    full_policy_excess: np.ndarray,
    full_benchmark_raw: np.ndarray,
    full_equal_weight_raw: np.ndarray,
    full_restricted_random_raw: np.ndarray,
    full_markowitz_raw: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize policy performance against the benchmark with significance tests."""
    rows: dict[str, dict[str, float | bool]] = {}
    for split_name, policy_raw, policy_excess, benchmark_raw, equal_weight_raw, restricted_random_raw, markowitz_raw in [
        ("train", train_policy_raw, train_policy_excess, train_benchmark_raw, train_equal_weight_raw, train_restricted_random_raw, train_markowitz_raw),
        (
            "validation",
            validation_policy_raw,
            validation_policy_excess,
            validation_benchmark_raw,
            validation_equal_weight_raw,
            validation_restricted_random_raw,
            validation_markowitz_raw,
        ),
        ("all", full_policy_raw, full_policy_excess, full_benchmark_raw, full_equal_weight_raw, full_restricted_random_raw, full_markowitz_raw),
    ]:
        t_statistic, p_value = _one_sided_greater_test(policy_excess)
        equal_weight_excess = np.asarray(policy_raw, dtype=np.float64) - np.asarray(equal_weight_raw, dtype=np.float64)
        equal_weight_t, equal_weight_p = _one_sided_greater_test(equal_weight_excess)
        restricted_random_excess = np.asarray(policy_raw, dtype=np.float64) - np.asarray(restricted_random_raw, dtype=np.float64)
        restricted_random_t, restricted_random_p = _one_sided_greater_test(restricted_random_excess)
        markowitz_excess = np.asarray(policy_raw, dtype=np.float64) - np.asarray(markowitz_raw, dtype=np.float64)
        markowitz_t, markowitz_p = _one_sided_greater_test(markowitz_excess)
        rows[split_name] = {
            "samples": len(policy_excess),
            "benchmark_mean_raw_return": float(np.mean(benchmark_raw)),
            "equal_weight_mean_raw_return": float(np.mean(equal_weight_raw)),
            "restricted_random_mean_raw_return": float(np.mean(restricted_random_raw)),
            "markowitz_mean_raw_return": float(np.mean(markowitz_raw)),
            "policy_mean_raw_return": float(np.mean(policy_raw)),
            "policy_mean_excess_return": float(np.mean(policy_excess)),
            "policy_mean_excess_vs_equal_weight": float(np.mean(equal_weight_excess)),
            "policy_mean_excess_vs_restricted_random": float(np.mean(restricted_random_excess)),
            "policy_mean_excess_vs_markowitz": float(np.mean(markowitz_excess)),
            "t_statistic": t_statistic,
            "p_value": p_value,
            "significant_outperformance": bool(np.isfinite(p_value) and p_value < alpha and np.mean(policy_excess) > 0.0),
            "equal_weight_t_statistic": equal_weight_t,
            "equal_weight_p_value": equal_weight_p,
            "equal_weight_significant_outperformance": bool(
                np.isfinite(equal_weight_p) and equal_weight_p < alpha and np.mean(equal_weight_excess) > 0.0
            ),
            "restricted_random_t_statistic": restricted_random_t,
            "restricted_random_p_value": restricted_random_p,
            "restricted_random_significant_outperformance": bool(
                np.isfinite(restricted_random_p)
                and restricted_random_p < alpha
                and np.mean(restricted_random_excess) > 0.0
            ),
            "markowitz_t_statistic": markowitz_t,
            "markowitz_p_value": markowitz_p,
            "markowitz_significant_outperformance": bool(
                np.isfinite(markowitz_p) and markowitz_p < alpha and np.mean(markowitz_excess) > 0.0
            ),
        }
    summary = pd.DataFrame(rows).T
    summary.index.name = "Split"
    return summary


def _benchmark_selection_key(
    benchmark_excess_returns: np.ndarray,
    equal_weight_excess_returns: np.ndarray,
    restricted_random_excess_returns: np.ndarray,
    markowitz_excess_returns: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Rank checkpoints by multi-baseline significance first, then excess-return breadth."""
    benchmark_t, benchmark_p = _one_sided_greater_test(benchmark_excess_returns)
    equal_weight_t, equal_weight_p = _one_sided_greater_test(equal_weight_excess_returns)
    restricted_random_t, restricted_random_p = _one_sided_greater_test(restricted_random_excess_returns)
    markowitz_t, markowitz_p = _one_sided_greater_test(markowitz_excess_returns)
    significance_count = float(
        sum(
            [
                np.isfinite(benchmark_p) and benchmark_p < 0.05 and np.mean(benchmark_excess_returns) > 0.0,
                np.isfinite(equal_weight_p) and equal_weight_p < 0.05 and np.mean(equal_weight_excess_returns) > 0.0,
                np.isfinite(restricted_random_p)
                and restricted_random_p < 0.05
                and np.mean(restricted_random_excess_returns) > 0.0,
                np.isfinite(markowitz_p) and markowitz_p < 0.05 and np.mean(markowitz_excess_returns) > 0.0,
            ]
        )
    )
    average_excess = float(
        np.mean(
            [
                np.mean(benchmark_excess_returns),
                np.mean(equal_weight_excess_returns),
                np.mean(restricted_random_excess_returns),
                np.mean(markowitz_excess_returns),
            ]
        )
    )
    average_t = float(np.mean([benchmark_t, equal_weight_t, restricted_random_t, markowitz_t]))
    markowitz_mean = float(np.mean(markowitz_excess_returns))
    benchmark_mean = float(np.mean(benchmark_excess_returns))
    safe_average_t = average_t if np.isfinite(average_t) else float("-inf")
    return significance_count, average_excess, markowitz_mean, benchmark_mean, safe_average_t


def train_transformer_actor_critic(
    dataset: OfflinePortfolioDataset,
    config: RLTrainingConfig,
    *,
    device: str | torch.device | None = None,
    progress_callback: Callable[[int, pd.DataFrame], None] | None = None,
) -> RLTrainingResult:
    """Train a transformer actor-critic policy from offline demonstrations."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_assets = len(dataset.tickers)
    feature_dim = dataset.states.shape[-1]
    split_index = max(1, int(len(dataset.states) * (1.0 - config.validation_fraction)))
    split_index = min(split_index, len(dataset.states) - 1) if len(dataset.states) > 1 else len(dataset.states)

    train_dataset = dataset.subset(range(split_index))
    validation_dataset = dataset.subset(range(split_index, len(dataset.states))) if split_index < len(dataset.states) else dataset.subset(range(len(dataset.states)))

    reward_mean = float(train_dataset.rewards.mean())
    reward_std = float(train_dataset.rewards.std() + 1e-6)

    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(train_dataset.states, dtype=torch.float32),
            torch.as_tensor(train_dataset.actions, dtype=torch.float32),
            torch.as_tensor((train_dataset.rewards - reward_mean) / reward_std, dtype=torch.float32),
        ),
        batch_size=min(config.batch_size, len(train_dataset.states)),
        shuffle=True,
    )

    model = CrossAssetAttentionActorCritic(
        num_assets=num_assets,
        lookback_window=dataset.lookback_window,
        feature_dim=feature_dim,
        hidden_dim=config.hidden_dim,
        attention_heads=config.attention_heads,
        attention_layers=config.attention_layers,
        dropout=config.dropout,
    ).to(target_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history_rows: list[dict[str, float]] = []
    best_epoch = 0
    best_validation_key: tuple[float, float, float, float] | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "actor": 0.0, "critic": 0.0, "bc": 0.0, "entropy": 0.0, "samples": 0}

        for states_batch, actions_batch, rewards_batch in train_loader:
            states_batch = states_batch.to(target_device)
            actions_batch = actions_batch.to(target_device)
            rewards_batch = rewards_batch.to(target_device)

            optimizer.zero_grad()
            distribution, policy_mean, context = model.policy_distribution(states_batch)
            q_demo = model.q_values(context, actions_batch)
            q_policy = model.q_values(context, policy_mean)

            bc_loss = F.mse_loss(policy_mean, actions_batch)
            critic_loss = F.mse_loss(q_demo, rewards_batch)
            entropy = distribution.entropy().mean()
            actor_loss = -q_policy.mean() + config.actor_bc_weight * bc_loss - config.entropy_weight * entropy
            loss = actor_loss + critic_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()

            batch_size = len(states_batch)
            totals["loss"] += float(loss.item()) * batch_size
            totals["actor"] += float(actor_loss.item()) * batch_size
            totals["critic"] += float(critic_loss.item()) * batch_size
            totals["bc"] += float(bc_loss.item()) * batch_size
            totals["entropy"] += float(entropy.item()) * batch_size
            totals["samples"] += batch_size

        train_policy_actions = _predict_policy_actions(
            model,
            train_dataset.states,
            batch_size=config.batch_size,
            device=target_device,
        )
        validation_policy_actions = _predict_policy_actions(
            model,
            validation_dataset.states,
            batch_size=config.batch_size,
            device=target_device,
        )

        train_policy_raw, train_policy_excess = _evaluate_policy_returns(
            train_policy_actions,
            train_dataset.forward_segments,
            train_dataset.benchmark_rewards,
        )
        validation_policy_raw, validation_policy_excess = _evaluate_policy_returns(
            validation_policy_actions,
            validation_dataset.forward_segments,
            validation_dataset.benchmark_rewards,
        )
        validation_policy_excess_vs_equal_weight = validation_policy_raw - validation_dataset.equal_weight_rewards
        validation_policy_excess_vs_restricted_random = (
            validation_policy_raw - validation_dataset.restricted_random_rewards
        )
        validation_policy_excess_vs_markowitz = validation_policy_raw - validation_dataset.markowitz_rewards
        validation_t_statistic, validation_p_value = _one_sided_greater_test(validation_policy_excess)
        validation_key = _benchmark_selection_key(
            validation_policy_excess,
            validation_policy_excess_vs_equal_weight,
            validation_policy_excess_vs_restricted_random,
            validation_policy_excess_vs_markowitz,
        )
        if best_validation_key is None or validation_key > best_validation_key:
            best_validation_key = validation_key
            best_epoch = epoch
            best_state_dict = deepcopy(model.state_dict())

        history_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": totals["loss"] / max(totals["samples"], 1),
                "train_actor_loss": totals["actor"] / max(totals["samples"], 1),
                "train_critic_loss": totals["critic"] / max(totals["samples"], 1),
                "train_bc_loss": totals["bc"] / max(totals["samples"], 1),
                "train_entropy": totals["entropy"] / max(totals["samples"], 1),
                "train_demo_training_reward": float(train_dataset.rewards.mean()),
                "train_demo_excess_return": float((train_dataset.raw_rewards - train_dataset.benchmark_rewards).mean()),
                "train_policy_excess_return": float(train_policy_excess.mean()),
                "train_policy_training_reward": float(
                    _compose_reward_vector(
                        train_policy_raw,
                        benchmark_rewards=train_dataset.benchmark_rewards,
                        equal_weight_rewards=train_dataset.equal_weight_rewards,
                        restricted_random_rewards=train_dataset.restricted_random_rewards,
                        markowitz_rewards=train_dataset.markowitz_rewards,
                        reward_weight_raw=dataset.reward_weight_raw,
                        reward_weight_vs_benchmark=dataset.reward_weight_vs_benchmark,
                        reward_weight_vs_equal_weight=dataset.reward_weight_vs_equal_weight,
                        reward_weight_vs_restricted_random=dataset.reward_weight_vs_restricted_random,
                        reward_weight_vs_markowitz=dataset.reward_weight_vs_markowitz,
                    ).mean()
                ),
                "validation_demo_training_reward": float(validation_dataset.rewards.mean()),
                "validation_demo_excess_return": float(
                    (validation_dataset.raw_rewards - validation_dataset.benchmark_rewards).mean()
                ),
                "validation_policy_excess_return": float(validation_policy_excess.mean()),
                "validation_policy_training_reward": float(
                    _compose_reward_vector(
                        validation_policy_raw,
                        benchmark_rewards=validation_dataset.benchmark_rewards,
                        equal_weight_rewards=validation_dataset.equal_weight_rewards,
                        restricted_random_rewards=validation_dataset.restricted_random_rewards,
                        markowitz_rewards=validation_dataset.markowitz_rewards,
                        reward_weight_raw=dataset.reward_weight_raw,
                        reward_weight_vs_benchmark=dataset.reward_weight_vs_benchmark,
                        reward_weight_vs_equal_weight=dataset.reward_weight_vs_equal_weight,
                        reward_weight_vs_restricted_random=dataset.reward_weight_vs_restricted_random,
                        reward_weight_vs_markowitz=dataset.reward_weight_vs_markowitz,
                    ).mean()
                ),
                "validation_policy_excess_vs_markowitz": float(validation_policy_excess_vs_markowitz.mean()),
                "validation_t_statistic": float(validation_t_statistic),
                "validation_p_value": float(validation_p_value),
            }
        )
        if progress_callback is not None:
            progress_callback(epoch, pd.DataFrame(history_rows).set_index("epoch"))

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    train_policy_actions = _predict_policy_actions(
        model,
        train_dataset.states,
        batch_size=config.batch_size,
        device=target_device,
    )
    validation_policy_actions = _predict_policy_actions(
        model,
        validation_dataset.states,
        batch_size=config.batch_size,
        device=target_device,
    )
    full_policy_actions = _predict_policy_actions(
        model,
        dataset.states,
        batch_size=config.batch_size,
        device=target_device,
    )

    train_policy_raw, train_policy_excess = _evaluate_policy_returns(
        train_policy_actions,
        train_dataset.forward_segments,
        train_dataset.benchmark_rewards,
    )
    train_policy_excess_vs_equal_weight = train_policy_raw - train_dataset.equal_weight_rewards
    train_policy_excess_vs_restricted_random = train_policy_raw - train_dataset.restricted_random_rewards
    train_policy_excess_vs_markowitz = train_policy_raw - train_dataset.markowitz_rewards
    train_policy_training_rewards = _compose_reward_vector(
        train_policy_raw,
        benchmark_rewards=train_dataset.benchmark_rewards,
        equal_weight_rewards=train_dataset.equal_weight_rewards,
        restricted_random_rewards=train_dataset.restricted_random_rewards,
        markowitz_rewards=train_dataset.markowitz_rewards,
        reward_weight_raw=dataset.reward_weight_raw,
        reward_weight_vs_benchmark=dataset.reward_weight_vs_benchmark,
        reward_weight_vs_equal_weight=dataset.reward_weight_vs_equal_weight,
        reward_weight_vs_restricted_random=dataset.reward_weight_vs_restricted_random,
        reward_weight_vs_markowitz=dataset.reward_weight_vs_markowitz,
    )
    validation_policy_raw, validation_policy_excess = _evaluate_policy_returns(
        validation_policy_actions,
        validation_dataset.forward_segments,
        validation_dataset.benchmark_rewards,
    )
    validation_policy_excess_vs_equal_weight = validation_policy_raw - validation_dataset.equal_weight_rewards
    validation_policy_excess_vs_restricted_random = (
        validation_policy_raw - validation_dataset.restricted_random_rewards
    )
    validation_policy_excess_vs_markowitz = validation_policy_raw - validation_dataset.markowitz_rewards
    validation_policy_training_rewards = _compose_reward_vector(
        validation_policy_raw,
        benchmark_rewards=validation_dataset.benchmark_rewards,
        equal_weight_rewards=validation_dataset.equal_weight_rewards,
        restricted_random_rewards=validation_dataset.restricted_random_rewards,
        markowitz_rewards=validation_dataset.markowitz_rewards,
        reward_weight_raw=dataset.reward_weight_raw,
        reward_weight_vs_benchmark=dataset.reward_weight_vs_benchmark,
        reward_weight_vs_equal_weight=dataset.reward_weight_vs_equal_weight,
        reward_weight_vs_restricted_random=dataset.reward_weight_vs_restricted_random,
        reward_weight_vs_markowitz=dataset.reward_weight_vs_markowitz,
    )
    full_policy_raw, full_policy_excess = _evaluate_policy_returns(
        full_policy_actions,
        dataset.forward_segments,
        dataset.benchmark_rewards,
    )
    full_policy_excess_vs_equal_weight = full_policy_raw - dataset.equal_weight_rewards
    full_policy_excess_vs_restricted_random = full_policy_raw - dataset.restricted_random_rewards
    full_policy_excess_vs_markowitz = full_policy_raw - dataset.markowitz_rewards
    full_policy_training_rewards = _compose_reward_vector(
        full_policy_raw,
        benchmark_rewards=dataset.benchmark_rewards,
        equal_weight_rewards=dataset.equal_weight_rewards,
        restricted_random_rewards=dataset.restricted_random_rewards,
        markowitz_rewards=dataset.markowitz_rewards,
        reward_weight_raw=dataset.reward_weight_raw,
        reward_weight_vs_benchmark=dataset.reward_weight_vs_benchmark,
        reward_weight_vs_equal_weight=dataset.reward_weight_vs_equal_weight,
        reward_weight_vs_restricted_random=dataset.reward_weight_vs_restricted_random,
        reward_weight_vs_markowitz=dataset.reward_weight_vs_markowitz,
    )
    benchmark_summary = _build_benchmark_summary(
        train_policy_raw=train_policy_raw,
        train_policy_excess=train_policy_excess,
        train_benchmark_raw=train_dataset.benchmark_rewards,
        train_equal_weight_raw=train_dataset.equal_weight_rewards,
        train_restricted_random_raw=train_dataset.restricted_random_rewards,
        train_markowitz_raw=train_dataset.markowitz_rewards,
        validation_policy_raw=validation_policy_raw,
        validation_policy_excess=validation_policy_excess,
        validation_benchmark_raw=validation_dataset.benchmark_rewards,
        validation_equal_weight_raw=validation_dataset.equal_weight_rewards,
        validation_restricted_random_raw=validation_dataset.restricted_random_rewards,
        validation_markowitz_raw=validation_dataset.markowitz_rewards,
        full_policy_raw=full_policy_raw,
        full_policy_excess=full_policy_excess,
        full_benchmark_raw=dataset.benchmark_rewards,
        full_equal_weight_raw=dataset.equal_weight_rewards,
        full_restricted_random_raw=dataset.restricted_random_rewards,
        full_markowitz_raw=dataset.markowitz_rewards,
    )

    prediction_rows: list[dict[str, object]] = []
    for sample_idx, (
        metadata_row,
        demo_action,
        policy_action,
        demo_training_reward,
        demo_raw,
        demo_benchmark_raw,
        demo_equal_weight_raw,
        demo_restricted_random_raw,
        demo_markowitz_raw,
        policy_training_reward,
        policy_raw,
        policy_excess,
        policy_excess_vs_equal_weight,
        policy_excess_vs_restricted_random,
        policy_excess_vs_markowitz,
    ) in enumerate(
        zip(
            dataset.metadata.itertuples(index=False),
            dataset.actions,
            full_policy_actions,
            dataset.rewards,
            dataset.raw_rewards,
            dataset.benchmark_rewards,
            dataset.equal_weight_rewards,
            dataset.restricted_random_rewards,
            dataset.markowitz_rewards,
            full_policy_training_rewards,
            full_policy_raw,
            full_policy_excess,
            full_policy_excess_vs_equal_weight,
            full_policy_excess_vs_restricted_random,
            full_policy_excess_vs_markowitz,
            strict=True,
        )
    ):
        row = {
            "sample_id": sample_idx,
            "objective": metadata_row.objective,
            "rebalance_date": metadata_row.rebalance_date,
            "forward_start": metadata_row.forward_start,
            "forward_end": metadata_row.forward_end,
            "demo_training_reward": float(demo_training_reward),
            "demo_raw_return": float(demo_raw),
            "demo_benchmark_return": float(demo_benchmark_raw),
            "demo_equal_weight_return": float(demo_equal_weight_raw),
            "demo_restricted_random_return": float(demo_restricted_random_raw),
            "demo_markowitz_return": float(demo_markowitz_raw),
            "demo_excess_return": float(demo_raw - demo_benchmark_raw),
            "demo_excess_vs_equal_weight": float(demo_raw - demo_equal_weight_raw),
            "demo_excess_vs_restricted_random": float(demo_raw - demo_restricted_random_raw),
            "demo_excess_vs_markowitz": float(demo_raw - demo_markowitz_raw),
            "policy_training_reward": float(policy_training_reward),
            "policy_raw_return": float(policy_raw),
            "policy_excess_return": float(policy_excess),
            "policy_excess_vs_equal_weight": float(policy_excess_vs_equal_weight),
            "policy_excess_vs_restricted_random": float(policy_excess_vs_restricted_random),
            "policy_excess_vs_markowitz": float(policy_excess_vs_markowitz),
        }
        for ticker, demo_weight, policy_weight in zip(dataset.tickers, demo_action, policy_action, strict=True):
            row[f"demo_weight_{ticker}"] = float(demo_weight)
            row[f"policy_weight_{ticker}"] = float(policy_weight)
        prediction_rows.append(row)

    policy_predictions = pd.DataFrame(prediction_rows)
    policy_predictions.index.name = "SampleId"

    evaluation_summary = pd.DataFrame(
        {
            "train": {
                "samples": len(train_dataset.states),
                "demo_mean_training_reward": float(train_dataset.rewards.mean()),
                "policy_mean_excess_return": float(train_policy_excess.mean()),
                "policy_mean_training_reward": float(train_policy_training_rewards.mean()),
                "demo_mean_raw_return": float(train_dataset.raw_rewards.mean()),
                "demo_mean_excess_return": float((train_dataset.raw_rewards - train_dataset.benchmark_rewards).mean()),
                "demo_mean_excess_vs_equal_weight": float(
                    (train_dataset.raw_rewards - train_dataset.equal_weight_rewards).mean()
                ),
                "demo_mean_excess_vs_restricted_random": float(
                    (train_dataset.raw_rewards - train_dataset.restricted_random_rewards).mean()
                ),
                "demo_mean_excess_vs_markowitz": float(
                    (train_dataset.raw_rewards - train_dataset.markowitz_rewards).mean()
                ),
                "policy_mean_raw_return": float(train_policy_raw.mean()),
                "policy_mean_excess_vs_equal_weight": float(train_policy_excess_vs_equal_weight.mean()),
                "policy_mean_excess_vs_restricted_random": float(train_policy_excess_vs_restricted_random.mean()),
                "policy_mean_excess_vs_markowitz": float(train_policy_excess_vs_markowitz.mean()),
                "mean_abs_weight_error": float(np.mean(np.abs(train_policy_actions - train_dataset.actions))),
            },
            "validation": {
                "samples": len(validation_dataset.states),
                "demo_mean_training_reward": float(validation_dataset.rewards.mean()),
                "policy_mean_excess_return": float(validation_policy_excess.mean()),
                "policy_mean_training_reward": float(validation_policy_training_rewards.mean()),
                "demo_mean_raw_return": float(validation_dataset.raw_rewards.mean()),
                "demo_mean_excess_return": float(
                    (validation_dataset.raw_rewards - validation_dataset.benchmark_rewards).mean()
                ),
                "demo_mean_excess_vs_equal_weight": float(
                    (validation_dataset.raw_rewards - validation_dataset.equal_weight_rewards).mean()
                ),
                "demo_mean_excess_vs_restricted_random": float(
                    (validation_dataset.raw_rewards - validation_dataset.restricted_random_rewards).mean()
                ),
                "demo_mean_excess_vs_markowitz": float(
                    (validation_dataset.raw_rewards - validation_dataset.markowitz_rewards).mean()
                ),
                "policy_mean_raw_return": float(validation_policy_raw.mean()),
                "policy_mean_excess_vs_equal_weight": float(validation_policy_excess_vs_equal_weight.mean()),
                "policy_mean_excess_vs_restricted_random": float(
                    validation_policy_excess_vs_restricted_random.mean()
                ),
                "policy_mean_excess_vs_markowitz": float(validation_policy_excess_vs_markowitz.mean()),
                "mean_abs_weight_error": float(np.mean(np.abs(validation_policy_actions - validation_dataset.actions))),
            },
            "all": {
                "samples": len(dataset.states),
                "demo_mean_training_reward": float(dataset.rewards.mean()),
                "policy_mean_excess_return": float(full_policy_excess.mean()),
                "policy_mean_training_reward": float(full_policy_training_rewards.mean()),
                "demo_mean_raw_return": float(dataset.raw_rewards.mean()),
                "demo_mean_excess_return": float((dataset.raw_rewards - dataset.benchmark_rewards).mean()),
                "demo_mean_excess_vs_equal_weight": float(
                    (dataset.raw_rewards - dataset.equal_weight_rewards).mean()
                ),
                "demo_mean_excess_vs_restricted_random": float(
                    (dataset.raw_rewards - dataset.restricted_random_rewards).mean()
                ),
                "demo_mean_excess_vs_markowitz": float(
                    (dataset.raw_rewards - dataset.markowitz_rewards).mean()
                ),
                "policy_mean_raw_return": float(full_policy_raw.mean()),
                "policy_mean_excess_vs_equal_weight": float(full_policy_excess_vs_equal_weight.mean()),
                "policy_mean_excess_vs_restricted_random": float(
                    full_policy_excess_vs_restricted_random.mean()
                ),
                "policy_mean_excess_vs_markowitz": float(full_policy_excess_vs_markowitz.mean()),
                "mean_abs_weight_error": float(np.mean(np.abs(full_policy_actions - dataset.actions))),
            },
        }
    ).T
    evaluation_summary.index.name = "Split"
    model_score_summary = build_model_score_summary(benchmark_summary, evaluation_summary)

    latest_policy_weights = pd.Series(
        full_policy_actions[-1],
        index=dataset.tickers,
        name="policy_weight",
    )
    latest_policy_weights.index.name = "Ticker"

    history = pd.DataFrame(history_rows).set_index("epoch")
    history.index.name = "Epoch"
    history["selected_checkpoint"] = history.index == best_epoch

    return RLTrainingResult(
        model=model,
        config=config,
        history=history,
        evaluation_summary=evaluation_summary,
        benchmark_summary=benchmark_summary,
        policy_predictions=policy_predictions,
        latest_policy_weights=latest_policy_weights,
        tickers=dataset.tickers,
        selected_epoch=best_epoch,
        model_score_summary=model_score_summary,
    )


def save_actor_critic_artifacts(
    result: RLTrainingResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save model weights and evaluation artifacts to disk."""
    destination = ensure_directory(output_dir)
    figures_dir = ensure_directory(destination / "figures")

    config_path = destination / "rl_config.json"
    config_path.write_text(json.dumps(asdict(result.config), indent=2), encoding="utf-8")

    model_path = destination / "actor_critic_policy.pt"
    torch.save(
        {
            "state_dict": result.model.state_dict(),
            "tickers": result.tickers,
            "training_config": asdict(result.config),
            "selected_epoch": result.selected_epoch,
        },
        model_path,
    )

    paths = {
        "config": config_path,
        "model": model_path,
        "training_history": save_frame(result.history, destination / "training_history.csv"),
        "evaluation_summary": save_frame(result.evaluation_summary, destination / "evaluation_summary.csv"),
        "benchmark_summary": save_frame(result.benchmark_summary, destination / "benchmark_summary.csv"),
        "model_score_summary": save_frame(result.model_score_summary, destination / "model_score_summary.csv"),
        "policy_predictions": save_frame(result.policy_predictions.set_index("sample_id"), destination / "policy_predictions.csv"),
        "latest_policy_weights": save_frame(result.latest_policy_weights, destination / "latest_policy_weights.csv"),
        "training_diagnostics_fig": plot_rl_training_diagnostics(result.history, figures_dir / "training_diagnostics.png"),
        "benchmark_comparison_fig": plot_rl_benchmark_comparison(result.benchmark_summary, figures_dir / "benchmark_comparison.png"),
        "policy_cumulative_returns_fig": plot_rl_policy_cumulative_returns(
            result.policy_predictions,
            figures_dir / "policy_cumulative_returns.png",
        ),
        "latest_policy_weights_fig": plot_rl_latest_weights(result.latest_policy_weights, figures_dir / "latest_policy_weights.png"),
    }
    return paths

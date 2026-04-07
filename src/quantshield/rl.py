"""Core transformer actor-critic policy training on saved portfolio weight demonstrations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch.utils.data import DataLoader, TensorDataset

from quantshield.plotting import (
    plot_rl_benchmark_comparison,
    plot_rl_latest_weights,
    plot_rl_policy_cumulative_returns,
    plot_rl_training_diagnostics,
)
from quantshield.utils import ensure_directory, save_frame


def _normalize_simplex(weights: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    clipped = np.clip(weights, eps, None)
    return clipped / clipped.sum()


def _segment_cumulative_return(segment_returns: np.ndarray, weights: np.ndarray) -> float:
    daily_returns = segment_returns @ weights
    return float(np.prod(1.0 + daily_returns) - 1.0)


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
    hidden_dim: int = 192
    attention_heads: int = 6
    attention_layers: int = 4
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    epochs: int = 120
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
    forward_segments: list[np.ndarray]
    metadata: pd.DataFrame
    tickers: list[str]
    feature_names: list[str]
    lookback_window: int

    def subset(self, indices: Sequence[int]) -> "OfflinePortfolioDataset":
        subset_indices = list(indices)
        return OfflinePortfolioDataset(
            states=self.states[subset_indices],
            actions=self.actions[subset_indices],
            rewards=self.rewards[subset_indices],
            raw_rewards=self.raw_rewards[subset_indices],
            benchmark_rewards=self.benchmark_rewards[subset_indices],
            forward_segments=[self.forward_segments[index] for index in subset_indices],
            metadata=self.metadata.iloc[subset_indices].reset_index(drop=True),
            tickers=self.tickers,
            feature_names=self.feature_names,
            lookback_window=self.lookback_window,
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

            if benchmark_ticker not in returns.columns:
                raise ValueError(f"Benchmark ticker '{benchmark_ticker}' is not available in the return panel.")
            benchmark_segment = returns.iloc[start_idx : end_idx + 1][benchmark_ticker].to_numpy(dtype=np.float32)
            benchmark_reward = float(np.prod(1.0 + benchmark_segment) - 1.0)
            excess_reward = raw_reward - benchmark_reward

            states.append(_state_features(window))
            actions.append(action.astype(np.float32))
            rewards.append(float(excess_reward))
            raw_rewards.append(float(raw_reward))
            benchmark_rewards.append(float(benchmark_reward))
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
                    "excess_reward": excess_reward,
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
        forward_segments=[forward_segments[index] for index in order],
        metadata=metadata,
        tickers=tickers,
        feature_names=feature_names,
        lookback_window=lookback_window,
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
        if num_assets != self.num_assets:
            raise ValueError(f"Expected {self.num_assets} assets but received {num_assets}.")
        flattened = states.reshape(batch_size, num_assets, lookback * features)
        embeddings = self.input_projection(flattened) + self.asset_embedding[:, :num_assets, :]
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
    validation_policy_raw: np.ndarray,
    validation_policy_excess: np.ndarray,
    validation_benchmark_raw: np.ndarray,
    full_policy_raw: np.ndarray,
    full_policy_excess: np.ndarray,
    full_benchmark_raw: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Summarize policy performance against the benchmark with significance tests."""
    rows: dict[str, dict[str, float | bool]] = {}
    for split_name, policy_raw, policy_excess, benchmark_raw in [
        ("train", train_policy_raw, train_policy_excess, train_benchmark_raw),
        ("validation", validation_policy_raw, validation_policy_excess, validation_benchmark_raw),
        ("all", full_policy_raw, full_policy_excess, full_benchmark_raw),
    ]:
        t_statistic, p_value = _one_sided_greater_test(policy_excess)
        rows[split_name] = {
            "samples": len(policy_excess),
            "benchmark_mean_raw_return": float(np.mean(benchmark_raw)),
            "policy_mean_raw_return": float(np.mean(policy_raw)),
            "policy_mean_excess_return": float(np.mean(policy_excess)),
            "t_statistic": t_statistic,
            "p_value": p_value,
            "significant_outperformance": bool(np.isfinite(p_value) and p_value < alpha and np.mean(policy_excess) > 0.0),
        }
    summary = pd.DataFrame(rows).T
    summary.index.name = "Split"
    return summary


def train_transformer_actor_critic(
    dataset: OfflinePortfolioDataset,
    config: RLTrainingConfig,
    *,
    device: str | torch.device | None = None,
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

        history_rows.append(
            {
                "epoch": epoch,
                "train_total_loss": totals["loss"] / max(totals["samples"], 1),
                "train_actor_loss": totals["actor"] / max(totals["samples"], 1),
                "train_critic_loss": totals["critic"] / max(totals["samples"], 1),
                "train_bc_loss": totals["bc"] / max(totals["samples"], 1),
                "train_entropy": totals["entropy"] / max(totals["samples"], 1),
                "train_demo_excess_return": float(train_dataset.rewards.mean()),
                "train_policy_excess_return": float(train_policy_excess.mean()),
                "validation_demo_excess_return": float(validation_dataset.rewards.mean()),
                "validation_policy_excess_return": float(validation_policy_excess.mean()),
            }
        )

    full_policy_actions = _predict_policy_actions(
        model,
        dataset.states,
        batch_size=config.batch_size,
        device=target_device,
    )
    full_policy_raw, full_policy_excess = _evaluate_policy_returns(
        full_policy_actions,
        dataset.forward_segments,
        dataset.benchmark_rewards,
    )
    benchmark_summary = _build_benchmark_summary(
        train_policy_raw=train_policy_raw,
        train_policy_excess=train_policy_excess,
        train_benchmark_raw=train_dataset.benchmark_rewards,
        validation_policy_raw=validation_policy_raw,
        validation_policy_excess=validation_policy_excess,
        validation_benchmark_raw=validation_dataset.benchmark_rewards,
        full_policy_raw=full_policy_raw,
        full_policy_excess=full_policy_excess,
        full_benchmark_raw=dataset.benchmark_rewards,
    )

    prediction_rows: list[dict[str, object]] = []
    for sample_idx, (metadata_row, demo_action, policy_action, demo_raw, demo_excess, policy_raw, policy_excess) in enumerate(
        zip(
            dataset.metadata.itertuples(index=False),
            dataset.actions,
            full_policy_actions,
            dataset.raw_rewards,
            dataset.rewards,
            full_policy_raw,
            full_policy_excess,
            strict=True,
        )
    ):
        row = {
            "sample_id": sample_idx,
            "objective": metadata_row.objective,
            "rebalance_date": metadata_row.rebalance_date,
            "forward_start": metadata_row.forward_start,
            "forward_end": metadata_row.forward_end,
            "demo_raw_return": float(demo_raw),
            "demo_excess_return": float(demo_excess),
            "policy_raw_return": float(policy_raw),
            "policy_excess_return": float(policy_excess),
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
                "demo_mean_excess_return": float(train_dataset.rewards.mean()),
                "policy_mean_excess_return": float(train_policy_excess.mean()),
                "demo_mean_raw_return": float(train_dataset.raw_rewards.mean()),
                "policy_mean_raw_return": float(train_policy_raw.mean()),
                "mean_abs_weight_error": float(np.mean(np.abs(train_policy_actions - train_dataset.actions))),
            },
            "validation": {
                "samples": len(validation_dataset.states),
                "demo_mean_excess_return": float(validation_dataset.rewards.mean()),
                "policy_mean_excess_return": float(validation_policy_excess.mean()),
                "demo_mean_raw_return": float(validation_dataset.raw_rewards.mean()),
                "policy_mean_raw_return": float(validation_policy_raw.mean()),
                "mean_abs_weight_error": float(np.mean(np.abs(validation_policy_actions - validation_dataset.actions))),
            },
            "all": {
                "samples": len(dataset.states),
                "demo_mean_excess_return": float(dataset.rewards.mean()),
                "policy_mean_excess_return": float(full_policy_excess.mean()),
                "demo_mean_raw_return": float(dataset.raw_rewards.mean()),
                "policy_mean_raw_return": float(full_policy_raw.mean()),
                "mean_abs_weight_error": float(np.mean(np.abs(full_policy_actions - dataset.actions))),
            },
        }
    ).T
    evaluation_summary.index.name = "Split"

    latest_policy_weights = pd.Series(
        full_policy_actions[-1],
        index=dataset.tickers,
        name="policy_weight",
    )
    latest_policy_weights.index.name = "Ticker"

    history = pd.DataFrame(history_rows).set_index("epoch")
    history.index.name = "Epoch"

    return RLTrainingResult(
        model=model,
        config=config,
        history=history,
        evaluation_summary=evaluation_summary,
        benchmark_summary=benchmark_summary,
        policy_predictions=policy_predictions,
        latest_policy_weights=latest_policy_weights,
        tickers=dataset.tickers,
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
        },
        model_path,
    )

    paths = {
        "config": config_path,
        "model": model_path,
        "training_history": save_frame(result.history, destination / "training_history.csv"),
        "evaluation_summary": save_frame(result.evaluation_summary, destination / "evaluation_summary.csv"),
        "benchmark_summary": save_frame(result.benchmark_summary, destination / "benchmark_summary.csv"),
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

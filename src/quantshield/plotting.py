"""Matplotlib plotting functions for QuantShield reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quantshield.attribution import percentage_contribution_to_risk
from quantshield.metrics import cumulative_returns, drawdown_series
from quantshield.utils import ensure_directory


def _finalize_figure(path: str | Path) -> Path:
    output_path = Path(path)
    ensure_directory(output_path.parent)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def plot_price_history(prices: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    prices.plot(ax=ax, linewidth=1.2)
    ax.set_title("Price History")
    ax.set_ylabel("Adjusted Close")
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_correlation_heatmap(returns: pd.DataFrame, path: str | Path) -> Path:
    correlation = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(correlation.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(correlation.columns)))
    ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlation.index)))
    ax.set_yticklabels(correlation.index)
    ax.set_title("Return Correlation Heatmap")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return _finalize_figure(path)


def plot_cumulative_return_curves(returns: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    cumulative_returns(returns).plot(ax=ax, linewidth=1.4)
    ax.set_title("Cumulative Returns")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_rolling_volatility(returns: pd.DataFrame, path: str | Path, window: int = 63, periods_per_year: int = 252) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    rolling_vol = returns.rolling(window=window).std() * np.sqrt(periods_per_year)
    rolling_vol.plot(ax=ax, linewidth=1.2)
    ax.set_title(f"Rolling Volatility ({window} Days)")
    ax.set_ylabel("Annualized Volatility")
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_drawdown(returns: pd.Series, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    drawdown_series(returns).plot(ax=ax, linewidth=1.3)
    ax.fill_between(returns.index, drawdown_series(returns).values, 0.0, alpha=0.2)
    ax.set_title("Portfolio Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_weights_over_time(weights_history: pd.DataFrame, path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(weights_history) == 1:
        weights_history.T.plot(kind="bar", ax=ax)
    else:
        ax.stackplot(weights_history.index, weights_history.T.values, labels=weights_history.columns)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title("Portfolio Weights Over Time")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_risk_contribution(weights: pd.Series, covariance: pd.DataFrame, path: str | Path) -> Path:
    contribution = percentage_contribution_to_risk(weights, covariance).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    contribution.plot(kind="bar", ax=ax)
    ax.set_title("Risk Contribution by Asset")
    ax.set_ylabel("Percent of Portfolio Risk")
    ax.grid(axis="y", alpha=0.25)
    return _finalize_figure(path)


def plot_efficient_frontier(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    path: str | Path,
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    points: int = 25,
) -> Path:
    from quantshield.config import OptimizationConfig
    from quantshield.optimization import optimize_portfolio

    risk_aversion_grid = np.linspace(0.1, 10.0, points)
    frontier_points: list[tuple[float, float]] = []
    for risk_aversion in risk_aversion_grid:
        config = OptimizationConfig(
            objective="mean_variance",
            risk_aversion=float(risk_aversion),
            min_weight=min_weight,
            max_weight=max_weight,
        )
        result = optimize_portfolio(mean_returns, covariance, config)
        frontier_points.append((result.expected_volatility, result.expected_return))

    frontier = pd.DataFrame(frontier_points, columns=["volatility", "return"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(frontier["volatility"], frontier["return"], marker="o")
    ax.set_title("Approximate Efficient Frontier")
    ax.set_xlabel("Expected Volatility")
    ax.set_ylabel("Expected Return")
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_rl_training_diagnostics(history: pd.DataFrame, path: str | Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    loss_columns = [
        column
        for column in ["train_total_loss", "train_actor_loss", "train_critic_loss", "train_bc_loss"]
        if column in history.columns
    ]
    history[loss_columns].plot(ax=axes[0], linewidth=1.2)
    axes[0].set_title("RL Training Diagnostics")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.25)

    return_columns = [
        column
        for column in ["train_policy_excess_return", "validation_policy_excess_return"]
        if column in history.columns
    ]
    history[return_columns].plot(ax=axes[1], linewidth=1.4)
    if "train_demo_excess_return" in history.columns:
        axes[1].plot(history.index, history["train_demo_excess_return"], linestyle="--", linewidth=1.0, label="train_demo_excess_return")
    if "validation_demo_excess_return" in history.columns:
        axes[1].plot(
            history.index,
            history["validation_demo_excess_return"],
            linestyle="--",
            linewidth=1.0,
            label="validation_demo_excess_return",
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean Excess Return")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    return _finalize_figure(path)


def plot_rl_benchmark_comparison(benchmark_summary: pd.DataFrame, path: str | Path) -> Path:
    summary = benchmark_summary.copy()
    labels = summary.index.tolist()
    x = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(x - width / 2, summary["benchmark_mean_raw_return"], width=width, label="Benchmark")
    axes[0].bar(x + width / 2, summary["policy_mean_raw_return"], width=width, label="Policy")
    axes[0].set_title("Policy vs Benchmark Mean Raw Return")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Mean Return Per Rebalance")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    bars = axes[1].bar(x, summary["policy_mean_excess_return"])
    axes[1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[1].set_title("Policy Excess Return vs Benchmark")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Mean Excess Return Per Rebalance")
    axes[1].grid(axis="y", alpha=0.25)
    for bar, p_value, significant in zip(
        bars,
        summary["p_value"].to_numpy(),
        summary["significant_outperformance"].to_numpy(),
        strict=True,
    ):
        y = float(bar.get_height())
        offset = 0.001 if y >= 0.0 else -0.001
        va = "bottom" if y >= 0.0 else "top"
        label = f"p={p_value:.3g}"
        if bool(significant):
            label += " *"
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, y + offset, label, ha="center", va=va, fontsize=8)
    return _finalize_figure(path)


def plot_rl_policy_cumulative_returns(policy_predictions: pd.DataFrame, path: str | Path) -> Path:
    frame = policy_predictions.copy()
    benchmark_returns = frame["policy_raw_return"] - frame["policy_excess_return"]
    frame["benchmark_raw_return"] = benchmark_returns

    if "rebalance_date" in frame.columns:
        frame["rebalance_date"] = pd.to_datetime(frame["rebalance_date"])
        grouped = (
            frame.sort_values(["rebalance_date", "sample_id"])
            .groupby("rebalance_date")[["policy_raw_return", "demo_raw_return", "benchmark_raw_return"]]
            .mean()
        )
    else:
        grouped = frame[["policy_raw_return", "demo_raw_return", "benchmark_raw_return"]].copy()
        grouped.index = pd.RangeIndex(start=1, stop=len(grouped) + 1, name="RebalanceNumber")

    compounded = (1.0 + grouped).cumprod()
    compounded = compounded.rename(
        columns={
            "policy_raw_return": "Policy",
            "demo_raw_return": "Demonstration Average",
            "benchmark_raw_return": "Benchmark",
        }
    )

    fig, ax = plt.subplots(figsize=(11, 5.5))
    compounded.plot(ax=ax, linewidth=1.5)
    ax.set_title("Compounded Returns Across Rebalance Periods")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    return _finalize_figure(path)


def plot_rl_latest_weights(weights: pd.Series, path: str | Path) -> Path:
    ordered = weights.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ordered.plot(kind="bar", ax=ax)
    ax.set_title("Latest Policy Weights")
    ax.set_ylabel("Weight")
    ax.set_ylim(0.0, max(1.0, float(ordered.max()) * 1.15))
    ax.grid(axis="y", alpha=0.25)
    return _finalize_figure(path)

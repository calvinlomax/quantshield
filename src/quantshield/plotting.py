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

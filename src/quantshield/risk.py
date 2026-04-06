"""Risk estimators used by QuantShield."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass(slots=True)
class RiskEstimate:
    """Estimated portfolio inputs."""

    mean: pd.Series
    covariance: pd.DataFrame
    estimator: str
    sample_size: int


@dataclass(slots=True)
class RiskConfig:
    """Local estimator configuration."""

    mean_estimator: str = "historical"
    covariance_estimator: str = "ledoit_wolf"
    ewma_span: int = 60
    annualize: bool = True


def historical_mean(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Estimate expected returns from sample means."""
    mean = returns.mean()
    return mean * periods_per_year if annualize else mean


def historical_covariance(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Sample covariance estimator."""
    covariance = returns.cov()
    return covariance * periods_per_year if annualize else covariance


def ledoit_wolf_covariance(
    returns: pd.DataFrame,
    *,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance estimator."""
    estimator = LedoitWolf().fit(returns.values)
    covariance = pd.DataFrame(
        estimator.covariance_,
        index=returns.columns,
        columns=returns.columns,
    )
    return covariance * periods_per_year if annualize else covariance


def exponentially_weighted_covariance(
    returns: pd.DataFrame,
    *,
    span: int = 60,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Exponentially weighted covariance estimator using decaying observation weights."""
    if span < 2:
        raise ValueError("EWMA span must be at least 2.")

    alpha = 2.0 / (span + 1.0)
    weights = np.array([(1.0 - alpha) ** power for power in range(len(returns) - 1, -1, -1)], dtype=float)
    weights /= weights.sum()

    centered = returns - np.average(returns.values, axis=0, weights=weights)
    covariance = centered.mul(weights, axis=0).T.dot(centered)
    covariance = pd.DataFrame(covariance, index=returns.columns, columns=returns.columns)
    return covariance * periods_per_year if annualize else covariance


def compare_covariance_estimators(
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    ewma_span: int = 60,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Estimate several covariance matrices and summarize their properties."""
    estimators = {
        "historical": historical_covariance(returns, annualize=True, periods_per_year=periods_per_year),
        "ledoit_wolf": ledoit_wolf_covariance(returns, annualize=True, periods_per_year=periods_per_year),
        "ewma": exponentially_weighted_covariance(
            returns,
            span=ewma_span,
            annualize=True,
            periods_per_year=periods_per_year,
        ),
    }

    summary = pd.DataFrame(
        {
            name: {
                "trace": float(np.trace(matrix.values)),
                "condition_number": float(np.linalg.cond(matrix.values)),
                "average_variance": float(np.mean(np.diag(matrix.values))),
            }
            for name, matrix in estimators.items()
        }
    ).T
    return estimators, summary


def estimate_risk(
    returns: pd.DataFrame,
    config: RiskConfig,
    *,
    periods_per_year: int = 252,
) -> RiskEstimate:
    """Estimate expected returns and covariance from a historical sample."""
    if returns.empty:
        raise ValueError("Cannot estimate risk from an empty return frame.")

    mean = historical_mean(
        returns,
        annualize=config.annualize,
        periods_per_year=periods_per_year,
    )

    estimator_name = config.covariance_estimator.lower()
    if estimator_name == "historical":
        covariance = historical_covariance(
            returns,
            annualize=config.annualize,
            periods_per_year=periods_per_year,
        )
    elif estimator_name == "ledoit_wolf":
        covariance = ledoit_wolf_covariance(
            returns,
            annualize=config.annualize,
            periods_per_year=periods_per_year,
        )
    elif estimator_name in {"ewma", "exponentially_weighted"}:
        covariance = exponentially_weighted_covariance(
            returns,
            span=config.ewma_span,
            annualize=config.annualize,
            periods_per_year=periods_per_year,
        )
        estimator_name = "ewma"
    else:
        raise ValueError(f"Unsupported covariance estimator: {config.covariance_estimator}")

    return RiskEstimate(
        mean=mean,
        covariance=covariance,
        estimator=estimator_name,
        sample_size=len(returns),
    )

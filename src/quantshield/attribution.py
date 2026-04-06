"""Risk attribution helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_variance(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Compute portfolio variance."""
    w = weights.reindex(covariance.index).fillna(0.0).values
    return float(w.T @ covariance.values @ w)


def portfolio_volatility(weights: pd.Series, covariance: pd.DataFrame) -> float:
    """Compute portfolio volatility."""
    variance = portfolio_variance(weights, covariance)
    return float(np.sqrt(max(variance, 0.0)))


def marginal_contribution_to_risk(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Marginal contribution to total portfolio volatility."""
    vol = portfolio_volatility(weights, covariance)
    if vol <= 0.0:
        return pd.Series(0.0, index=covariance.index, name="marginal_risk")
    w = weights.reindex(covariance.index).fillna(0.0)
    marginal = covariance.dot(w) / vol
    marginal.name = "marginal_risk"
    return marginal


def component_contribution_to_risk(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Asset-level component contribution to portfolio volatility."""
    weights = weights.reindex(covariance.index).fillna(0.0)
    contribution = weights * marginal_contribution_to_risk(weights, covariance)
    contribution.name = "component_risk"
    return contribution


def percentage_contribution_to_risk(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Percentage contribution to portfolio volatility."""
    components = component_contribution_to_risk(weights, covariance)
    total = float(components.sum())
    if abs(total) <= 1e-12:
        return pd.Series(0.0, index=components.index, name="percentage_risk")
    pct = components / total
    pct.name = "percentage_risk"
    return pct


def risk_attribution_table(weights: pd.Series, covariance: pd.DataFrame) -> pd.DataFrame:
    """Combine key risk attribution statistics in a single table."""
    weights = weights.reindex(covariance.index).fillna(0.0)
    table = pd.concat(
        [
            weights.rename("weight"),
            marginal_contribution_to_risk(weights, covariance),
            component_contribution_to_risk(weights, covariance),
            percentage_contribution_to_risk(weights, covariance),
        ],
        axis=1,
    )
    table = table.sort_values("component_risk", ascending=False)
    table.index.name = "Ticker"
    return table

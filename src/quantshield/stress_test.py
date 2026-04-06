"""Scenario-based portfolio stress testing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantshield.attribution import component_contribution_to_risk, portfolio_volatility


@dataclass(slots=True)
class StressScenarioResult:
    """Output from a single stress scenario."""

    name: str
    stressed_return: float | None
    stressed_volatility: float | None
    contributions: pd.Series
    notes: str


def custom_shock(weights: pd.Series, shock_vector: pd.Series, name: str = "custom_shock") -> StressScenarioResult:
    """Apply a user-supplied return shock vector."""
    aligned_shocks = shock_vector.reindex(weights.index).fillna(0.0)
    contributions = weights * aligned_shocks
    return StressScenarioResult(
        name=name,
        stressed_return=float(contributions.sum()),
        stressed_volatility=None,
        contributions=contributions.sort_values(),
        notes="Custom return shock applied asset by asset.",
    )


def equity_market_shock(
    weights: pd.Series,
    asset_class_map: dict[str, str],
    *,
    shock: float = -0.20,
) -> StressScenarioResult:
    """Shock equity-like assets by a fixed percentage."""
    vector = pd.Series(
        {
            ticker: shock if asset_class_map.get(ticker) in {"equity", "real_estate"} else 0.0
            for ticker in weights.index
        }
    )
    return custom_shock(weights, vector, name="equity_market_shock")


def interest_rate_shock_proxy(
    weights: pd.Series,
    asset_class_map: dict[str, str],
    *,
    shock: float = -0.10,
) -> StressScenarioResult:
    """Proxy an interest-rate shock by stressing bond-like assets."""
    vector = pd.Series(
        {
            ticker: shock if asset_class_map.get(ticker) == "bond" else 0.0
            for ticker in weights.index
        }
    )
    return custom_shock(weights, vector, name="interest_rate_shock_proxy")


def single_asset_crash(weights: pd.Series, ticker: str, *, shock: float = -0.30) -> StressScenarioResult:
    """Stress a single asset."""
    vector = pd.Series(0.0, index=weights.index)
    if ticker not in vector.index:
        raise KeyError(f"Ticker '{ticker}' not found in portfolio weights.")
    vector.loc[ticker] = shock
    return custom_shock(weights, vector, name=f"single_asset_crash:{ticker}")


def correlation_spike_scenario(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    floor_correlation: float = 0.75,
    volatility_multiplier: float = 1.25,
) -> StressScenarioResult:
    """Increase cross-asset correlations and volatilities to simulate a risk-off regime."""
    aligned_cov = covariance.reindex(index=weights.index, columns=weights.index).fillna(0.0)
    vol = np.sqrt(np.diag(aligned_cov.values))
    base_corr = aligned_cov.values / np.outer(vol, vol)
    base_corr = np.nan_to_num(base_corr, nan=0.0, posinf=0.0, neginf=0.0)
    stressed_corr = np.where(np.eye(len(weights), dtype=bool), 1.0, np.maximum(base_corr, floor_correlation))
    stressed_vol = vol * volatility_multiplier
    stressed_cov = pd.DataFrame(
        stressed_corr * np.outer(stressed_vol, stressed_vol),
        index=weights.index,
        columns=weights.index,
    )
    contributions = component_contribution_to_risk(weights, stressed_cov)
    return StressScenarioResult(
        name="correlation_spike",
        stressed_return=None,
        stressed_volatility=portfolio_volatility(weights, stressed_cov),
        contributions=contributions.sort_values(ascending=False),
        notes="Returns are unchanged in this scenario; only portfolio volatility is stressed.",
    )


def run_default_stress_tests(
    weights: pd.Series,
    covariance: pd.DataFrame,
    asset_class_map: dict[str, str],
) -> dict[str, StressScenarioResult]:
    """Run the default stress scenario set."""
    return {
        "equity_market_shock": equity_market_shock(weights, asset_class_map),
        "interest_rate_shock_proxy": interest_rate_shock_proxy(weights, asset_class_map),
        "correlation_spike": correlation_spike_scenario(weights, covariance),
        "single_asset_crash": single_asset_crash(weights, weights.idxmax()),
    }


def stress_results_table(results: dict[str, StressScenarioResult]) -> pd.DataFrame:
    """Summarize stress scenario outputs in tabular form."""
    rows: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        ordered = (
            result.contributions.sort_values(ascending=False)
            if result.stressed_return is None
            else result.contributions.sort_values()
        )
        top_contributor = ordered.index[0] if not ordered.empty else None
        top_value = ordered.iloc[0] if not ordered.empty else None
        rows[name] = {
            "stressed_return": result.stressed_return,
            "stressed_volatility": result.stressed_volatility,
            "largest_loss_contributor": top_contributor,
            "largest_loss_contribution": top_value,
            "notes": result.notes,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "Scenario"
    return table

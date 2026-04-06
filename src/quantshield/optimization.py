"""Portfolio optimization routines with practical constraints."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quantshield.attribution import component_contribution_to_risk, portfolio_volatility
from quantshield.config import OptimizationConfig as SettingsOptimizationConfig
from quantshield.metrics import herfindahl_index, top_weight_share

OptimizationConfig = SettingsOptimizationConfig


@dataclass(slots=True)
class OptimizationResult:
    """Output from an optimization run."""

    weights: pd.Series
    success: bool
    objective_value: float
    message: str
    expected_return: float
    expected_volatility: float
    turnover: float
    concentration: float
    top_weight_share: float
    constraint_violations: dict[str, float]


def _resolve_weight_bounds(
    tickers: list[str],
    minimum: float | dict[str, float],
    maximum: float | dict[str, float],
    *,
    long_only: bool,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            float(minimum.get(ticker, 0.0)) if isinstance(minimum, dict) else float(minimum)
            for ticker in tickers
        ],
        dtype=float,
    )
    upper = np.array(
        [
            float(maximum.get(ticker, 1.0)) if isinstance(maximum, dict) else float(maximum)
            for ticker in tickers
        ],
        dtype=float,
    )
    if long_only:
        lower = np.maximum(lower, 0.0)
    if np.any(upper < lower):
        raise ValueError("Each max weight must be greater than or equal to the corresponding min weight.")
    if lower.sum() > 1.0 + 1e-12:
        raise ValueError("Minimum weight bounds are infeasible because they sum to more than 1.")
    if upper.sum() < 1.0 - 1e-12:
        raise ValueError("Maximum weight bounds are infeasible because they sum to less than 1.")
    return lower, upper


def bounded_equal_weight(
    tickers: list[str],
    minimum: float | dict[str, float],
    maximum: float | dict[str, float],
    *,
    long_only: bool = True,
) -> pd.Series:
    """Construct a feasible equal-weight-style portfolio inside box bounds."""
    lower, upper = _resolve_weight_bounds(tickers, minimum, maximum, long_only=long_only)
    weights = lower.copy()
    remaining = 1.0 - weights.sum()
    tolerance = 1e-12

    while remaining > tolerance:
        capacity = upper - weights
        open_mask = capacity > tolerance
        if not np.any(open_mask):
            break
        allocation = np.minimum(capacity[open_mask], remaining / open_mask.sum())
        weights[open_mask] += allocation
        remaining = 1.0 - weights.sum()

    weights = weights / weights.sum()
    return pd.Series(weights, index=tickers, name="weight")


def _turnover(weights: pd.Series, previous_weights: pd.Series | None) -> float:
    if previous_weights is None:
        return 0.0
    aligned_previous = previous_weights.reindex(weights.index).fillna(0.0)
    return float(0.5 * np.abs(weights - aligned_previous).sum())


def _check_constraints(
    weights: pd.Series,
    covariance: pd.DataFrame,
    config: OptimizationConfig,
    asset_class_map: dict[str, str] | None = None,
) -> dict[str, float]:
    violations: dict[str, float] = {}

    weight_sum_gap = abs(float(weights.sum()) - 1.0)
    if weight_sum_gap > 1e-6:
        violations["sum_to_one"] = weight_sum_gap

    lower, upper = _resolve_weight_bounds(
        list(weights.index),
        config.min_weight,
        config.max_weight,
        long_only=config.long_only,
    )
    lower_gap = float(np.maximum(lower - weights.values, 0.0).max()) if len(weights) else 0.0
    upper_gap = float(np.maximum(weights.values - upper, 0.0).max()) if len(weights) else 0.0
    if lower_gap > 1e-6:
        violations["min_weight"] = lower_gap
    if upper_gap > 1e-6:
        violations["max_weight"] = upper_gap

    if config.target_volatility is not None:
        realized_vol = portfolio_volatility(weights, covariance)
        if realized_vol > config.target_volatility + 1e-6:
            violations["target_volatility"] = realized_vol - config.target_volatility

    if asset_class_map and config.exposure_caps:
        asset_groups = pd.Series(asset_class_map).reindex(weights.index)
        grouped = weights.groupby(asset_groups).sum()
        for group, cap in config.exposure_caps.items():
            exposure = float(grouped.get(group, 0.0))
            if exposure > cap + 1e-6:
                violations[f"exposure_cap:{group}"] = exposure - cap
    return violations


def optimize_portfolio(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    config: OptimizationConfig,
    *,
    previous_weights: pd.Series | None = None,
    asset_class_map: dict[str, str] | None = None,
) -> OptimizationResult:
    """Optimize a portfolio subject to common practical constraints."""
    tickers = list(covariance.columns)
    mean_returns = mean_returns.reindex(tickers).fillna(0.0)
    covariance = covariance.reindex(index=tickers, columns=tickers).fillna(0.0)

    if config.objective == "equal_weight":
        weights = bounded_equal_weight(
            tickers,
            config.min_weight,
            config.max_weight,
            long_only=config.long_only,
        )
        return OptimizationResult(
            weights=weights,
            success=True,
            objective_value=0.0,
            message="Constructed bounded equal-weight benchmark.",
            expected_return=float(weights.dot(mean_returns)),
            expected_volatility=portfolio_volatility(weights, covariance),
            turnover=_turnover(weights, previous_weights),
            concentration=herfindahl_index(weights),
            top_weight_share=top_weight_share(weights),
            constraint_violations=_check_constraints(weights, covariance, config, asset_class_map),
        )

    lower, upper = _resolve_weight_bounds(
        tickers,
        config.min_weight,
        config.max_weight,
        long_only=config.long_only,
    )
    bounds = list(zip(lower, upper))

    starting_weights = bounded_equal_weight(
        tickers,
        config.min_weight,
        config.max_weight,
        long_only=config.long_only,
    )

    if previous_weights is not None:
        candidate = previous_weights.reindex(tickers).fillna(0.0)
        candidate = pd.Series(np.clip(candidate.values, lower, upper), index=tickers, name="weight")
        if abs(candidate.sum()) > 1e-12:
            candidate = candidate / candidate.sum()
            if ((candidate >= lower - 1e-9) & (candidate <= upper + 1e-9)).all():
                starting_weights = candidate.rename("weight")

    previous_vector = None if previous_weights is None else previous_weights.reindex(tickers).fillna(0.0).values

    def turnover_penalty(vector: np.ndarray) -> float:
        if previous_vector is None or config.turnover_penalty <= 0.0:
            return 0.0
        return config.turnover_penalty * float(np.square(vector - previous_vector).sum())

    def min_variance_objective(vector: np.ndarray) -> float:
        return float(vector.T @ covariance.values @ vector) + turnover_penalty(vector)

    def mean_variance_objective(vector: np.ndarray) -> float:
        variance = float(vector.T @ covariance.values @ vector)
        expected_return = float(vector @ mean_returns.values)
        utility = expected_return - 0.5 * config.risk_aversion * variance
        return -utility + turnover_penalty(vector)

    def risk_parity_objective(vector: np.ndarray) -> float:
        weights = pd.Series(vector, index=tickers)
        components = component_contribution_to_risk(weights, covariance)
        target = float(components.sum()) / len(components)
        return float(np.square(components.values - target).sum()) + turnover_penalty(vector)

    objectives = {
        "min_variance": min_variance_objective,
        "mean_variance": mean_variance_objective,
        "risk_parity": risk_parity_objective,
    }
    if config.objective not in objectives:
        raise ValueError(f"Unsupported optimization objective: {config.objective}")

    constraints: list[dict[str, object]] = [{"type": "eq", "fun": lambda vector: np.sum(vector) - 1.0}]

    if config.target_volatility is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda vector: config.target_volatility**2 - float(vector.T @ covariance.values @ vector),
            }
        )

    if asset_class_map and config.exposure_caps:
        for group, cap in config.exposure_caps.items():
            indices = [idx for idx, ticker in enumerate(tickers) if asset_class_map.get(ticker) == group]
            if not indices:
                continue
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda vector, idx=indices, group_cap=cap: group_cap - float(np.sum(vector[idx])),
                }
            )

    solution = minimize(
        objectives[config.objective],
        x0=starting_weights.values,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-10},
    )

    if solution.success:
        final_weights = pd.Series(solution.x, index=tickers, name="weight")
        message = solution.message
        success = True
        objective_value = float(solution.fun)
    elif config.fallback_to_equal_weight:
        final_weights = bounded_equal_weight(
            tickers,
            config.min_weight,
            config.max_weight,
            long_only=config.long_only,
        )
        message = f"Optimization failed ({solution.message}). Fell back to bounded equal weight."
        success = False
        objective_value = float("nan")
    else:
        raise RuntimeError(f"Optimization failed: {solution.message}")

    expected_return = float(final_weights.dot(mean_returns))
    expected_vol = portfolio_volatility(final_weights, covariance)
    violations = _check_constraints(final_weights, covariance, config, asset_class_map)

    return OptimizationResult(
        weights=final_weights,
        success=success,
        objective_value=objective_value,
        message=str(message),
        expected_return=expected_return,
        expected_volatility=expected_vol,
        turnover=_turnover(final_weights, previous_weights),
        concentration=herfindahl_index(final_weights),
        top_weight_share=top_weight_share(final_weights),
        constraint_violations=violations,
    )

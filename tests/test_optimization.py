from __future__ import annotations

import numpy as np
import pandas as pd

from quantshield.config import OptimizationConfig
from quantshield.optimization import optimize_portfolio


def sample_inputs() -> tuple[pd.Series, pd.DataFrame]:
    mean = pd.Series({"A": 0.10, "B": 0.08, "C": 0.06})
    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.05, 0.01],
            [0.00, 0.01, 0.03],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    return mean, covariance


def test_min_variance_weights_sum_to_one() -> None:
    mean, covariance = sample_inputs()
    config = OptimizationConfig(objective="min_variance", min_weight=0.0, max_weight=0.8)
    result = optimize_portfolio(mean, covariance, config)
    assert abs(result.weights.sum() - 1.0) < 1e-6
    assert (result.weights >= -1e-10).all()


def test_mean_variance_respects_bounds() -> None:
    mean, covariance = sample_inputs()
    config = OptimizationConfig(objective="mean_variance", min_weight=0.1, max_weight=0.6)
    result = optimize_portfolio(mean, covariance, config)
    assert np.all(result.weights.values >= 0.1 - 1e-6)
    assert np.all(result.weights.values <= 0.6 + 1e-6)


def test_risk_parity_is_long_only() -> None:
    mean, covariance = sample_inputs()
    config = OptimizationConfig(objective="risk_parity", min_weight=0.0, max_weight=0.8)
    result = optimize_portfolio(mean, covariance, config)
    assert (result.weights >= -1e-10).all()

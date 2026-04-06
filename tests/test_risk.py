from __future__ import annotations

import numpy as np
import pandas as pd

from quantshield.risk import (
    RiskConfig,
    compare_covariance_estimators,
    estimate_risk,
    exponentially_weighted_covariance,
)


def make_returns() -> pd.DataFrame:
    data = np.array(
        [
            [0.01, 0.02, -0.01],
            [0.00, 0.01, 0.01],
            [0.02, -0.01, 0.00],
            [-0.01, 0.00, 0.01],
            [0.01, 0.01, 0.02],
        ]
    )
    return pd.DataFrame(data, columns=["A", "B", "C"])


def test_estimate_risk_ledoit_wolf_shapes() -> None:
    returns = make_returns()
    estimate = estimate_risk(returns, RiskConfig(covariance_estimator="ledoit_wolf"))
    assert list(estimate.mean.index) == ["A", "B", "C"]
    assert estimate.covariance.shape == (3, 3)
    assert np.all(np.diag(estimate.covariance.values) > 0.0)


def test_ewma_covariance_is_symmetric() -> None:
    returns = make_returns()
    covariance = exponentially_weighted_covariance(returns, span=3)
    np.testing.assert_allclose(covariance.values, covariance.values.T, atol=1e-12)


def test_compare_covariance_estimators_returns_summary() -> None:
    returns = make_returns()
    estimators, summary = compare_covariance_estimators(returns, ewma_span=3)
    assert {"historical", "ledoit_wolf", "ewma"} == set(estimators)
    assert {"trace", "condition_number", "average_variance"} <= set(summary.columns)

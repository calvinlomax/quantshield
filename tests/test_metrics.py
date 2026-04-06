from __future__ import annotations

import pandas as pd

from quantshield.metrics import annualized_return, drawdown_series, herfindahl_index, max_drawdown


def test_annualized_return_positive_series() -> None:
    returns = pd.Series([0.01] * 252)
    assert annualized_return(returns) > 0.0


def test_drawdown_is_non_positive() -> None:
    returns = pd.Series([0.10, -0.05, 0.02, -0.10, 0.03])
    drawdown = drawdown_series(returns)
    assert (drawdown <= 1e-12).all()
    assert max_drawdown(returns) <= 0.0


def test_herfindahl_index() -> None:
    weights = pd.Series([0.5, 0.3, 0.2])
    assert abs(herfindahl_index(weights) - 0.38) < 1e-12

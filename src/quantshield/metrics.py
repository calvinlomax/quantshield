"""Performance and concentration metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_returns(returns: pd.Series | pd.DataFrame, start_value: float = 1.0) -> pd.Series | pd.DataFrame:
    """Convert period returns into a cumulative wealth index."""
    return start_value * (1.0 + returns).cumprod()


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Geometric annualized return."""
    if returns.empty:
        return np.nan
    total_growth = float((1.0 + returns).prod())
    years = len(returns) / periods_per_year
    if years <= 0.0 or total_growth <= 0.0:
        return np.nan
    return total_growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of returns."""
    if returns.empty:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio."""
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year=periods_per_year)
    if ann_vol == 0.0 or np.isnan(ann_vol):
        return np.nan
    return (ann_return - risk_free_rate) / ann_vol


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """Annualized Sortino ratio."""
    downside = returns[returns < 0.0]
    if downside.empty:
        return np.nan
    downside_vol = float(downside.std(ddof=1) * np.sqrt(periods_per_year))
    if downside_vol == 0.0 or np.isnan(downside_vol):
        return np.nan
    ann_return = annualized_return(returns, periods_per_year=periods_per_year)
    return (ann_return - risk_free_rate) / downside_vol


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdowns from a return stream."""
    wealth = cumulative_returns(returns)
    running_peak = wealth.cummax()
    return wealth / running_peak - 1.0


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown."""
    if returns.empty:
        return np.nan
    return float(drawdown_series(returns).min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar ratio."""
    drawdown = abs(max_drawdown(returns))
    if drawdown == 0.0 or np.isnan(drawdown):
        return np.nan
    return annualized_return(returns, periods_per_year=periods_per_year) / drawdown


def herfindahl_index(weights: pd.Series) -> float:
    """Portfolio concentration statistic."""
    return float(np.square(weights).sum())


def top_weight_share(weights: pd.Series) -> float:
    """Largest portfolio weight."""
    return float(weights.max())


def average_turnover(turnover: pd.Series) -> float:
    """Average turnover over rebalance dates."""
    if turnover.empty:
        return np.nan
    return float(turnover.mean())


def performance_summary(
    returns: pd.DataFrame,
    *,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Create a table of common performance statistics."""
    rows: dict[str, dict[str, float]] = {}
    for column in returns.columns:
        series = returns[column].dropna()
        rows[column] = {
            "annualized_return": annualized_return(series, periods_per_year=periods_per_year),
            "annualized_volatility": annualized_volatility(series, periods_per_year=periods_per_year),
            "sharpe_ratio": sharpe_ratio(
                series,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
            "sortino_ratio": sortino_ratio(
                series,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),
            "max_drawdown": max_drawdown(series),
            "calmar_ratio": calmar_ratio(series, periods_per_year=periods_per_year),
        }
    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "Portfolio"
    return summary

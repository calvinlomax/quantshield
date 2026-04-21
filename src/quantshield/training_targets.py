"""Shared helpers for building training targets from explicit ticker universes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantshield.config import OptimizationConfig
from quantshield.optimization import optimize_portfolio
from quantshield.universe import CANONICAL_TOP_ETF_ASSET_CLASS_MAP
from quantshield.utils import generate_schedule


def infer_asset_class_map(tickers: list[str]) -> dict[str, str]:
    """Infer a coarse asset-class map for explicit ticker universes."""
    commodity = {"GLD", "SLV", "DBC", "USO"}
    bond = {"TLT", "IEF", "SHY", "LQD", "AGG", "BND"}
    real_estate = {"VNQ", "IYR", "SCHH"}
    asset_class_map: dict[str, str] = {}
    for ticker in tickers:
        if ticker in CANONICAL_TOP_ETF_ASSET_CLASS_MAP:
            asset_class_map[ticker] = CANONICAL_TOP_ETF_ASSET_CLASS_MAP[ticker]
        elif ticker in commodity:
            asset_class_map[ticker] = "commodity"
        elif ticker in bond:
            asset_class_map[ticker] = "bond"
        elif ticker in real_estate:
            asset_class_map[ticker] = "real_estate"
        else:
            asset_class_map[ticker] = "equity"
    return asset_class_map


def build_forward_weight_histories(
    returns: pd.DataFrame,
    *,
    tickers: list[str],
    lookback_window: int,
    rebalance_frequency: str,
    asset_class_map: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Build forward-looking objective targets for a chosen ticker universe."""
    rebalance_dates = list(generate_schedule(returns.index, rebalance_frequency))
    histories: dict[str, list[pd.Series]] = {
        "best_asset": [],
        "best_asset_anchor": [],
        "best_asset_mirror": [],
        "top2_blend": [],
        "oracle_softmax": [],
        "forward_mean_variance": [],
        "forward_risk_parity": [],
        "forward_min_variance": [],
    }
    index: list[pd.Timestamp] = []

    for position, rebalance_date in enumerate(rebalance_dates):
        window = returns.loc[:rebalance_date, tickers].iloc[-lookback_window:]
        if len(window) < lookback_window:
            continue
        start_idx = returns.index.get_loc(rebalance_date) + 1
        end_idx = (
            returns.index.get_loc(rebalance_dates[position + 1])
            if position < len(rebalance_dates) - 1
            else len(returns.index) - 1
        )
        forward_segment = returns.iloc[start_idx : end_idx + 1][tickers]
        if forward_segment.empty:
            continue

        cumulative_returns = (1.0 + forward_segment).prod() - 1.0
        best_ticker = str(cumulative_returns.idxmax())
        best_asset_weights = pd.Series(0.0, index=tickers)
        best_asset_weights.loc[best_ticker] = 1.0

        top2 = cumulative_returns.sort_values(ascending=False).head(2).clip(lower=0.0)
        if float(top2.sum()) <= 0.0:
            top2_blend_weights = pd.Series(1.0 / len(tickers), index=tickers)
        else:
            top2_blend_weights = pd.Series(0.0, index=tickers)
            top2_blend_weights.loc[top2.index] = top2 / top2.sum()

        logits = cumulative_returns.to_numpy(dtype=float)
        logits = logits - float(logits.max())
        softmax = pd.Series(np.exp(6.0 * logits), index=tickers)
        oracle_softmax_weights = softmax / float(softmax.sum())

        annualized_mean = forward_segment.mean() * 252
        annualized_covariance = forward_segment.cov() * 252

        mean_variance = optimize_portfolio(
            annualized_mean,
            annualized_covariance,
            OptimizationConfig(
                objective="mean_variance",
                risk_aversion=0.35,
                long_only=True,
                min_weight=0.0,
                max_weight=1.0,
                turnover_penalty=0.0,
            ),
            asset_class_map=asset_class_map,
        ).weights
        risk_parity = optimize_portfolio(
            annualized_mean,
            annualized_covariance,
            OptimizationConfig(
                objective="risk_parity",
                long_only=True,
                min_weight=0.0,
                max_weight=1.0,
                turnover_penalty=0.0,
            ),
            asset_class_map=asset_class_map,
        ).weights
        min_variance = optimize_portfolio(
            annualized_mean,
            annualized_covariance,
            OptimizationConfig(
                objective="min_variance",
                long_only=True,
                min_weight=0.0,
                max_weight=1.0,
                turnover_penalty=0.0,
            ),
            asset_class_map=asset_class_map,
        ).weights

        histories["best_asset"].append(best_asset_weights)
        histories["best_asset_anchor"].append(best_asset_weights)
        histories["best_asset_mirror"].append(best_asset_weights)
        histories["top2_blend"].append(top2_blend_weights)
        histories["oracle_softmax"].append(oracle_softmax_weights)
        histories["forward_mean_variance"].append(mean_variance)
        histories["forward_risk_parity"].append(risk_parity)
        histories["forward_min_variance"].append(min_variance)
        index.append(pd.Timestamp(rebalance_date))

    return {
        objective: pd.DataFrame(weights, index=index).reindex(columns=tickers).fillna(0.0)
        for objective, weights in histories.items()
    }

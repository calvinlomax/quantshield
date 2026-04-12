"""Composite scoring for trained QuantShield policy models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _safe_value(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    return numeric if math.isfinite(numeric) else float(default)


def _safe_flag(row: pd.Series, key: str) -> float:
    value = row.get(key, False)
    return float(bool(value))


def _bounded_signal(value: float, *, scale: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(np.tanh(value / max(scale, 1e-9)))


def build_model_score_summary(
    benchmark_summary: pd.DataFrame,
    evaluation_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Create a composite score table from saved model evaluation artifacts."""
    rows: dict[str, dict[str, float]] = {}
    for split_name in benchmark_summary.index:
        if split_name not in evaluation_summary.index:
            continue
        benchmark_row = benchmark_summary.loc[split_name]
        evaluation_row = evaluation_summary.loc[split_name]

        benchmark_component = 30.0 * _bounded_signal(
            _safe_value(benchmark_row, "policy_mean_excess_return"),
            scale=0.010,
        )
        equal_weight_component = 20.0 * _bounded_signal(
            _safe_value(benchmark_row, "policy_mean_excess_vs_equal_weight"),
            scale=0.010,
        )
        restricted_random_component = 15.0 * _bounded_signal(
            _safe_value(benchmark_row, "policy_mean_excess_vs_restricted_random"),
            scale=0.010,
        )
        markowitz_component = 25.0 * _bounded_signal(
            _safe_value(benchmark_row, "policy_mean_excess_vs_markowitz"),
            scale=0.010,
        )
        raw_return_component = 10.0 * _bounded_signal(
            _safe_value(evaluation_row, "policy_mean_raw_return"),
            scale=0.010,
        )
        t_statistics = [
            _safe_value(benchmark_row, "t_statistic"),
            _safe_value(benchmark_row, "equal_weight_t_statistic"),
            _safe_value(benchmark_row, "restricted_random_t_statistic"),
            _safe_value(benchmark_row, "markowitz_t_statistic"),
        ]
        average_t_statistic = float(np.mean(t_statistics))
        significance_component = 15.0 * _bounded_signal(average_t_statistic, scale=3.0)
        significance_bonus = 10.0 * float(
            np.mean(
                [
                    _safe_flag(benchmark_row, "significant_outperformance"),
                    _safe_flag(benchmark_row, "equal_weight_significant_outperformance"),
                    _safe_flag(benchmark_row, "restricted_random_significant_outperformance"),
                    _safe_flag(benchmark_row, "markowitz_significant_outperformance"),
                ]
            )
        )
        weight_error_penalty = 15.0 * min(max(_safe_value(evaluation_row, "mean_abs_weight_error") / 0.25, 0.0), 1.0)
        composite_score = (
            benchmark_component
            + equal_weight_component
            + restricted_random_component
            + markowitz_component
            + raw_return_component
            + significance_component
            + significance_bonus
            - weight_error_penalty
        )
        rows[str(split_name)] = {
            "benchmark_component": benchmark_component,
            "equal_weight_component": equal_weight_component,
            "restricted_random_component": restricted_random_component,
            "markowitz_component": markowitz_component,
            "raw_return_component": raw_return_component,
            "significance_component": significance_component,
            "significance_bonus": significance_bonus,
            "weight_error_penalty": weight_error_penalty,
            "composite_score": composite_score,
        }

    summary = pd.DataFrame.from_dict(rows, orient="index")
    summary.index.name = "Split"
    return summary

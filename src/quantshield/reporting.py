"""Reporting helpers for saved artifacts and console summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantshield.attribution import risk_attribution_table
from quantshield.metrics import average_turnover, herfindahl_index, top_weight_share
from quantshield.stress_test import StressScenarioResult, stress_results_table
from quantshield.utils import ensure_directory, format_percent


def build_summary_text(
    *,
    tickers: list[str],
    sample_start: str,
    sample_end: str,
    lookback_days: int,
    rebalance_frequency: str,
    covariance_estimator: str,
    objective: str,
    final_weights: pd.Series,
    performance_summary: pd.DataFrame,
    turnover: pd.Series,
    risk_attribution: pd.DataFrame,
    stress_summary: pd.DataFrame,
    latest_constraint_status: str,
) -> str:
    """Create a plain-text summary report."""
    lines = [
        "QuantShield Summary Report",
        "=========================",
        "",
        f"Universe: {', '.join(tickers)}",
        f"Sample window: {sample_start} to {sample_end}",
        f"Lookback window: {lookback_days} trading days",
        f"Rebalance frequency: {rebalance_frequency}",
        f"Covariance estimator: {covariance_estimator}",
        f"Optimization objective: {objective}",
        f"Latest constraint check: {latest_constraint_status or 'no violations reported'}",
        "",
        "Final weights:",
    ]
    for ticker, weight in final_weights.sort_values(ascending=False).items():
        lines.append(f"  {ticker}: {format_percent(weight)}")

    lines.extend(
        [
            "",
            "Performance summary:",
            performance_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
            "",
            f"Average turnover: {average_turnover(turnover):0.4f}",
            f"Herfindahl index: {herfindahl_index(final_weights):0.4f}",
            f"Top weight share: {top_weight_share(final_weights):0.4f}",
            "",
            "Risk attribution summary:",
            risk_attribution[["weight", "component_risk", "percentage_risk"]]
            .head(5)
            .to_string(float_format=lambda value: f"{value:0.4f}"),
            "",
            "Stress test summary:",
            stress_summary.to_string(float_format=lambda value: f"{value:0.4f}"),
        ]
    )
    return "\n".join(lines)


def write_summary_text(summary: str, path: str | Path) -> Path:
    """Write a summary report to disk."""
    destination = Path(path)
    ensure_directory(destination.parent)
    destination.write_text(summary, encoding="utf-8")
    return destination


def build_risk_and_stress_reports(
    weights: pd.Series,
    covariance: pd.DataFrame,
    stress_results: dict[str, StressScenarioResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reusable risk attribution and stress summary tables."""
    return risk_attribution_table(weights, covariance), stress_results_table(stress_results)

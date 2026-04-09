"""Configuration loading and strongly typed application settings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class DataConfig:
    """Configuration for local market data retrieval."""

    tickers: list[str]
    asset_class_map: dict[str, str] = field(default_factory=dict)
    start_date: str = "2015-01-01"
    end_date: str | None = None
    cache_dir: str = "data/raw"
    use_cache: bool = True
    force_refresh: bool = False


@dataclass(slots=True)
class PreprocessingConfig:
    """Configuration for return series preparation."""

    return_type: str = "simple"
    annualization_factor: int = 252
    drop_all_nan_assets: bool = True
    forward_fill_prices: bool = True


@dataclass(slots=True)
class RiskConfig:
    """Configuration for risk estimators."""

    mean_estimator: str = "historical"
    covariance_estimator: str = "ledoit_wolf"
    ewma_span: int = 60
    annualize: bool = True


@dataclass(slots=True)
class OptimizationConfig:
    """Configuration for constrained portfolio optimization."""

    objective: str = "min_variance"
    risk_aversion: float = 3.0
    long_only: bool = True
    min_weight: float | dict[str, float] = 0.0
    max_weight: float | dict[str, float] = 1.0
    turnover_penalty: float = 0.0
    target_volatility: float | None = None
    fallback_to_equal_weight: bool = True
    exposure_caps: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class BacktestConfig:
    """Configuration for rolling backtests."""

    lookback_days: int = 252
    expanding_window: bool = False
    min_history_days: int = 126
    rebalance_frequency: str = "ME"
    benchmark_ticker: str = "SPY"


@dataclass(slots=True)
class ReportingConfig:
    """Configuration for output artifacts."""

    output_dir: str = "outputs"
    figures_dir: str = "outputs/figures"
    tables_dir: str = "outputs/tables"
    rolling_vol_window: int = 63


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    data: DataConfig
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise TypeError(f"Expected configuration section '{key}' to be a mapping.")
    return value


def from_dict(config: dict[str, Any]) -> AppConfig:
    """Build an AppConfig from a raw dictionary."""
    return AppConfig(
        data=DataConfig(**_section(config, "data")),
        preprocessing=PreprocessingConfig(**_section(config, "preprocessing")),
        risk=RiskConfig(**_section(config, "risk")),
        optimization=OptimizationConfig(**_section(config, "optimization")),
        backtest=BacktestConfig(**_section(config, "backtest")),
        reporting=ReportingConfig(**_section(config, "reporting")),
    )


def load_config(path: str | Path) -> AppConfig:
    """Load a YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}
    if not isinstance(raw_config, dict):
        raise TypeError("Top-level YAML config must be a mapping.")
    return from_dict(raw_config)

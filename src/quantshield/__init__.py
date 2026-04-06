"""QuantShield package for ML-first portfolio policy training and benchmark generation."""

from quantshield.backtest import BacktestConfig, BacktestResult, run_rolling_backtest
from quantshield.config import AppConfig, load_config
from quantshield.data_loader import MarketDataLoader
from quantshield.optimization import OptimizationConfig, OptimizationResult, optimize_portfolio
from quantshield.pipeline import prepare_market_data, run_pipeline, save_pipeline_artifacts
from quantshield.risk import RiskConfig, RiskEstimate, estimate_risk
from quantshield.tuned_suite import TUNED_PRESETS, TunedSuiteResult, run_tuned_objective_suite

__all__ = [
    "AppConfig",
    "BacktestConfig",
    "BacktestResult",
    "MarketDataLoader",
    "OptimizationConfig",
    "OptimizationResult",
    "RiskConfig",
    "RiskEstimate",
    "TUNED_PRESETS",
    "estimate_risk",
    "load_config",
    "optimize_portfolio",
    "prepare_market_data",
    "run_rolling_backtest",
    "run_pipeline",
    "run_tuned_objective_suite",
    "save_pipeline_artifacts",
    "TunedSuiteResult",
]

__version__ = "0.1.0"

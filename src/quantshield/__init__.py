"""QuantShield package."""

from quantshield.backtest import BacktestConfig, BacktestResult, run_rolling_backtest
from quantshield.config import AppConfig, load_config
from quantshield.data_loader import MarketDataLoader
from quantshield.optimization import OptimizationConfig, OptimizationResult, optimize_portfolio
from quantshield.risk import RiskConfig, RiskEstimate, estimate_risk

__all__ = [
    "AppConfig",
    "BacktestConfig",
    "BacktestResult",
    "MarketDataLoader",
    "OptimizationConfig",
    "OptimizationResult",
    "RiskConfig",
    "RiskEstimate",
    "estimate_risk",
    "load_config",
    "optimize_portfolio",
    "run_rolling_backtest",
]

__version__ = "0.1.0"

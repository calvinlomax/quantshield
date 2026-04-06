# QuantShield

QuantShield is a local Python project for constraint-aware portfolio construction, risk estimation, rolling backtesting, stress testing, and reporting. It is designed around a practical workflow:

1. Download market data locally from `yfinance`
2. Cache raw prices for reproducibility
3. Clean and align the price panel
4. Estimate returns and covariance
5. Solve constrained allocation problems
6. Run an out-of-sample rolling backtest
7. Stress test and decompose risk
8. Save tables and matplotlib figures

The base config universe is a diversified ETF basket:
`SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, VNQ`

The canonical objective suite in this repo is the tuned suite, which uses:
`SPY, QQQ, GLD`

## Why It Is Useful

QuantShield provides a stable baseline for portfolio research without relying on pre-downloaded CSVs or cloud-only data workflows. It emphasizes:

- local reproducibility through raw-data caching
- modular risk estimators and optimizers
- realistic portfolio constraints
- explicit walk-forward backtesting to avoid lookahead bias
- transparent outputs in CSV, TXT, and PNG form

## Project Architecture

```text
QuantShield/
  README.md
  requirements.txt
  requirements-rl.txt
  pyproject.toml
  config/
    default_config.yaml
  data/
    raw/
    processed/
  outputs/
    figures/
    tables/
  src/
    quantshield/
      __init__.py
      attribution.py
      backtest.py
      config.py
      data_loader.py
      metrics.py
      optimization.py
      pipeline.py
      plotting.py
      preprocessing.py
      rl.py
      reporting.py
      risk.py
      stress_test.py
      utils.py
  scripts/
    fetch_data.py
    run_backtest.py
    run_pipeline.py
    run_tuned_suite.py
    train_rl_policy.py
  notebooks/
    exploratory_analysis.ipynb
    portfolio_demo.ipynb
  tests/
    conftest.py
    test_data_loader.py
    test_metrics.py
    test_optimization.py
    test_rl.py
    test_risk.py
```

## Methodology

### Data

- Primary source: `yfinance`
- Field used: adjusted close, with a documented fallback to close if needed
- Raw downloads are cached under `data/raw/`
- Cleaned aligned prices and daily returns are saved under `data/processed/`

### Return Preparation

- Supports simple returns and log returns
- Sorts and de-duplicates dates
- Forward-fills interior missing price gaps
- Drops non-overlapping dates after alignment
- Annualization defaults to `252` trading days

### Risk Estimation

Implemented estimators:

- historical mean return
- historical covariance
- Ledoit-Wolf shrinkage covariance
- exponentially weighted covariance

Recommended default:

- covariance estimator: `ledoit_wolf`

### Portfolio Optimization

Implemented allocation methods:

- equal weight benchmark
- minimum variance
- mean-variance
- risk parity style approximation

Supported constraints:

- weights sum to 1
- long-only by default
- configurable min and max asset weights
- optional turnover penalty relative to previous weights
- optional target volatility cap
- optional exposure caps using asset-class metadata

The optimizer uses `scipy.optimize.minimize` with SLSQP so objectives and constraints remain easy to extend.

### Rolling Backtest

The backtester:

- uses only data available up to each rebalance date
- supports rolling and expanding training windows
- rebalances monthly by default
- compares the optimized portfolio to equal weight and a benchmark like `SPY`
- records rebalance logs, weight history, and turnover

### Risk Attribution

Implemented:

- marginal contribution to risk
- component contribution to risk
- percentage contribution to risk

### Stress Testing

Implemented scenarios:

- equity market shock
- interest-rate shock proxy on bond-like assets
- correlation spike regime
- single-asset crash
- user-defined custom shock vector

### Transformer Actor-Critic Policy

An optional RL extension is included for policy learning from the saved tuned-suite weights.

- offline dataset source: saved `weights_history.csv` files from the tuned suite
- state representation: trailing cross-asset return window with raw returns, z-scored returns, and cumulative-return features
- architecture: transformer-style cross-asset attention encoder over asset tokens
- actor head: continuous-action Dirichlet policy on the portfolio simplex
- critic head: action-conditioned value head for offline actor-critic training
- reward target: next-period excess return versus `SPY`
- usage: experimental policy-learning layer on top of the classical optimizer suite

## Installation

Python 3.11+ is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional RL dependencies:

```bash
pip install -r requirements-rl.txt
```

Optional editable install:

```bash
pip install -e .
```

Editable install with RL extras:

```bash
pip install -e .[rl]
```

## How To Fetch Data Locally Using yfinance

This project expects market data to be downloaded locally when you run it.

```bash
python scripts/fetch_data.py
```

Force a fresh download instead of using cache:

```bash
python scripts/fetch_data.py --force-refresh
```

Override the universe and date range:

```bash
python scripts/fetch_data.py \
  --tickers SPY QQQ TLT GLD \
  --start-date 2018-01-01 \
  --end-date 2024-12-31 \
  --force-refresh
```

## How To Run The Backtest

Run the backtest only:

```bash
python scripts/run_backtest.py
```

Run the full pipeline:

```bash
python scripts/run_pipeline.py
```

Run the canonical objective suite. It uses the tuned `SPY,QQQ,GLD` subset and objective-specific overrides:

```bash
python scripts/run_tuned_suite.py
```

Train the transformer actor-critic policy from the saved tuned-suite weights:

```bash
python scripts/train_rl_policy.py
```

Override the optimizer objective or covariance estimator from the CLI:

```bash
python scripts/run_pipeline.py \
  --objective mean_variance \
  --covariance-estimator ewma
```

## Configuration

The default settings live in `config/default_config.yaml`.

Key sections:

- `data`: ticker universe, asset-class map, cache behavior, date range
- `preprocessing`: return type and annualization factor
- `risk`: covariance estimator choice
- `optimization`: objective, bounds, turnover penalty, target volatility, exposure caps
- `backtest`: lookback window, expanding vs rolling, rebalance frequency, benchmark
- `reporting`: output locations and rolling volatility window

Example changes:

- replace the default ticker basket
- tighten max weight bounds
- add exposure caps such as `equity: 0.70`
- switch from `min_variance` to `risk_parity`
- switch from `ledoit_wolf` to `ewma`

Tuned suite presets:

- universe: `SPY, QQQ, GLD`
- `mean_variance`: historical covariance, `lookback_days=252`, `risk_aversion=0.1`, `max_weight=1.0`
- `risk_parity`: Ledoit-Wolf covariance, `lookback_days=252`, `max_weight=0.70`
- `min_variance`: Ledoit-Wolf covariance, `lookback_days=252`, `max_weight=0.70`
- `equal_weight`: `max_weight=1.0`

## Outputs

Generated tables include:

- cleaned prices
- daily returns
- performance summary
- comparison return streams
- rebalance log
- turnover
- weight history
- final weights
- risk attribution
- stress summary
- covariance estimator comparison
- text summary report

Generated figures include:

- price history
- return correlation heatmap
- cumulative return curves
- rolling volatility
- drawdown chart
- portfolio weights over time
- risk contribution chart
- approximate efficient frontier

The canonical suite writes one report bundle per objective under `outputs/tuned_objective_runs/` plus a top-level `tuned_objective_comparison.csv`.

The RL policy trainer writes its artifacts under `outputs/rl_policy/`, including:

- `training_history.csv`
- `evaluation_summary.csv`
- `policy_predictions.csv`
- `latest_policy_weights.csv`
- `actor_critic_policy.pt`
- `rl_config.json`

## Example Console Output Shape

```text
QuantShield Summary Report
=========================

Universe: SPY, QQQ, GLD
Lookback window: 252 trading days
Rebalance frequency: M
Covariance estimator: historical
Optimization objective: mean_variance

Final weights:
  GLD: 100.00%
  SPY: 0.00%
  QQQ: 0.00%

Performance summary:
             annualized_return  annualized_volatility  sharpe_ratio  ...
portfolio               0.1772                 0.2052        0.8637  ...
equal_weight            0.1584                 0.1452        1.0908  ...
benchmark               0.1320                 0.1797        0.7343  ...
```

## Testing

Run tests after installing dependencies:

```bash
python -m pytest
```

## Notes And Assumptions

- The project avoids lookahead bias by training only on data known at each rebalance date.
- Ledoit-Wolf is used as the recommended default because it is usually more stable than the raw sample covariance in small or noisy samples.
- Exposure caps are optional and only applied when asset metadata is available.
- The correlation spike scenario is a volatility stress rather than a direct directional return shock.
- The canonical objective suite is backtest-tuned on the same historical sample, so its outperformance versus SPY is exploratory rather than out-of-sample evidence.
- The transformer actor-critic module is also trained on historical tuned-suite outputs, so it should be treated as an experimental offline RL component rather than a production signal.

## Future Extensions

The codebase is structured so the following can be added cleanly:

- transaction cost model
- Black-Litterman overlay
- factor model decomposition
- regime detection
- bootstrap confidence intervals
- richer CLI options

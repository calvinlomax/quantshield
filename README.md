# QuantShield

QuantShield is a local machine learning portfolio research project centered on a transformer-based actor-critic policy. The classical portfolio engine is still in the repo, but it now serves a supporting role: it downloads market data from `yfinance`, generates constrained benchmark portfolios, and produces the demonstration weights used to train the policy.

The primary workflow is:

1. Download and cache market data locally from `yfinance`
2. Run a dense tuned benchmark suite on `SPY, QQQ, GLD`
3. Convert saved weight histories into an offline training dataset
4. Train a cross-asset attention actor-critic with a continuous-action simplex head
5. Save the policy checkpoint, predictions, evaluation tables, and benchmark comparisons

The default classical backtest config is tuned to beat the `SPY` benchmark on the saved historical sample:
`SPY, QQQ, GLD` with `mean_variance`, historical covariance, `lookback=252`, `risk_aversion=0.1`

The broader ETF basket is still available for classical experiments in `config/broad_universe_config.yaml`:
`SPY, QQQ, IWM, EFA, EEM, TLT, LQD, GLD, VNQ`

## Why QuantShield

QuantShield is useful if you want a reproducible local workflow for portfolio ML without depending on pre-downloaded datasets or cloud-only research stacks.

- All training data is fetched locally with Python from `yfinance`
- Raw prices are cached for reproducibility
- The ML model is benchmarked against explicit constrained allocation rules
- The policy is trained on saved weight histories instead of opaque manual labels
- Outputs are written as CSV, TXT, PNG, and `.pt` artifacts for inspection

## What Is The Central Model

The central model is a transformer-style actor-critic policy implemented in [`src/quantshield/rl.py`](src/quantshield/rl.py).

- Input state: trailing cross-asset return window
- State features per asset: raw returns, volatility-normalized returns, cumulative returns
- Encoder: cross-asset attention block over asset tokens
- Actor head: Dirichlet continuous-action head that outputs portfolio weights on the simplex
- Critic head: action-conditioned value head
- Training target: next-period excess return versus `SPY`
- Default promoted architecture: `hidden_dim=192`, `attention_heads=6`, `attention_layers=4`, `batch_size=128`, `epochs=120`

This is an offline policy-learning setup. The classical optimizer suite is the demonstration generator and benchmark layer, not the main product.

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
    tuned_objective_runs/
    rl_policy/
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
      reporting.py
      risk.py
      rl.py
      stress_test.py
      tuned_suite.py
      utils.py
  scripts/
    fetch_data.py
    run_backtest.py
    run_ml_pipeline.py
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

## ML-First Methodology

### 1. Local Data Collection

- Primary source: `yfinance`
- Prices are fetched locally in Python when the pipeline runs
- Raw downloads are cached under `data/raw/`
- Cleaned aligned prices and daily returns are saved under `data/processed/`
- No pre-downloaded CSVs are used as the primary source

### 2. Benchmark Demonstration Generation

The tuned benchmark suite is defined in [`src/quantshield/tuned_suite.py`](src/quantshield/tuned_suite.py).

- Universe: `SPY, QQQ, GLD`
- Objectives: `min_variance`, `mean_variance`, `risk_parity`, `equal_weight`
- Rebalance style: rolling walk-forward backtest with weekly (`W-FRI`) rebalances in the ML pipeline
- Constraint set: long-only, weights sum to one, max-weight controls, optional turnover penalty
- Risk estimators: historical covariance and Ledoit-Wolf shrinkage

These benchmark portfolios generate the saved `weights_history.csv` files used by the ML policy trainer.

### 3. Offline Policy Dataset

The training dataset is built from:

- saved tuned-suite weight histories
- locally prepared return windows
- realized forward returns between rebalance dates

Each sample contains:

- a lookback window of cross-asset features
- the demonstrated portfolio weights
- realized raw return
- realized excess return versus `SPY`

### 4. Transformer Actor-Critic Training

The policy trainer:

- encodes each asset as a token
- uses cross-asset attention to model interactions between assets
- predicts continuous portfolio weights through a Dirichlet policy head
- learns a critic on action-conditioned rewards
- tracks training and validation performance against the benchmark demonstrations
- saves a benchmark significance report so policy excess return versus `SPY` is explicit rather than implied
- uses a much denser demonstration set than the monthly baseline, increasing the offline sample count from roughly `544` monthly samples to roughly `2,352` weekly samples on the cached 2015-2026 window

### 5. Evaluation

The saved evaluation artifacts include:

- policy checkpoint
- training history
- policy-vs-demo predictions
- latest policy weights
- evaluation summary on train, validation, and full samples
- ML pipeline summary text

## Installation

Python `3.11+` is required.

Base setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the ML dependency layer:

```bash
pip install -r requirements-rl.txt
```

Optional editable install:

```bash
pip install -e .[rl]
```

## Run The Main ML Workflow

This is the canonical command for the repo:

```bash
python scripts/run_ml_pipeline.py
```

It will:

1. fetch or reuse cached market data from `yfinance`
2. run the tuned benchmark suite
3. build the offline policy dataset
4. train the transformer actor-critic model
5. save benchmark artifacts under `outputs/ml_tuned_objective_runs/`
6. save ML artifacts under `outputs/rl_policy/`
7. save RL figures under `outputs/rl_policy/figures/`

The default ML run now uses the promoted larger model:

- `suite_rebalance_frequency=W-FRI`
- `hidden_dim=192`
- `attention_heads=6`
- `attention_layers=4`
- `batch_size=128`
- `epochs=120`

Useful overrides:

```bash
python scripts/run_ml_pipeline.py \
  --epochs 60 \
  --batch-size 64 \
  --lookback-window 84
```

Reuse an existing tuned suite without regenerating it:

```bash
python scripts/run_ml_pipeline.py --skip-suite
```

## Lower-Level Scripts

If you want to run individual layers instead of the main ML workflow:

Fetch and cache data only:

```bash
python scripts/fetch_data.py
```

Generate the tuned benchmark suite only:

```bash
python scripts/run_tuned_suite.py
```

Train the policy from existing benchmark outputs only:

```bash
python scripts/train_rl_policy.py
```

Run the broader classical benchmark pipeline only:

```bash
python scripts/run_pipeline.py
```

Run the broader classical backtest only:

```bash
python scripts/run_backtest.py
```

## Configuration

Default settings live in `config/default_config.yaml`.

Important sections:

- `data`: ticker universe, asset-class map, cache behavior, date range
- `preprocessing`: return type and annualization factor
- `risk`: covariance estimator choice
- `optimization`: objective, bounds, turnover penalty, target volatility, exposure caps
- `backtest`: lookback window, expanding vs rolling, rebalance frequency, benchmark
- `reporting`: output locations and rolling volatility window

The main ML workflow uses the tuned benchmark presets in [`src/quantshield/tuned_suite.py`](src/quantshield/tuned_suite.py). The base YAML config still controls the underlying data fetch, benchmark ticker, and shared preprocessing settings.

For the old broader 9-asset classical setup, use:

```bash
python scripts/run_pipeline.py --config config/broad_universe_config.yaml
```

## Outputs

Primary ML artifacts are written to `outputs/rl_policy/`:

- `actor_critic_policy.pt`
- `rl_config.json`
- `training_history.csv`
- `benchmark_summary.csv`
- `evaluation_summary.csv`
- `policy_predictions.csv`
- `latest_policy_weights.csv`
- `ml_pipeline_summary.txt`

During model tuning, a size sweep can also be saved as:

- `model_size_sweep.csv`

RL figures are written to `outputs/rl_policy/figures/`:

- `training_diagnostics.png`
- `benchmark_comparison.png`
- `policy_cumulative_returns.png`
- `latest_policy_weights.png`

Supporting benchmark artifacts are written to `outputs/ml_tuned_objective_runs/` by default:

- one report bundle per objective
- `tuned_objective_comparison.csv`
- per-objective `weights_history.csv`
- per-objective `summary_report.txt`

The repo also writes cleaned prices and returns under `data/processed/`.

## Example Console Output Shape

```text
QuantShield ML pipeline complete.

Tuned benchmark suite:
               tickers  annualized_return  benchmark_return  excess_return_vs_spy  ...
Objective
mean_variance  SPY,QQQ,GLD             0.1620            0.1327                0.0292  ...
equal_weight   SPY,QQQ,GLD             0.1572            0.1327                0.0245  ...
risk_parity    SPY,QQQ,GLD             0.1497            0.1327                0.0170  ...
min_variance   SPY,QQQ,GLD             0.1347            0.1327                0.0020  ...

Policy benchmark comparison:
            samples  benchmark_mean_raw_return  policy_mean_raw_return  policy_mean_excess_return  t_statistic  p_value  significant_outperformance
Split
train      1795.0000                     0.0026                  0.0028                     0.0002       4.5137   0.0000                        True
validation  449.0000                     0.0029                  0.0029                     0.0000       3.8552   0.0001                        True
all        2244.0000                     0.0027                  0.0028                     0.0002       4.5329   0.0000                        True

Policy evaluation summary:
            samples  demo_mean_excess_return  policy_mean_excess_return  ...
Split
train      1795.0000                  -0.0003                      0.0002  ...
validation  449.0000                   0.0024                      0.0000  ...
all        2244.0000                   0.0002                      0.0002  ...
```

## Testing

Run tests after installing dependencies:

```bash
python -m pytest
```

## Notes And Assumptions

- The classical benchmark suite is backtest-tuned on the same historical sample, so its outperformance versus `SPY` is exploratory rather than out-of-sample evidence.
- The transformer actor-critic is trained offline from those benchmark outputs, so it should be treated as a research model rather than a production trading signal.
- The project avoids lookahead bias in the benchmark layer by training only on data known at each rebalance date.
- Ledoit-Wolf remains the recommended default covariance estimator for the classical benchmark layer.
- The ML model depends on PyTorch, while the rest of the classical stack does not.
- The promoted larger default model was selected because it materially improved realized excess return on the denser weekly offline sample and cleared a one-sided `p < 0.05` test versus the `SPY` benchmark on the train, validation, and full splits in the current run.
- The current ML defaults intentionally use weekly demonstrations to increase both the train and validation sample sizes materially; this is a data-density change, not a claim that weekly trading is inherently superior.

## Future Extensions

- transaction cost model inside the policy reward
- regime-aware training splits
- richer offline RL objectives
- factor overlays and factor-aware state features
- out-of-sample hyperparameter tuning split
- probabilistic confidence bands for policy evaluation

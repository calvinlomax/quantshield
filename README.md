# QuantShield

QuantShield is a local machine learning portfolio research project centered on a transformer-based actor-critic policy, with a cross-platform PySide6 desktop app as the main user-facing product. The classical portfolio engine is still in the repo, but it now serves a supporting role: it downloads market data from `yfinance`, generates constrained benchmark portfolios, and produces the demonstration weights used to train the policy.

The primary product workflow is:

1. Train or load a saved transformer actor-critic checkpoint
2. Launch the desktop app
3. Build a custom ticker universe in the desktop app, choose dates, benchmark, rebalance frequency, and starting capital
4. Download and cache market data locally from `yfinance`
5. Run deterministic policy inference across history
6. Watch the backtest replay step by step with playback controls and a synchronized scrubber

The canonical training and benchmark universe is the top 10 ETF basket used across the repo:
`VOO, IVV, SPY, VTI, QQQ, VEA, VUG, GLD, IEFA, VTV`

The broader ETF basket is still available for classical experiments in `config/broad_universe_config.yaml`:
`VOO, IVV, SPY, VTI, QQQ, VEA, VUG, GLD, IEFA, VTV`

## Why QuantShield

QuantShield is useful if you want a reproducible local workflow for portfolio ML without depending on pre-downloaded datasets or cloud-only research stacks.

- All training and inference data is fetched locally with Python from `yfinance`
- Raw prices are cached for reproducibility
- The ML model is benchmarked against explicit constrained allocation rules
- The policy is trained on saved weight histories instead of opaque manual labels
- The desktop app replays the saved policy over historical data instead of only producing static reports
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

This is an offline policy-learning setup. The classical optimizer suite is the demonstration generator and benchmark layer, not the main product. The main product surface is the desktop replay app that loads the trained checkpoint for inference.

## Project Architecture

```text
QuantShield/
  README.md
  requirements.txt
  requirements-rl.txt
  requirements-app.txt
  pyproject.toml
  config/
    broad_universe_config.yaml
    default_config.yaml
  data/
    raw/
    processed/
  outputs/
    figures/
    ml_tuned_objective_runs/
    rl_policy/
    tables/
    tuned_objective_runs/
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
    quantshield_app/
      __init__.py
      main.py
      services/
        checkpoint_service.py
        input_parser.py
        market_data_service.py
        replay_service.py
      ui/
        charts.py
        main_window.py
      viewmodels/
        replay_controller.py
  scripts/
    fetch_data.py
    run_desktop_app.py
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
    test_desktop_app.py
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

- Universe: `VOO, IVV, SPY, VTI, QQQ, VEA, VUG, GLD, IEFA, VTV`
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

### 5. Desktop Inference Replay

The desktop app is the default way to consume the trained model.

- It loads a saved `actor_critic_policy.pt` checkpoint
- It lets the user build a custom replay universe with an add-ticker popup and per-letter suggestions
- It downloads or reuses cached `yfinance` history for the selected dates
- It rebuilds policy state windows from the downloaded return history
- It runs deterministic policy inference through the saved actor-critic
- It simulates a historical replay step by step
- It shows synchronized charts, playback controls, current weights, and summary metrics

### 6. Evaluation

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

Install the desktop app dependency layer:

```bash
pip install -r requirements-app.txt
```

Optional editable installs:

```bash
pip install -e .[rl]
pip install -e .[app]
```

## Launch The Desktop App

This is the main inference interface for QuantShield:

```bash
python scripts/run_desktop_app.py
```

The app lets you:

- choose a saved checkpoint from the repo outputs
- use `Add Ticker` to open a popup search window with per-letter suggestions from `yfinance`
- build a custom replay universe with at least 5 selected tickers
- choose start date, end date, rebalance frequency, benchmark ticker, and starting capital
- run inference over the selected history
- watch the replay start automatically once preparation is complete
- use `Play`, `Pause`, `Restart`, `Step Back`, `Step Forward`, the speed selector, and the timeline slider to inspect any timestep

The desktop app uses cached downloads when possible. If the requested history is not cached, it fetches the required price data from `yfinance`, stores it under `data/raw/`, preprocesses returns locally, and then starts the replay.

Ticker selection rules:

- The backend policy is trained on the 10-ETF canonical universe listed above.
- The desktop app now supports arbitrary user-entered `yfinance` tickers at inference time.
- The app requires at least 5 selected tickers before it will run a replay.
- The `Add Ticker` popup allows either suggestion-based selection or direct manual symbol entry.

Optional checkpoint search override:

```bash
python scripts/run_desktop_app.py --checkpoint-root outputs/rl_policy
```

## Desktop Replay Flow

1. Train the policy with the backend ML pipeline, or reuse an existing checkpoint in `outputs/rl_policy/`.
2. Launch `python scripts/run_desktop_app.py`.
3. Select a checkpoint in the model selector.
4. Use `Add Ticker` to build a custom portfolio universe of at least 5 names.
5. Choose the start date, end date, rebalance frequency, benchmark, and starting capital.
6. Click `Run Replay`.
7. After replay preparation finishes, playback begins automatically.
8. Pause or scrub the slider to inspect earlier or later portfolio states instantly.

What “watching a backtest in real time” means in the app:

- The app is replaying historical inference, not streaming live prices.
- It advances through past timesteps one step at a time so the equity curve, current weights, return heat map, and benchmark comparison update continuously.
- The slider stays synchronized with playback and also lets you jump backward or forward immediately.

## Run The Backend ML Workflow

The command-line ML pipeline remains the backend path that creates the saved checkpoint used by the desktop app:

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

If you want to run individual backend layers instead of the main ML workflow:

Fetch and cache data only:

```bash
python scripts/fetch_data.py
```

Generate the tuned benchmark suite only:

```bash
python scripts/run_tuned_suite.py
```

That command writes the smaller monthly classical suite to `outputs/tuned_objective_runs/`.

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

### Desktop App Inputs

The desktop app consumes:

- `outputs/rl_policy/actor_critic_policy.pt`
- `outputs/rl_policy/rl_config.json`
- cached market data under `data/raw/`

During replay it renders live in-app charts for:

- cumulative portfolio value vs benchmark
- allocation history over time
- current allocation horizontal bar chart
- timestamp-anchored asset return heat map

### Backend ML Artifacts

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
mean_variance  VOO,IVV,SPY,VTI,QQQ,VEA,VUG,GLD,IEFA,VTV  0.1410  0.1194  0.0216  ...
equal_weight   VOO,IVV,SPY,VTI,QQQ,VEA,VUG,GLD,IEFA,VTV  0.1322  0.1194  0.0128  ...
risk_parity    VOO,IVV,SPY,VTI,QQQ,VEA,VUG,GLD,IEFA,VTV  0.1288  0.1194  0.0094  ...
min_variance   VOO,IVV,SPY,VTI,QQQ,VEA,VUG,GLD,IEFA,VTV  0.1215  0.1194  0.0021  ...

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

Desktop-app specific coverage includes:

- ticker parsing
- market-data preparation edge cases
- checkpoint discovery and loading
- replay stepping
- slider scrubbing state via the replay controller

## Notes And Assumptions

- The classical benchmark suite is backtest-tuned on the same historical sample, so its outperformance versus `SPY` is exploratory rather than out-of-sample evidence.
- The transformer actor-critic is trained offline from those benchmark outputs, so it should be treated as a research model rather than a production trading signal.
- The desktop app is an inference and replay surface, not a training interface.
- The policy is trained on the canonical 10-ETF universe, but the actor head is used at inference time on arbitrary user-selected `yfinance` baskets of at least 5 tickers.
- The project avoids lookahead bias in the benchmark layer by training only on data known at each rebalance date.
- Ledoit-Wolf remains the recommended default covariance estimator for the classical benchmark layer.
- The ML model depends on PyTorch, while the desktop app adds PySide6 on top of that.
- The promoted larger default model was selected because it materially improved realized excess return on the denser weekly offline sample and cleared a one-sided `p < 0.05` test versus the `SPY` benchmark on the train, validation, and full splits in the current run.
- The current ML defaults intentionally use weekly demonstrations to increase both the train and validation sample sizes materially; this is a data-density change, not a claim that weekly trading is inherently superior.

## Future Extensions

- transaction cost model inside the policy reward
- regime-aware training splits
- richer offline RL objectives
- factor overlays and factor-aware state features
- out-of-sample hyperparameter tuning split
- probabilistic confidence bands for policy evaluation

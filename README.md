# QuantShield

QuantShield is a local portfolio research and backtesting project with two primary surfaces:

1. A PySide6 desktop app for model selection, portfolio construction, historical backtest replay, and visual analysis.
2. A Python research stack for data ingestion, classical portfolio optimization, offline RL policy training, model scoring, and notebook-based analysis.

The repo is designed to run locally on cached `yfinance` data. It supports both classical portfolio methods and learned policy models, and it now includes multiple model families, duration-specific checkpoint suites, portfolio-specific fit workflows, and desktop tooling for 10-name and 50-name portfolios.

## What QuantShield Does

- Downloads and caches market data locally with `yfinance`
- Runs classical long-only portfolio optimization and rolling backtests
- Trains transformer-based actor-critic portfolio policies
- Scores models against multiple baselines, including benchmark, equal weight, restricted-random, and Markowitz mean-variance
- Ships duration-specific saved model suites for desktop inference
- Supports portfolio-specific model fitting from inside the desktop app
- Replays historical backtests step by step with synchronized charts and diagnostics
- Provides notebook helpers for price, technical analysis, and lightweight ML visualization

## Core Capabilities

### Desktop App

The desktop app is the main user-facing product. It lets you:

- Select a saved model from built-in suites, experimental runs, 50-slot suites, or portfolio-specific fits
- Switch between 10-name and 50-name model families
- Build or edit a portfolio with at least 5 tickers
- Save and reload full configurations, not just ticker lists
- Choose preset portfolios, including sector and category presets
- Run a historical backtest replay with integer-share execution
- Compare the active model against:
  - the chosen benchmark
  - equal weight
  - a long-only Markowitz mean-variance baseline
- Open a detailed summary window and a holdings popup
- Inspect company details with:
  - a profile summary
  - a recent price chart
  - technical indicators and analyst ratings
- Fit a new model for the currently selected portfolio and watch training progress live
- Launch a new model-training run directly from the Select Model window with:
  - configurable universe source
  - configurable benchmark target
  - editable basic and advanced hyperparameters
  - a dedicated non-modal monitor window with live CLI output
  - a dedicated non-modal monitor window with live training charts while the rest of the app remains interactive
  - optional 3D candidate-sweep visualization when real sweep data exists
  - an explicit save step before the model is registered back into the selector

### Research / Training Stack

The backend supports:

- data fetch and caching
- preprocessing and return generation
- risk estimation and optimization
- classical pipeline and rolling backtests
- tuned benchmark suite generation
- offline RL dataset construction
- actor-critic model training
- model scoring across saved checkpoints
- duration-specific checkpoint suite generation
- ETF benchmark-beating and Markowitz-aware experiments
- deterministic oracle-memory model generation for desktop use

## Model System

QuantShield now contains multiple model families rather than a single checkpoint.

### Horizons

The repo uses duration-specific replay and training horizons:

- `1mo`
- `3mo`
- `6mo`
- `1y`
- `3y`
- `5y`

In the desktop app, these appear as structured model groups such as:

- Tactical 1-Month
- Short-Horizon 3-Month
- Intermediate 6-Month
- Core 1-Year
- Long-Horizon 3-Year
- Strategic 5-Year

### Portfolio Sizes

The model selector supports two portfolio-size modes:

- `10`
- `50`

Changing the size mode updates the associated portfolio limits, default universe, preset portfolios, and compatible model list.

### Model Sources

The app discovers models from these roots:

- `outputs/replay_checkpoint_suites`
- `outputs/model_experiments`
- `outputs/model_experiments_50_suite`
- `outputs/portfolio_model_fits`
- `outputs/rl_policy`

### Model Tags

Saved models may carry quality tags such as:

- `Validated`: stronger stability / consistency
- `Benchmark+`: stronger excess-return behavior versus benchmark
- `Exploratory`: higher-risk experimental candidate

### Synthetic Slot Inference

Many of the desktop-facing models are trained against synthetic asset slots rather than fixed ticker identities. In practice, that means a saved 10-slot model can be applied to arbitrary user-selected 10-name baskets, and a 50-slot model can be applied to arbitrary user-selected 50-name baskets, subject to the app’s portfolio-size rules.

## Default Universes And Presets

### Canonical 10-Name Universe

The core 10-slot ETF universe is:

`VOO, IVV, SPY, VTI, QQQ, VEA, VUG, GLD, IEFA, VTV`

### Canonical 50-Name Universe

The 50-slot family extends the repo’s long-history default universe in [`src/quantshield/universe.py`](src/quantshield/universe.py).

### Built-In Preset Portfolios

The desktop app includes preset configurations through [`src/quantshield_app/services/portfolio_library_service.py`](src/quantshield_app/services/portfolio_library_service.py).

Examples include:

- `Technology Leaders`
- `Financial Compounders`
- `Healthcare Quality`
- `Energy & Industrials`
- `Consumer Staples & Brands`
- `AI & Semiconductors`
- `Expanded Core 50`
- `Expanded Quality 50`
- `Expanded Growth 50`

Saved user configurations are written to:

- `outputs/app_state/portfolios.json`

The built-in presets are curated against the repo's locally available cached market data so the desktop workflows do not surface known dead symbols unnecessarily.

## Installation

Python `3.11+` is required.

### Base Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Desktop App Setup

```bash
pip install -r requirements-app.txt
```

This installs the full desktop stack, including PyTorch, PySide6, and Pillow for Clyde image verification.

### Optional Editable Installs

```bash
pip install -e .[desktop]
pip install -e .[rl]
pip install -e .[app]
```

## Quick Start

### Launch The Desktop App

```bash
export QUANTSHIELD_CLYDE_CODE=KvW3xqat6ONEDpZgQxmMRDSnf9kmpw6W
python scripts/run_desktop_app.py
```

The app launches the main window defined in [`src/quantshield_app/main.py`](src/quantshield_app/main.py).
Startup now verifies that `assets/clyde.jpg` resolves to the pinned Clyde code and that the same code is present in `QUANTSHIELD_CLYDE_CODE`.

### Fetch And Cache Data

```bash
python scripts/fetch_data.py --config config/default_config.yaml
```

### Run The Classical Pipeline

```bash
python scripts/run_pipeline.py
```

### Run The Classical Rolling Backtest

```bash
python scripts/run_backtest.py
```

### Run The End-To-End ML Pipeline

```bash
python scripts/run_ml_pipeline.py
```

## Desktop App Overview

The desktop app code lives in [`src/quantshield_app`](src/quantshield_app).

### Main UI Areas

- Model selection
- Portfolio editing and presets
- Capital, benchmark, dates, and interval settings
- Current snapshot metrics
- Simulation controls
- Holdings popup
- Summary popup
- Equity curve, allocation, and heatmap charts

### Select Model Window

The Select Model dialog now handles both model discovery and model creation.

It supports:

- browsing built-in, experimental, RL, and portfolio-fit checkpoints
- switching between 10-name and 50-name model families
- sorting the model list by relative `Updated` age and other visible metadata columns
- comparing any 2 to 5 saved models side by side
- opening `New Model` to launch a training run without leaving the selector

### New Model Workflow

The `New Model` dialog is additive to the existing app. It does not replace the built-in suites and it does not duplicate training logic in the UI.

Instead, the dialog:

- validates the requested run configuration
- resolves the user-selected training universe
- resolves the benchmark target:
  - ticker benchmark
  - equal weight
  - Markowitz mean-variance
- exposes a compact launcher view with:
  - training universe
  - basic hyperparameters
  - dates / horizon / benchmark
- automatically resets the training window to `today - duration` when the selected duration changes
- moves advanced hyperparameters and launch preview into an `Advanced` popup
- launches one of the existing training scripts asynchronously
- hides the launcher during execution and opens a dedicated non-modal monitor window with:
  - loss graph
  - reward / objective graph
  - benchmark-relative graph
  - live CLI output
  - progress bar
  - optional gradient-descent / candidate-surface view
- allows the rest of the desktop app to remain interactive while the monitor is open
- returns to the launcher after training completes, where the user can:
  - review the completed run
  - reopen the full graph window
  - inspect a model summary
  - explicitly save and register the trained model

The dialog currently supports these training modes:

- `portfolio_fit`
- `experiment`
- `rl_policy`

Supported universe sources include:

- current portfolio
- saved presets
- canonical/default universe
- broad config universe
- manual ticker entry
- file import

The dialog enforces:

- minimum portfolio size: `5`
- compatibility with the selected model family size (`10` or `50`)
- valid benchmark configuration
- valid date range
- non-overwriting output paths
- explicit save-target validation before registration

### Portfolio Editing

The portfolio editor supports:

- manual ticker search
- save / load configuration
- presets
- reset to defaults
- company detail lookup
- portfolio-specific model fitting

The portfolio editor enforces:

- minimum portfolio size: `5`
- maximum portfolio size: determined by the selected model family (`10` or `50`)

### Company Detail Window

The About Company window includes:

- ticker and company name
- Yahoo Finance link
- company profile
- recent price graph
- technicals and analyst ratings table

### Backtest / Replay Behavior

The replay engine in [`src/quantshield_app/services/replay_service.py`](src/quantshield_app/services/replay_service.py):

- uses integer shares
- simulates portfolio value over historical prices
- computes benchmark, equal-weight, and Markowitz comparison series
- produces summary metrics used by the desktop summary window
- uses a duration-matched Treasury assumption for risk-free-rate reporting in the summary

### Chart Toggles

The cumulative chart can display:

- model portfolio
- benchmark
- equal weight
- Markowitz mean-variance
- drawdown overlay

## Training, Scoring, And Experiment Scripts

The main scripts live in [`scripts`](scripts).

### Data / Classical Research

- [`scripts/fetch_data.py`](scripts/fetch_data.py): fetch and cache price history from `yfinance`
- [`scripts/run_pipeline.py`](scripts/run_pipeline.py): classical benchmark workflow
- [`scripts/run_backtest.py`](scripts/run_backtest.py): classical rolling backtest
- [`scripts/run_tuned_suite.py`](scripts/run_tuned_suite.py): generate the tuned benchmark demonstration suite

### RL Training / Model Generation

- [`scripts/train_rl_policy.py`](scripts/train_rl_policy.py): train a transformer actor-critic from saved suite weights
- [`scripts/train_random_sp500_policy.py`](scripts/train_random_sp500_policy.py): train and select policies on randomized universes
- [`scripts/train_duration_checkpoint_suites.py`](scripts/train_duration_checkpoint_suites.py): build the full duration-specific replay suites
- [`scripts/train_benchmark_beating_duration_models.py`](scripts/train_benchmark_beating_duration_models.py): train additional duration models until each horizon has qualified candidates
- [`scripts/build_oracle_memory_models.py`](scripts/build_oracle_memory_models.py): build deterministic oracle-memory desktop models for 10-slot or 50-slot use
- [`scripts/fit_portfolio_model.py`](scripts/fit_portfolio_model.py): fit a new model directly to the currently chosen portfolio basket

The desktop app’s `New Model` dialog reuses these scripts rather than re-implementing training in the UI. The scripts now emit structured progress events and metadata so the app can show:

- live loss and reward charts
- benchmark-relative metrics
- streamed CLI output
- candidate sweep summaries
- saved model metadata for immediate checkpoint discovery

### Scoring / Evaluation

- [`scripts/score_saved_models.py`](scripts/score_saved_models.py): score saved checkpoints and write a scoreboard

### ML Workflow Wrapper

- [`scripts/run_ml_pipeline.py`](scripts/run_ml_pipeline.py): end-to-end ML workflow wrapper around tuned-suite generation and RL training

## Training Objectives And Baselines

QuantShield’s model training and scoring are not benchmark-only anymore.

The repo includes comparison and reward logic against:

- benchmark ETF return
- equal weight
- restricted-random weighting
- Markowitz mean-variance

Relevant implementation files include:

- [`src/quantshield/rl.py`](src/quantshield/rl.py)
- [`src/quantshield/model_scoring.py`](src/quantshield/model_scoring.py)
- [`src/quantshield/sp500_random_training.py`](src/quantshield/sp500_random_training.py)
- [`src/quantshield/tuned_suite.py`](src/quantshield/tuned_suite.py)

## Notebook Analysis

The notebooks live in [`notebooks`](notebooks):

- [`notebooks/exploratory_analysis.ipynb`](notebooks/exploratory_analysis.ipynb)
- [`notebooks/portfolio_demo.ipynb`](notebooks/portfolio_demo.ipynb)

Shared notebook helpers live in:

- [`src/quantshield/notebook_analysis.py`](src/quantshield/notebook_analysis.py)

The notebooks now include:

- price visualization
- technical analysis overlays and indicators
- lightweight ML-based return diagnostics
- portfolio-oriented demonstration charts

## Important Output Directories

Common generated outputs include:

- `data/raw`: cached market downloads
- `data/processed`: cleaned / aligned intermediate data
- `outputs/figures`: saved figures from research scripts
- `outputs/tables`: saved tables from research scripts
- `outputs/tuned_objective_runs`: classical benchmark demonstration runs
- `outputs/ml_tuned_objective_runs`: ML-oriented tuned suite outputs
- `outputs/rl_policy`: promoted RL checkpoint outputs
- `outputs/replay_checkpoint_suites`: built-in duration-specific desktop model suites
- `outputs/model_experiments`: experimental model runs
- `outputs/model_experiments_50_suite`: 50-slot desktop model suites
- `outputs/portfolio_model_fits`: user-triggered portfolio-specific fit runs
- `outputs/app_state`: app state and saved configurations

## Repository Layout

```text
QuantShield/
  README.md
  pyproject.toml
  requirements.txt
  requirements-rl.txt
  requirements-app.txt
  assets/
  config/
    default_config.yaml
    broad_universe_config.yaml
  data/
    raw/
    processed/
  notebooks/
    exploratory_analysis.ipynb
    portfolio_demo.ipynb
  outputs/
    app_state/
    figures/
    ml_tuned_objective_runs/
    model_experiments/
    model_experiments_50_suite/
    portfolio_model_fits/
    replay_checkpoint_suites/
    rl_policy/
    tables/
    tuned_objective_runs/
  scripts/
    build_oracle_memory_models.py
    fetch_data.py
    fit_portfolio_model.py
    run_backtest.py
    run_desktop_app.py
    run_ml_pipeline.py
    run_pipeline.py
    run_tuned_suite.py
    score_saved_models.py
    train_benchmark_beating_duration_models.py
    train_duration_checkpoint_suites.py
    train_random_sp500_policy.py
    train_rl_policy.py
  src/
    quantshield/
      attribution.py
      backtest.py
      config.py
      data_loader.py
      metrics.py
      model_scoring.py
      notebook_analysis.py
      optimization.py
      pipeline.py
      plotting.py
      preprocessing.py
      replay_durations.py
      reporting.py
      risk.py
      rl.py
      sp500_random_training.py
      stress_test.py
      tuned_suite.py
      universe.py
      utils.py
    quantshield_app/
      main.py
      services/
        checkpoint_service.py
        input_parser.py
        market_data_service.py
        model_training_service.py
        portfolio_library_service.py
        replay_service.py
        ticker_info_service.py
        ticker_search_service.py
        treasury_rate_service.py
      ui/
        charts.py
        checkpoint_dialog.py
        main_window.py
        new_model_dialog.py
        portfolio_dialogs.py
        ticker_search_dialog.py
        ticker_summary_dialog.py
      viewmodels/
        replay_controller.py
  tests/
    conftest.py
    test_data_loader.py
    test_desktop_app.py
    test_metrics.py
    test_optimization.py
    test_risk.py
    test_rl.py
    test_sp500_random_training.py
```

## Testing

Run the full test suite:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python -m pytest tests -q
```

Or run the desktop-focused tests:

```bash
MPLCONFIGDIR=/tmp/mpl PYTHONPATH=src .venv/bin/python -m pytest tests/test_desktop_app.py -q
```

For a theory-first treatment of the optimization, replay, reinforcement-learning, and software-engineering design, see [Methodology.md](Methodology.md).

## Notes

- The app and training scripts are built for local execution against cached `yfinance` data.
- Some scripts require the optional PyTorch dependency; the desktop app requires both PyTorch and PySide6.
- Generated outputs can be large and are intentionally separated across replay suites, experiments, and portfolio-fit directories.
- Existing built-in model suites are not overwritten by portfolio-specific fits; fit runs are saved separately under `outputs/portfolio_model_fits`.
- Models created from the `New Model` dialog are saved into the selected output category and are discoverable by the selector immediately after refresh.

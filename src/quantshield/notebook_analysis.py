"""Reusable notebook helpers for price, technical-analysis, and ML dashboards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


ML_FEATURE_COLUMNS = [
    "return_lag_1",
    "return_lag_5",
    "return_lag_10",
    "volatility_20",
    "sma_20_gap",
    "sma_50_gap",
    "bb_width",
    "rsi_14",
    "macd",
    "macd_hist",
]


def _finish_figure(fig: plt.Figure) -> None:
    fig.tight_layout()
    if "agg" in plt.get_backend().lower():
        plt.close(fig)
        return
    plt.show()


def build_ta_feature_frame(price: pd.Series, asset_returns: pd.Series) -> pd.DataFrame:
    """Create a technical-analysis feature table aligned to a single asset."""
    price_series = pd.Series(price, copy=True).rename("price").dropna()
    return_series = pd.Series(asset_returns, copy=True).rename("return").dropna()
    frame = pd.concat([price_series, return_series], axis=1, join="inner").dropna()

    frame["sma_20"] = frame["price"].rolling(20).mean()
    frame["sma_50"] = frame["price"].rolling(50).mean()
    rolling_std_20 = frame["price"].rolling(20).std()
    frame["bb_upper"] = frame["sma_20"] + 2.0 * rolling_std_20
    frame["bb_lower"] = frame["sma_20"] - 2.0 * rolling_std_20
    frame["bb_width"] = (frame["bb_upper"] - frame["bb_lower"]) / frame["sma_20"]

    delta = frame["price"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    average_loss = loss.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
    rs = average_gain / average_loss.replace(0.0, np.nan)
    frame["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    ema_12 = frame["price"].ewm(span=12, adjust=False).mean()
    ema_26 = frame["price"].ewm(span=26, adjust=False).mean()
    frame["macd"] = ema_12 - ema_26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_hist"] = frame["macd"] - frame["macd_signal"]

    frame["return_lag_1"] = frame["return"].shift(1)
    frame["return_lag_5"] = frame["return"].rolling(5).sum().shift(1)
    frame["return_lag_10"] = frame["return"].rolling(10).sum().shift(1)
    frame["volatility_20"] = frame["return"].rolling(20).std()
    frame["sma_20_gap"] = frame["price"] / frame["sma_20"] - 1.0
    frame["sma_50_gap"] = frame["price"] / frame["sma_50"] - 1.0
    frame["target_next_return"] = frame["return"].shift(-1)
    return frame.dropna()


def plot_price_technical_dashboard(frame: pd.DataFrame, ticker: str) -> None:
    """Plot price, RSI, and MACD for a single asset."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(frame.index, frame["price"], label="Close", linewidth=1.4, color="#1f77b4")
    axes[0].plot(frame.index, frame["sma_20"], label="SMA 20", linewidth=1.0, color="#ff7f0e")
    axes[0].plot(frame.index, frame["sma_50"], label="SMA 50", linewidth=1.0, color="#2ca02c")
    axes[0].fill_between(
        frame.index,
        frame["bb_lower"].to_numpy(),
        frame["bb_upper"].to_numpy(),
        alpha=0.15,
        color="#1f77b4",
        label="Bollinger Bands",
    )
    axes[0].set_title(f"{ticker} Price and Technical Overlay")
    axes[0].set_ylabel("Price")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(frame.index, frame["rsi_14"], color="#9467bd", linewidth=1.2)
    axes[1].axhline(70.0, color="#d62728", linestyle="--", linewidth=0.9)
    axes[1].axhline(30.0, color="#2ca02c", linestyle="--", linewidth=0.9)
    axes[1].set_title("RSI (14)")
    axes[1].set_ylabel("RSI")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].grid(alpha=0.25)

    axes[2].plot(frame.index, frame["macd"], label="MACD", linewidth=1.1, color="#1f77b4")
    axes[2].plot(frame.index, frame["macd_signal"], label="Signal", linewidth=1.1, color="#ff7f0e")
    axes[2].bar(frame.index, frame["macd_hist"], alpha=0.35, color="#7f7f7f", label="Histogram")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[2].set_title("MACD")
    axes[2].set_ylabel("MACD")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="upper left")

    _finish_figure(fig)


def fit_ml_return_model(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.80,
    alpha: float = 1.0,
) -> tuple[Ridge, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Fit a simple ridge model to next-day returns from TA-derived features."""
    if len(frame) < 80:
        raise ValueError("Need at least 80 observations to fit the notebook ML model.")

    split_index = int(len(frame) * float(train_fraction))
    split_index = max(split_index, 40)
    split_index = min(split_index, len(frame) - 20)

    train_frame = frame.iloc[:split_index]
    test_frame = frame.iloc[split_index:].copy()

    model = Ridge(alpha=float(alpha))
    model.fit(train_frame[ML_FEATURE_COLUMNS], train_frame["target_next_return"])

    predictions = pd.Series(
        model.predict(test_frame[ML_FEATURE_COLUMNS]),
        index=test_frame.index,
        name="predicted_next_return",
    )
    evaluation = pd.DataFrame(
        {
            "actual_next_return": test_frame["target_next_return"],
            "predicted_next_return": predictions,
        }
    )
    evaluation["ml_signal"] = (evaluation["predicted_next_return"] > 0.0).astype(float)
    evaluation["ml_strategy_return"] = evaluation["ml_signal"] * evaluation["actual_next_return"]
    evaluation["buy_and_hold_growth"] = (1.0 + evaluation["actual_next_return"]).cumprod()
    evaluation["ml_growth"] = (1.0 + evaluation["ml_strategy_return"]).cumprod()

    metrics = pd.DataFrame(
        {
            "train_samples": [len(train_frame)],
            "test_samples": [len(test_frame)],
            "mean_absolute_error": [mean_absolute_error(evaluation["actual_next_return"], evaluation["predicted_next_return"])],
            "r_squared": [r2_score(evaluation["actual_next_return"], evaluation["predicted_next_return"])],
            "buy_and_hold_total_return": [evaluation["buy_and_hold_growth"].iloc[-1] - 1.0],
            "ml_strategy_total_return": [evaluation["ml_growth"].iloc[-1] - 1.0],
            "prediction_hit_rate": [
                (
                    np.sign(evaluation["predicted_next_return"]).replace(0.0, 1.0)
                    == np.sign(evaluation["actual_next_return"]).replace(0.0, 1.0)
                ).mean()
            ],
        },
        index=["metrics"],
    )
    feature_importance = pd.Series(model.coef_, index=ML_FEATURE_COLUMNS, name="ridge_coefficient").sort_values()
    return model, evaluation, metrics, feature_importance


def plot_ml_dashboard(
    evaluation: pd.DataFrame,
    feature_importance: pd.Series,
    ticker: str,
) -> None:
    """Plot ML prediction diagnostics for a single asset."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)

    axes[0].plot(evaluation.index, evaluation["actual_next_return"], label="Actual", linewidth=1.1, color="#1f77b4")
    axes[0].plot(
        evaluation.index,
        evaluation["predicted_next_return"],
        label="Predicted",
        linewidth=1.1,
        color="#ff7f0e",
    )
    axes[0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_title(f"{ticker} Next-Day Return: Actual vs Ridge Prediction")
    axes[0].set_ylabel("Return")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper left")

    axes[1].plot(evaluation.index, evaluation["buy_and_hold_growth"], label="Buy & Hold", linewidth=1.2, color="#1f77b4")
    axes[1].plot(evaluation.index, evaluation["ml_growth"], label="ML Strategy", linewidth=1.2, color="#2ca02c")
    axes[1].set_title("Out-of-Sample Growth")
    axes[1].set_ylabel("Growth of $1")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper left")

    top_coefficients = feature_importance.sort_values()
    axes[2].barh(top_coefficients.index, top_coefficients.values, color="#7f7f7f")
    axes[2].set_title("Ridge Feature Coefficients")
    axes[2].set_xlabel("Coefficient")
    axes[2].grid(axis="x", alpha=0.25)

    _finish_figure(fig)

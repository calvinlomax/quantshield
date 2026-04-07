"""Matplotlib Qt canvases used by the desktop replay UI."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from quantshield_app.services.replay_service import PolicyReplayResult, ReplayFrame


class BaseReplayCanvas(FigureCanvasQTAgg):
    """Base matplotlib canvas embedded in the Qt desktop app."""

    def __init__(self, *, height: float) -> None:
        figure = Figure(figsize=(8.0, height), tight_layout=True)
        super().__init__(figure)
        self.figure = figure
        self.axes = self.figure.add_subplot(111)


class EquityCurveCanvas(BaseReplayCanvas):
    """Animated cumulative return comparison chart."""

    def __init__(self) -> None:
        super().__init__(height=3.2)
        self._dates: pd.DatetimeIndex | None = None
        self._portfolio_values: np.ndarray | None = None
        self._benchmark_values: np.ndarray | None = None
        self._portfolio_line = None
        self._benchmark_line = None
        self._portfolio_marker = None
        self._benchmark_marker = None

    def set_result(self, result: PolicyReplayResult) -> None:
        values = result.cumulative_values.copy()
        self.axes.clear()
        self._dates = pd.DatetimeIndex(values.index)
        self._portfolio_values = values["Portfolio"].to_numpy(dtype=float)
        self._benchmark_values = values["Benchmark"].to_numpy(dtype=float)

        self._portfolio_line, = self.axes.plot([], [], linewidth=1.8, label="Portfolio")
        self._benchmark_line, = self.axes.plot([], [], linewidth=1.5, label=f"Benchmark ({result.benchmark_ticker})")
        self._portfolio_marker, = self.axes.plot([], [], marker="o", linestyle="", markersize=4)
        self._benchmark_marker, = self.axes.plot([], [], marker="o", linestyle="", markersize=4)

        if len(self._dates) > 0:
            ymin = min(float(np.min(self._portfolio_values)), float(np.min(self._benchmark_values)))
            ymax = max(float(np.max(self._portfolio_values)), float(np.max(self._benchmark_values)))
            margin = max((ymax - ymin) * 0.08, max(ymax, 1.0) * 0.03)
            self.axes.set_xlim(self._dates[0], self._dates[-1])
            self.axes.set_ylim(ymin - margin, ymax + margin)

        self.axes.set_title("Cumulative Returns vs Benchmark")
        self.axes.set_ylabel("Portfolio Value")
        self.axes.grid(alpha=0.25)
        self.axes.legend(loc="best")
        self.update_frame(0)

    def update_frame(self, frame_index: int) -> None:
        if self._dates is None or self._portfolio_values is None or self._benchmark_values is None:
            return
        clamped = max(0, min(frame_index, len(self._dates) - 1))
        visible_dates = self._dates[: clamped + 1]
        portfolio_values = self._portfolio_values[: clamped + 1]
        benchmark_values = self._benchmark_values[: clamped + 1]

        self._portfolio_line.set_data(visible_dates, portfolio_values)
        self._benchmark_line.set_data(visible_dates, benchmark_values)
        self._portfolio_marker.set_data([visible_dates[-1]], [portfolio_values[-1]])
        self._benchmark_marker.set_data([visible_dates[-1]], [benchmark_values[-1]])
        self.draw_idle()


class AllocationHistoryCanvas(BaseReplayCanvas):
    """Historical allocation chart with a moving replay cursor."""

    def __init__(self) -> None:
        super().__init__(height=2.7)
        self._dates: pd.DatetimeIndex | None = None
        self._cursor = None

    def set_result(self, result: PolicyReplayResult) -> None:
        history = result.daily_weights.reindex(columns=result.requested_tickers).fillna(0.0)
        self.axes.clear()
        self._dates = pd.DatetimeIndex(history.index)

        if not history.empty:
            self.axes.stackplot(self._dates, history.T.values, labels=history.columns, alpha=0.90)
            if len(history.columns) <= 6:
                self.axes.legend(loc="upper left", ncols=min(len(history.columns), 3))
            self._cursor = self.axes.axvline(self._dates[0], color="black", linewidth=1.1, alpha=0.65)

        self.axes.set_title("Portfolio Allocation Through Time")
        self.axes.set_ylabel("Weight")
        self.axes.set_ylim(0.0, 1.0)
        self.axes.grid(alpha=0.20)
        self.draw_idle()

    def update_frame(self, frame_index: int) -> None:
        if self._dates is None or len(self._dates) == 0 or self._cursor is None:
            return
        clamped = max(0, min(frame_index, len(self._dates) - 1))
        date = self._dates[clamped]
        self._cursor.set_xdata([date, date])
        self.draw_idle()


class CurrentAllocationCanvas(BaseReplayCanvas):
    """Horizontal bar chart of current policy weights at the active replay step."""

    def __init__(self) -> None:
        super().__init__(height=2.6)
        self._tickers: list[str] = []
        self._bars = []

    def set_result(self, result: PolicyReplayResult) -> None:
        self.axes.clear()
        self._tickers = list(result.requested_tickers)
        bar_positions = np.arange(len(self._tickers))
        self._bars = self.axes.barh(bar_positions, np.zeros(len(self._tickers)))
        self.axes.set_yticks(bar_positions)
        self.axes.set_yticklabels(self._tickers)
        self.axes.set_xlim(0.0, 1.0)
        self.axes.set_xlabel("Weight")
        self.axes.set_title("Current Model Allocation")
        self.axes.grid(axis="x", alpha=0.25)
        self.draw_idle()

    def update_frame(self, frame: ReplayFrame) -> None:
        if not self._bars:
            return
        weights = frame.weights.reindex(self._tickers).fillna(0.0)
        for bar, value in zip(self._bars, weights.to_numpy(dtype=float), strict=True):
            bar.set_width(float(value))
        self.axes.set_title(f"Current Model Allocation ({frame.date.date().isoformat()})")
        self.draw_idle()


class TimestampHeatmapCanvas(BaseReplayCanvas):
    """Dynamic asset-return heatmap anchored to the active replay timestamp."""

    def __init__(self, *, trailing_window: int = 30) -> None:
        super().__init__(height=2.6)
        self.trailing_window = trailing_window
        self._asset_returns: pd.DataFrame | None = None
        self._colorbar = None

    def set_result(self, result: PolicyReplayResult) -> None:
        self._asset_returns = result.asset_returns.reindex(columns=result.requested_tickers).fillna(0.0)
        self.update_frame(0)

    def update_frame(self, frame_index: int) -> None:
        if self._asset_returns is None or self._asset_returns.empty:
            return
        clamped = max(0, min(frame_index, len(self._asset_returns.index) - 1))
        window = self._asset_returns.iloc[max(0, clamped - self.trailing_window + 1) : clamped + 1]

        self.axes.clear()
        image = self.axes.imshow(window.T.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", interpolation="nearest")
        self.axes.set_title(f"Trailing Return Heatmap Through {window.index[-1].date().isoformat()}")
        self.axes.set_ylabel("Ticker")
        self.axes.set_xlabel("Time Step")
        self.axes.set_yticks(np.arange(len(window.columns)))
        self.axes.set_yticklabels(list(window.columns))

        if len(window.index) >= 3:
            tick_positions = [0, len(window.index) // 2, len(window.index) - 1]
        else:
            tick_positions = list(range(len(window.index)))
        self.axes.set_xticks(tick_positions)
        self.axes.set_xticklabels([window.index[position].strftime("%Y-%m-%d") for position in tick_positions], rotation=20, ha="right")

        if self._colorbar is None:
            self._colorbar = self.figure.colorbar(image, ax=self.axes, shrink=0.85)
            self._colorbar.set_label("Return")
        else:
            self._colorbar.update_normal(image)
        self.draw_idle()

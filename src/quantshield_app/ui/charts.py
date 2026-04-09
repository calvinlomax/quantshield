"""Matplotlib Qt canvases used by the desktop replay UI."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from quantshield.metrics import drawdown_series
from quantshield_app.services.replay_service import PolicyReplayResult, ReplayFrame


def _format_compact_currency(value: float, _position: float) -> str:
    """Render axis currency labels like $400k, $1.5M, or $1.25B."""
    absolute = abs(float(value))
    sign = "-" if value < 0 else ""
    units = ((1_000_000_000.0, "B"), (1_000_000.0, "M"), (1_000.0, "k"))
    for divisor, suffix in units:
        if absolute >= divisor:
            scaled = absolute / divisor
            if scaled >= 100.0:
                decimals = 0
            elif scaled >= 10.0:
                decimals = 1
            else:
                decimals = 2
            formatted = f"{scaled:.{decimals}f}"
            if "." in formatted:
                formatted = formatted.rstrip("0").rstrip(".")
            return f"{sign}${formatted}{suffix}"
    return f"{sign}${absolute:,.0f}"


class BaseReplayCanvas(FigureCanvasQTAgg):
    """Base matplotlib canvas embedded in the Qt desktop app."""

    def __init__(self, *, height: float) -> None:
        figure = Figure(figsize=(8.0, height), constrained_layout=True)
        super().__init__(figure)
        self.figure = figure
        self.axes = self.figure.add_subplot(111)


class EquityCurveCanvas(BaseReplayCanvas):
    """Animated cumulative return comparison chart."""

    def __init__(self) -> None:
        super().__init__(height=3.2)
        self._dates: pd.DatetimeIndex | None = None
        self._portfolio_values: np.ndarray | None = None
        self._equal_weight_values: np.ndarray | None = None
        self._benchmark_values: np.ndarray | None = None
        self._portfolio_drawdown: np.ndarray | None = None
        self._portfolio_line = None
        self._equal_weight_line = None
        self._benchmark_line = None
        self._portfolio_marker = None
        self._equal_weight_marker = None
        self._benchmark_marker = None
        self._drawdown_axis = None
        self._drawdown_line = None
        self._drawdown_visible = False
        self._equal_weight_visible = False

    def set_result(self, result: PolicyReplayResult) -> None:
        values = result.cumulative_values.copy()
        self.figure.clear()
        self.axes = self.figure.add_subplot(111)
        self._drawdown_axis = None
        self._drawdown_line = None
        self._portfolio_line = None
        self._equal_weight_line = None
        self._benchmark_line = None
        self._portfolio_marker = None
        self._equal_weight_marker = None
        self._benchmark_marker = None
        self._dates = pd.DatetimeIndex(values.index)
        self._portfolio_values = values["Portfolio"].to_numpy(dtype=float)
        self._equal_weight_values = values["Equal Weight"].to_numpy(dtype=float) if "Equal Weight" in values.columns else None
        self._benchmark_values = values["Benchmark"].to_numpy(dtype=float)
        self._portfolio_drawdown = drawdown_series(result.comparison_returns["portfolio"]).to_numpy(dtype=float)

        self._portfolio_line, = self.axes.plot([], [], linewidth=1.8, label="Portfolio")
        self._equal_weight_line, = self.axes.plot([], [], linewidth=1.4, linestyle=":", color="#2ca02c", label="Equal Weight")
        self._benchmark_line, = self.axes.plot([], [], linewidth=1.5, label=f"{result.benchmark_ticker} Benchmark")
        self._portfolio_marker, = self.axes.plot([], [], marker="o", linestyle="", markersize=4)
        self._equal_weight_marker, = self.axes.plot([], [], marker="o", linestyle="", markersize=3.5, color="#2ca02c")
        self._benchmark_marker, = self.axes.plot([], [], marker="o", linestyle="", markersize=4)

        if len(self._dates) > 0:
            series_min = [float(np.min(self._portfolio_values)), float(np.min(self._benchmark_values))]
            series_max = [float(np.max(self._portfolio_values)), float(np.max(self._benchmark_values))]
            if self._equal_weight_values is not None:
                series_min.append(float(np.min(self._equal_weight_values)))
                series_max.append(float(np.max(self._equal_weight_values)))
            ymin = min(series_min)
            ymax = max(series_max)
            margin = max((ymax - ymin) * 0.08, max(ymax, 1.0) * 0.03)
            self.axes.set_xlim(self._dates[0], self._dates[-1])
            self.axes.set_ylim(ymin - margin, ymax + margin)

        self.axes.set_title("Cumulative Returns vs Benchmark")
        self.axes.set_ylabel("Portfolio Value")
        self.axes.yaxis.set_major_formatter(FuncFormatter(_format_compact_currency))
        self.axes.grid(alpha=0.25)
        self._drawdown_axis = self.axes.twinx()
        self._drawdown_line, = self._drawdown_axis.plot([], [], color="#d62728", linewidth=1.2, linestyle="--")
        self._drawdown_axis.set_ylabel("Drawdown")
        self._drawdown_axis.set_ylim(-1.0, 0.05)
        self._drawdown_axis.set_visible(self._drawdown_visible)
        self.set_equal_weight_visible(self._equal_weight_visible)
        self.update_frame(0)

    def set_drawdown_visible(self, visible: bool) -> None:
        self._drawdown_visible = bool(visible)
        if self._drawdown_axis is not None:
            self._drawdown_axis.set_visible(self._drawdown_visible)
        self.draw_idle()

    def set_equal_weight_visible(self, visible: bool) -> None:
        self._equal_weight_visible = bool(visible)
        if self._equal_weight_line is not None:
            self._equal_weight_line.set_visible(self._equal_weight_visible)
        if self._equal_weight_marker is not None:
            self._equal_weight_marker.set_visible(self._equal_weight_visible)
        self._refresh_legend()
        self.draw_idle()

    def update_frame(self, frame_index: int) -> None:
        if self._dates is None or self._portfolio_values is None or self._benchmark_values is None:
            return
        clamped = max(0, min(frame_index, len(self._dates) - 1))
        visible_dates = self._dates[: clamped + 1]
        portfolio_values = self._portfolio_values[: clamped + 1]
        equal_weight_values = self._equal_weight_values[: clamped + 1] if self._equal_weight_values is not None else None
        benchmark_values = self._benchmark_values[: clamped + 1]

        self._portfolio_line.set_data(visible_dates, portfolio_values)
        if self._equal_weight_line is not None and equal_weight_values is not None:
            self._equal_weight_line.set_data(visible_dates, equal_weight_values)
        self._benchmark_line.set_data(visible_dates, benchmark_values)
        self._portfolio_marker.set_data([visible_dates[-1]], [portfolio_values[-1]])
        if self._equal_weight_marker is not None and equal_weight_values is not None:
            self._equal_weight_marker.set_data([visible_dates[-1]], [equal_weight_values[-1]])
        self._benchmark_marker.set_data([visible_dates[-1]], [benchmark_values[-1]])
        if self._drawdown_axis is not None and self._drawdown_line is not None and self._portfolio_drawdown is not None:
            self._drawdown_line.set_data(visible_dates, self._portfolio_drawdown[: clamped + 1])
        self.draw_idle()

    def _refresh_legend(self) -> None:
        handles = []
        if self._portfolio_line is not None:
            handles.append(self._portfolio_line)
        if self._equal_weight_visible and self._equal_weight_line is not None:
            handles.append(self._equal_weight_line)
        if self._benchmark_line is not None:
            handles.append(self._benchmark_line)
        if handles:
            self.axes.legend(handles=handles, loc="best")


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
        self._xmax: float = 1.0

    def set_result(self, result: PolicyReplayResult) -> None:
        self._tickers = list(result.requested_tickers)
        aligned_weights = result.daily_weights.reindex(columns=sorted(result.requested_tickers)).fillna(0.0)
        observed_max = float(aligned_weights.to_numpy(dtype=float).max()) if not aligned_weights.empty else 0.0
        self._xmax = max(observed_max * 1.10, 0.05)
        self.axes.clear()
        self.draw_idle()

    def update_frame(self, frame: ReplayFrame) -> None:
        self.axes.clear()
        ordered_tickers = sorted(self._tickers) if self._tickers else sorted(frame.weights.index.tolist())
        weights = frame.weights.reindex(ordered_tickers).fillna(0.0).astype(float)
        positions = np.arange(len(weights))
        bars = self.axes.barh(positions, weights.to_numpy(dtype=float))
        self.axes.set_yticks(positions)
        self.axes.set_yticklabels(list(weights.index))
        self.axes.invert_yaxis()
        self.axes.set_xlim(0.0, self._xmax)
        self.axes.set_xlabel("Weight")
        self.axes.set_title(f"Allocation ({frame.date.strftime('%m/%d/%y')})")
        for bar, value in zip(bars, weights.to_numpy(dtype=float), strict=True):
            self.axes.text(bar.get_width(), bar.get_y() + bar.get_height() / 2.0, f" {value:.1%}", va="center", fontsize=8)
        self.axes.grid(axis="x", alpha=0.25)
        self.draw_idle()


class TimestampHeatmapCanvas(BaseReplayCanvas):
    """Dynamic asset-return heatmap anchored to the active replay timestamp."""

    def __init__(self, *, trailing_window: int = 30) -> None:
        super().__init__(height=2.6)
        self.trailing_window = trailing_window
        self._asset_returns: pd.DataFrame | None = None
        self._colorbar = None
        self._annotation = self.axes.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
        )
        self._annotation.set_visible(False)
        self.mpl_connect("motion_notify_event", self._on_hover)
        self._window: pd.DataFrame | None = None
        self._image = None

    def set_result(self, result: PolicyReplayResult) -> None:
        self._asset_returns = result.asset_returns.reindex(columns=result.requested_tickers).fillna(0.0)
        self.update_frame(0)

    def update_frame(self, frame_index: int) -> None:
        if self._asset_returns is None or self._asset_returns.empty:
            return
        clamped = max(0, min(frame_index, len(self._asset_returns.index) - 1))
        window = self._asset_returns.iloc[max(0, clamped - self.trailing_window + 1) : clamped + 1]
        self._window = window
        vmax = float(np.nanmax(np.abs(window.to_numpy(dtype=float)))) if not window.empty else 0.0
        vmax = max(vmax, 1e-6)

        self.axes.clear()
        self._image = self.axes.imshow(
            window.T.to_numpy(dtype=float),
            aspect="auto",
            cmap="coolwarm",
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        self.axes.set_title("Returns Heatmap")
        self.axes.set_yticks(np.arange(len(window.columns)))
        self.axes.set_yticklabels(list(window.columns))
        self.axes.tick_params(axis="y", labelsize=8)

        tick_positions = [0, len(window.index) - 1] if len(window.index) > 1 else [0]
        self.axes.set_xticks(tick_positions)
        self.axes.set_xticklabels([window.index[position].strftime("%d/%m/%y") for position in tick_positions], rotation=0, ha="center")

        if self._colorbar is None:
            self._colorbar = self.figure.colorbar(self._image, ax=self.axes, shrink=0.85)
            self._colorbar.set_label("Daily Return")
        else:
            self._colorbar.update_normal(self._image)
        self._colorbar.set_ticks([-vmax, 0.0, vmax])
        self._colorbar.set_ticklabels([f"{-vmax:.1%}", "0.0%", f"+{vmax:.1%}"])
        self._annotation.set_visible(False)
        self.draw_idle()

    def _on_hover(self, event) -> None:  # pragma: no cover - interactive UI behavior
        if self._window is None or self._image is None or event.inaxes != self.axes or event.xdata is None or event.ydata is None:
            if self._annotation.get_visible():
                self._annotation.set_visible(False)
                self.draw_idle()
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        if x < 0 or y < 0 or x >= len(self._window.index) or y >= len(self._window.columns):
            if self._annotation.get_visible():
                self._annotation.set_visible(False)
                self.draw_idle()
            return
        value = float(self._window.iloc[x, y])
        self._annotation.xy = (event.xdata, event.ydata)
        self._annotation.set_text(f"{self._window.columns[y]} | {self._window.index[x].strftime('%Y-%m-%d')}\n{value:.2%}")
        self._annotation.set_visible(True)
        self.draw_idle()

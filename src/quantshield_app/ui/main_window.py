"""Main desktop window for replaying QuantShield policy backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PySide6.QtCore import QObject, QDate, QThread, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from quantshield.utils import format_percent
from quantshield_app.services import CheckpointDescriptor, CheckpointService, MarketDataService, ReplayService, parse_ticker_input
from quantshield_app.services.replay_service import PolicyReplayResult
from quantshield_app.ui.charts import AllocationHistoryCanvas, CurrentAllocationCanvas, EquityCurveCanvas
from quantshield_app.viewmodels import ReplayController


DEFAULT_START_DATE = QDate(2018, 1, 1)
DEFAULT_REBALANCE_FREQUENCIES = ["W-FRI", "W-MON", "M"]
PLAYBACK_SPEEDS_MS: dict[str, int] = {
    "0.5x": 500,
    "1x": 250,
    "2x": 130,
    "4x": 70,
    "8x": 35,
}


@dataclass(slots=True)
class ReplayPreparationRequest:
    """Parameters handed from the UI thread to the background replay worker."""

    checkpoint_path: Path
    portfolio_tickers: list[str]
    start_date: str
    end_date: str | None
    rebalance_frequency: str
    benchmark_ticker: str
    starting_capital: float
    force_refresh: bool = False


class ReplayPreparationWorker(QObject):
    """Prepare replay data off the UI thread."""

    finished = Signal(object)
    error = Signal(str)
    status = Signal(str)

    def __init__(
        self,
        *,
        checkpoint_service: CheckpointService,
        market_data_service: MarketDataService,
        replay_service: ReplayService,
        request: ReplayPreparationRequest,
    ) -> None:
        super().__init__()
        self.checkpoint_service = checkpoint_service
        self.market_data_service = market_data_service
        self.replay_service = replay_service
        self.request = request

    def run(self) -> None:
        try:
            self.status.emit("Loading saved actor-critic checkpoint...")
            checkpoint = self.checkpoint_service.load_checkpoint(self.request.checkpoint_path, device="cpu")

            self.status.emit("Downloading or loading cached market data from yfinance...")
            market_data = self.market_data_service.prepare_market_data(
                portfolio_tickers=self.request.portfolio_tickers,
                benchmark_ticker=self.request.benchmark_ticker,
                start_date=self.request.start_date,
                end_date=self.request.end_date,
                lookback_window=checkpoint.training_config.lookback_window,
                force_refresh=self.request.force_refresh,
            )

            self.status.emit("Running deterministic inference and building replay frames...")
            replay_result = self.replay_service.build_replay(
                checkpoint=checkpoint,
                market_data=market_data,
                rebalance_frequency=self.request.rebalance_frequency,
                starting_capital=self.request.starting_capital,
            )
        except Exception as exc:  # pragma: no cover - exercised via integration/UI flow
            self.error.emit(str(exc))
            return

        self.finished.emit(replay_result)


class QuantShieldMainWindow(QMainWindow):
    """Primary desktop interface for QuantShield policy inference and replay."""

    def __init__(
        self,
        *,
        checkpoint_service: CheckpointService | None = None,
        market_data_service: MarketDataService | None = None,
        replay_service: ReplayService | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("QuantShield Desktop Replay")
        self.resize(1480, 920)

        self.checkpoint_service = checkpoint_service or CheckpointService()
        self.market_data_service = market_data_service or MarketDataService()
        self.replay_service = replay_service or ReplayService()
        self.controller = ReplayController()
        self.replay_result: PolicyReplayResult | None = None
        self._current_descriptor: CheckpointDescriptor | None = None
        self._worker_thread: QThread | None = None
        self._worker: ReplayPreparationWorker | None = None
        self._slider_is_syncing = False

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._load_checkpoints()
        self._set_playback_enabled(False)

    def closeEvent(self, event: Any) -> None:  # pragma: no cover - UI cleanup
        self.playback_timer.stop()
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(1000)
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)

        outer_layout = QHBoxLayout(root)
        splitter = QSplitter(Qt.Orientation.Horizontal, root)
        outer_layout.addWidget(splitter)

        controls_panel = QWidget(splitter)
        controls_panel.setMinimumWidth(410)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(12, 12, 12, 12)
        controls_layout.setSpacing(12)

        controls_layout.addWidget(self._build_run_group())
        controls_layout.addWidget(self._build_playback_group())
        controls_layout.addWidget(self._build_current_state_group())
        controls_layout.addWidget(self._build_summary_group())
        controls_layout.addWidget(self._build_weights_group())
        controls_layout.addStretch(1)

        charts_panel = QWidget(splitter)
        charts_layout = QVBoxLayout(charts_panel)
        charts_layout.setContentsMargins(12, 12, 12, 12)
        charts_layout.setSpacing(10)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal, charts_panel)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setTracking(True)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        charts_layout.addWidget(self.timeline_slider)

        self.timeline_label = QLabel("Replay position: 0 / 0")
        charts_layout.addWidget(self.timeline_label)

        self.equity_canvas = EquityCurveCanvas()
        self.allocation_history_canvas = AllocationHistoryCanvas()
        self.current_allocation_canvas = CurrentAllocationCanvas()

        charts_layout.addWidget(self.equity_canvas, stretch=5)
        charts_layout.addWidget(self.allocation_history_canvas, stretch=4)
        charts_layout.addWidget(self.current_allocation_canvas, stretch=3)

        splitter.addWidget(controls_panel)
        splitter.addWidget(charts_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("Replay Inputs")
        layout = QFormLayout(group)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        self.checkpoint_combo = QComboBox(group)
        self.checkpoint_combo.currentIndexChanged.connect(self._on_checkpoint_changed)

        self.ticker_input = QLineEdit(group)
        self.ticker_input.setPlaceholderText("SPY,QQQ,GLD")

        self.start_date_edit = QDateEdit(DEFAULT_START_DATE, group)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")

        self.end_date_edit = QDateEdit(QDate.currentDate(), group)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")

        self.rebalance_combo = QComboBox(group)
        self.rebalance_combo.addItems(DEFAULT_REBALANCE_FREQUENCIES)
        self.rebalance_combo.setCurrentText("W-FRI")

        self.benchmark_combo = QComboBox(group)
        self.benchmark_combo.setEditable(True)
        self.benchmark_combo.addItems(["SPY", "QQQ", "GLD"])
        self.benchmark_combo.setCurrentText("SPY")

        self.capital_input = QDoubleSpinBox(group)
        self.capital_input.setRange(100.0, 1_000_000_000.0)
        self.capital_input.setDecimals(2)
        self.capital_input.setSingleStep(10_000.0)
        self.capital_input.setValue(100_000.0)
        self.capital_input.setPrefix("$")

        self.run_button = QPushButton("Run Replay", group)
        self.run_button.clicked.connect(self._start_replay_preparation)

        layout.addRow("Checkpoint", self.checkpoint_combo)
        layout.addRow("Tickers", self.ticker_input)
        layout.addRow("Start Date", self.start_date_edit)
        layout.addRow("End Date", self.end_date_edit)
        layout.addRow("Rebalance", self.rebalance_combo)
        layout.addRow("Benchmark", self.benchmark_combo)
        layout.addRow("Starting Capital", self.capital_input)
        layout.addRow(self.run_button)
        return group

    def _build_playback_group(self) -> QGroupBox:
        group = QGroupBox("Playback Controls")
        layout = QGridLayout(group)

        self.play_button = QPushButton("Play", group)
        self.pause_button = QPushButton("Pause", group)
        self.restart_button = QPushButton("Restart", group)
        self.step_back_button = QPushButton("Step Back", group)
        self.step_forward_button = QPushButton("Step Forward", group)
        self.speed_combo = QComboBox(group)
        self.speed_combo.addItems(list(PLAYBACK_SPEEDS_MS.keys()))
        self.speed_combo.setCurrentText("2x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)

        self.play_button.clicked.connect(self._play)
        self.pause_button.clicked.connect(self._pause)
        self.restart_button.clicked.connect(self._restart)
        self.step_back_button.clicked.connect(self._step_backward)
        self.step_forward_button.clicked.connect(self._step_forward)

        layout.addWidget(self.play_button, 0, 0)
        layout.addWidget(self.pause_button, 0, 1)
        layout.addWidget(self.restart_button, 0, 2)
        layout.addWidget(self.step_back_button, 1, 0)
        layout.addWidget(self.step_forward_button, 1, 1)
        layout.addWidget(QLabel("Speed"), 2, 0)
        layout.addWidget(self.speed_combo, 2, 1, 1, 2)
        return group

    def _build_current_state_group(self) -> QGroupBox:
        group = QGroupBox("Current Replay State")
        layout = QFormLayout(group)

        self.current_date_label = QLabel("—")
        self.current_portfolio_value_label = QLabel("—")
        self.current_benchmark_value_label = QLabel("—")
        self.current_return_label = QLabel("—")
        self.current_benchmark_return_label = QLabel("—")
        self.current_excess_label = QLabel("—")
        self.current_turnover_label = QLabel("—")
        self.current_rebalance_label = QLabel("—")

        layout.addRow("Date", self.current_date_label)
        layout.addRow("Portfolio Value", self.current_portfolio_value_label)
        layout.addRow("Benchmark Value", self.current_benchmark_value_label)
        layout.addRow("Portfolio Return", self.current_return_label)
        layout.addRow("Benchmark Return", self.current_benchmark_return_label)
        layout.addRow("Excess Return", self.current_excess_label)
        layout.addRow("Turnover", self.current_turnover_label)
        layout.addRow("Rebalanced", self.current_rebalance_label)
        return group

    def _build_summary_group(self) -> QGroupBox:
        group = QGroupBox("Summary Metrics")
        layout = QFormLayout(group)

        self.metric_labels = {
            "annualized_return": QLabel("—"),
            "sharpe_ratio": QLabel("—"),
            "max_drawdown": QLabel("—"),
            "annualized_volatility": QLabel("—"),
            "total_return": QLabel("—"),
            "benchmark_total_return": QLabel("—"),
            "excess_total_return": QLabel("—"),
        }

        layout.addRow("Annualized Return", self.metric_labels["annualized_return"])
        layout.addRow("Sharpe", self.metric_labels["sharpe_ratio"])
        layout.addRow("Max Drawdown", self.metric_labels["max_drawdown"])
        layout.addRow("Volatility", self.metric_labels["annualized_volatility"])
        layout.addRow("Total Return", self.metric_labels["total_return"])
        layout.addRow("Benchmark Total Return", self.metric_labels["benchmark_total_return"])
        layout.addRow("Excess Total Return", self.metric_labels["excess_total_return"])
        return group

    def _build_weights_group(self) -> QGroupBox:
        group = QGroupBox("Current Weights")
        layout = QVBoxLayout(group)

        self.weights_table = QTableWidget(0, 2, group)
        self.weights_table.setHorizontalHeaderLabels(["Ticker", "Weight"])
        self.weights_table.horizontalHeader().setStretchLastSection(True)
        self.weights_table.verticalHeader().setVisible(False)
        self.weights_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.weights_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.weights_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout.addWidget(self.weights_table)
        return group

    def _load_checkpoints(self) -> None:
        try:
            descriptors = self.checkpoint_service.discover_checkpoints()
        except Exception as exc:
            self.status_bar.showMessage(f"Failed to discover checkpoints: {exc}")
            descriptors = []

        self.checkpoint_combo.blockSignals(True)
        self.checkpoint_combo.clear()
        for descriptor in descriptors:
            self.checkpoint_combo.addItem(descriptor.display_name, descriptor)
        self.checkpoint_combo.blockSignals(False)

        if descriptors:
            self._apply_descriptor(descriptors[0])
            self.run_button.setEnabled(True)
            self.status_bar.showMessage("Checkpoint discovered. Enter dates and run a replay.")
        else:
            self._current_descriptor = None
            self.run_button.setEnabled(False)
            self.status_bar.showMessage("No actor-critic checkpoints found. Run the ML pipeline first.")

    def _on_checkpoint_changed(self, index: int) -> None:
        descriptor = self.checkpoint_combo.itemData(index)
        if isinstance(descriptor, CheckpointDescriptor):
            self._apply_descriptor(descriptor)

    def _apply_descriptor(self, descriptor: CheckpointDescriptor) -> None:
        self._current_descriptor = descriptor
        self.ticker_input.setText(",".join(descriptor.tickers))

        benchmark_options = [descriptor.tickers[0], *[ticker for ticker in descriptor.tickers[1:] if ticker != descriptor.tickers[0]]]
        if "SPY" not in benchmark_options:
            benchmark_options.insert(0, "SPY")
        self.benchmark_combo.blockSignals(True)
        self.benchmark_combo.clear()
        self.benchmark_combo.addItems(benchmark_options)
        self.benchmark_combo.setCurrentText("SPY" if "SPY" in benchmark_options else benchmark_options[0])
        self.benchmark_combo.blockSignals(False)

    def _set_playback_enabled(self, enabled: bool) -> None:
        for widget in [
            self.play_button,
            self.pause_button,
            self.restart_button,
            self.step_back_button,
            self.step_forward_button,
            self.speed_combo,
            self.timeline_slider,
        ]:
            widget.setEnabled(enabled)

    def _start_replay_preparation(self) -> None:
        if self._current_descriptor is None:
            self._show_error("No saved model checkpoint is available. Train the policy first.")
            return

        try:
            tickers = parse_ticker_input(self.ticker_input.text())
        except ValueError as exc:
            self._show_error(str(exc))
            return

        benchmark_ticker = self.benchmark_combo.currentText().strip().upper()
        if not benchmark_ticker:
            self._show_error("Benchmark ticker cannot be empty.")
            return

        request = ReplayPreparationRequest(
            checkpoint_path=self._current_descriptor.path,
            portfolio_tickers=tickers,
            start_date=self.start_date_edit.date().toString("yyyy-MM-dd"),
            end_date=self.end_date_edit.date().toString("yyyy-MM-dd"),
            rebalance_frequency=self.rebalance_combo.currentText(),
            benchmark_ticker=benchmark_ticker,
            starting_capital=float(self.capital_input.value()),
        )
        self._prepare_replay_async(request)

    def _prepare_replay_async(self, request: ReplayPreparationRequest) -> None:
        self.run_button.setEnabled(False)
        self._set_playback_enabled(False)
        self.playback_timer.stop()

        self._worker_thread = QThread(self)
        self._worker = ReplayPreparationWorker(
            checkpoint_service=self.checkpoint_service,
            market_data_service=self.market_data_service,
            replay_service=self.replay_service,
            request=request,
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.status.connect(self.status_bar.showMessage)
        self._worker.finished.connect(self._on_replay_prepared)
        self._worker.error.connect(self._on_replay_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
            self._worker_thread = None
        self.run_button.setEnabled(self._current_descriptor is not None)

    def _on_replay_prepared(self, replay_result: object) -> None:
        if not isinstance(replay_result, PolicyReplayResult):
            self._show_error("Replay worker returned an unexpected result payload.")
            return
        self.replay_result = replay_result
        self.controller.set_frames(replay_result.frames)
        self._set_playback_enabled(True)

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, self.controller.max_index)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)

        self.equity_canvas.set_result(replay_result)
        self.allocation_history_canvas.set_result(replay_result)
        self.current_allocation_canvas.set_result(replay_result)
        self._update_summary_metrics(replay_result)
        self._render_current_frame(self.controller.current_frame())
        self.status_bar.showMessage("Replay prepared. Playback started automatically.")
        self._play()

    def _on_replay_error(self, message: str) -> None:
        self.playback_timer.stop()
        self._set_playback_enabled(False)
        self.status_bar.showMessage(message)
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "QuantShield Replay Error", message)

    def _on_speed_changed(self, label: str) -> None:
        if self.playback_timer.isActive():
            self.playback_timer.setInterval(PLAYBACK_SPEEDS_MS.get(label, 130))

    def _play(self) -> None:
        if not self.controller.has_frames:
            return
        interval = PLAYBACK_SPEEDS_MS.get(self.speed_combo.currentText(), 130)
        self.playback_timer.start(interval)
        self.status_bar.showMessage("Replay playback running.")

    def _pause(self) -> None:
        self.playback_timer.stop()
        self.status_bar.showMessage("Replay paused.")

    def _restart(self) -> None:
        if not self.controller.has_frames:
            return
        self.playback_timer.stop()
        frame = self.controller.restart()
        self._render_current_frame(frame)
        self._play()

    def _step_forward(self) -> None:
        if not self.controller.has_frames:
            return
        self.playback_timer.stop()
        frame = self.controller.step_forward()
        self._render_current_frame(frame)

    def _step_backward(self) -> None:
        if not self.controller.has_frames:
            return
        self.playback_timer.stop()
        frame = self.controller.step_backward()
        self._render_current_frame(frame)

    def _advance_playback(self) -> None:
        if not self.controller.has_frames:
            self.playback_timer.stop()
            return
        frame = self.controller.step_forward()
        self._render_current_frame(frame)
        if self.controller.current_index >= self.controller.max_index:
            self.playback_timer.stop()
            self.status_bar.showMessage("Replay finished. Use Restart or scrub the slider to inspect the backtest.")

    def _on_slider_changed(self, index: int) -> None:
        if self._slider_is_syncing or not self.controller.has_frames:
            return
        self.playback_timer.stop()
        frame = self.controller.scrub_to(index)
        self._render_current_frame(frame)
        self.status_bar.showMessage("Replay scrubbed to the selected timestep.")

    def _render_current_frame(self, frame) -> None:
        self._slider_is_syncing = True
        self.timeline_slider.setValue(frame.index)
        self._slider_is_syncing = False
        self.timeline_label.setText(f"Replay position: {frame.index + 1} / {self.controller.max_index + 1}")

        self.current_date_label.setText(frame.date.date().isoformat())
        self.current_portfolio_value_label.setText(f"${frame.portfolio_value:,.2f}")
        self.current_benchmark_value_label.setText(f"${frame.benchmark_value:,.2f}")
        self.current_return_label.setText(format_percent(frame.portfolio_return))
        self.current_benchmark_return_label.setText(format_percent(frame.benchmark_return))
        self.current_excess_label.setText(format_percent(frame.excess_return))
        self.current_turnover_label.setText(format_percent(frame.turnover))
        self.current_rebalance_label.setText("Yes" if frame.rebalanced else "No")

        self._update_weights_table(frame.weights)
        self.equity_canvas.update_frame(frame.index)
        self.allocation_history_canvas.update_frame(frame.index)
        self.current_allocation_canvas.update_frame(frame)

    def _update_summary_metrics(self, replay_result: PolicyReplayResult) -> None:
        metrics = replay_result.metrics
        self.metric_labels["annualized_return"].setText(format_percent(metrics["annualized_return"]))
        self.metric_labels["sharpe_ratio"].setText(f"{metrics['sharpe_ratio']:.3f}")
        self.metric_labels["max_drawdown"].setText(format_percent(metrics["max_drawdown"]))
        self.metric_labels["annualized_volatility"].setText(format_percent(metrics["annualized_volatility"]))
        self.metric_labels["total_return"].setText(format_percent(metrics["total_return"]))
        self.metric_labels["benchmark_total_return"].setText(format_percent(metrics["benchmark_total_return"]))
        self.metric_labels["excess_total_return"].setText(format_percent(metrics["excess_total_return"]))

    def _update_weights_table(self, weights: pd.Series) -> None:
        ordered = weights.sort_values(ascending=False)
        self.weights_table.setRowCount(len(ordered))
        for row_index, (ticker, value) in enumerate(ordered.items()):
            self.weights_table.setItem(row_index, 0, QTableWidgetItem(str(ticker)))
            self.weights_table.setItem(row_index, 1, QTableWidgetItem(format_percent(float(value))))


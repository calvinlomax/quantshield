"""Main desktop window for replaying QuantShield policy backtests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QDate, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from quantshield.replay_durations import (
    DEFAULT_REPLAY_DURATION_KEY,
    REPLAY_DURATION_PROFILES,
    duration_end_from_start,
    duration_start_from_end,
)
from quantshield.universe import CANONICAL_TOP_50_UNIVERSE, CANONICAL_TOP_ETF_UNIVERSE
from quantshield.utils import format_percent, generate_schedule
from quantshield_app.services import (
    CheckpointDescriptor,
    CheckpointService,
    MarketDataService,
    PortfolioLibraryService,
    ReplayService,
    TickerSearchService,
    TreasuryRateService,
)
from quantshield_app.services.replay_service import PolicyReplayResult
from quantshield_app.ui.charts import AllocationHistoryCanvas, CurrentAllocationCanvas, EquityCurveCanvas, TimestampHeatmapCanvas
from quantshield_app.ui.checkpoint_dialog import CheckpointSelectionDialog
from quantshield_app.ui.portfolio_dialogs import HoldingsBreakdownDialog, LoadPortfolioDialog, ReplaySummaryDialog, SavePortfolioDialog
from quantshield_app.ui.ticker_search_dialog import TickerSearchDialog
from quantshield_app.viewmodels import ReplayController


DEFAULT_START_DATE = QDate(2018, 1, 1)
REBALANCE_INTERVAL_OPTIONS: tuple[tuple[str, str, int], ...] = (
    ("1D", "B", 1),
    ("3D", "3B", 3),
    ("1W", "W-FRI", 5),
    ("2W", "2W-FRI", 10),
    ("1M", "ME", 21),
)
PLAYBACK_SPEEDS_MS: dict[str, int] = {
    "0.5x": 140,
    "1x": 70,
    "2x": 35,
    "4x": 18,
    "8x": 9,
}
ALLOWED_PORTFOLIO_SIZES: tuple[int, int] = (10, 50)
DEFAULT_MAX_PORTFOLIO_SIZE = 10
ASSET_GRID_COLUMNS = 5


@dataclass(slots=True)
class ReplayPreparationRequest:
    """Parameters handed from the UI thread to the background replay worker."""

    checkpoint_path: Path
    portfolio_tickers: list[str]
    start_date: str
    end_date: str | None
    rebalance_frequency: str
    rebalance_label: str
    rebalance_mode: str
    estimated_steps: int
    benchmark_ticker: str
    starting_capital: float
    force_refresh: bool = False


class ReplayPreparationWorker(QObject):
    """Prepare replay data off the UI thread."""

    finished = Signal(object)
    error = Signal(str)
    status = Signal(str)
    progress = Signal(int)

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
            self.progress.emit(10)
            self.status.emit("Loading saved actor-critic checkpoint...")
            checkpoint = self.checkpoint_service.load_checkpoint(self.request.checkpoint_path, device="cpu")

            self.progress.emit(40)
            self.status.emit("Downloading or loading cached market data from yfinance...")
            market_data = self.market_data_service.prepare_market_data(
                portfolio_tickers=self.request.portfolio_tickers,
                benchmark_ticker=self.request.benchmark_ticker,
                start_date=self.request.start_date,
                end_date=self.request.end_date,
                lookback_window=checkpoint.training_config.lookback_window,
                force_refresh=self.request.force_refresh,
            )

            self.progress.emit(75)
            self.status.emit("Running deterministic inference and building replay frames...")
            replay_result = self.replay_service.build_replay(
                checkpoint=checkpoint,
                market_data=market_data,
                rebalance_frequency=self.request.rebalance_frequency,
                starting_capital=self.request.starting_capital,
                rebalance_label=self.request.rebalance_label,
                rebalance_mode=self.request.rebalance_mode,
                estimated_steps=self.request.estimated_steps,
            )
            self.progress.emit(100)
            self.status.emit("Replay data prepared. Initializing charts...")
        except Exception as exc:  # pragma: no cover - exercised via integration/UI flow
            self.error.emit(str(exc))
            return

        self.finished.emit(replay_result)


class SpeedDisplayComboBox(QComboBox):
    """Playback speed selector with a richer closed-state label."""

    def closed_display_text(self) -> str:
        current = self.currentText().strip()
        return f"{current} Speed" if current else "Speed"

    def paintEvent(self, event: Any) -> None:  # pragma: no cover - UI paint only
        painter = QStylePainter(self)
        option = QStyleOptionComboBox()
        self.initStyleOption(option)
        option.currentText = self.closed_display_text()
        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, option)
        painter.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, option)


class ReplayLoadingDialog(QDialog):
    """Modal progress dialog shown while replay data is being prepared."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Loading Backtest")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.status_label = QLabel("Preparing replay...", self)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.log_view = QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(180)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.log_view)

    def append_log(self, message: str) -> None:
        if not message:
            return
        self.status_label.setText(message)
        self.log_view.appendPlainText(message)
        cursor = self.log_view.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.log_view.setTextCursor(cursor)

    def set_progress(self, value: int) -> None:
        self.progress_bar.setValue(max(0, min(int(value), 100)))


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
        self.setWindowTitle("QuantShield Desktop Backtest")
        self.resize(1420, 860)

        self.checkpoint_service = checkpoint_service or CheckpointService()
        self.market_data_service = market_data_service or MarketDataService()
        self.replay_service = replay_service or ReplayService(
            treasury_rate_service=TreasuryRateService(allow_online=True)
        )
        self.portfolio_library_service = PortfolioLibraryService()
        self.ticker_search_service = TickerSearchService()
        self.controller = ReplayController()
        self.replay_result: PolicyReplayResult | None = None
        self._current_descriptor: CheckpointDescriptor | None = None
        self._all_checkpoint_descriptors: list[CheckpointDescriptor] = []
        self._worker_thread: QThread | None = None
        self._worker: ReplayPreparationWorker | None = None
        self._loading_dialog: ReplayLoadingDialog | None = None
        self._close_after_worker = False
        self._pending_checkpoint_descriptor: CheckpointDescriptor | None = None
        self._slider_is_syncing = False
        self._syncing_dates = False
        self._splitter_initialized = False
        self._selected_tickers_state: list[str] = []
        self._current_interval_label = "—"
        self._current_interval_frequency = "B"
        self._current_interval_mode = "auto"
        self._current_estimated_steps = 0
        self._holdings_dialog: HoldingsBreakdownDialog | None = None
        self._max_portfolio_size = DEFAULT_MAX_PORTFOLIO_SIZE

        self.playback_timer = QTimer(self)
        self.playback_timer.timeout.connect(self._advance_playback)

        self._build_ui()
        self._sync_dates_from_duration(anchor="end")
        self._load_checkpoints()
        self._set_playback_enabled(False)

    def closeEvent(self, event: Any) -> None:  # pragma: no cover - UI cleanup
        self.playback_timer.stop()
        if self._worker_thread is not None and self._worker_thread.isRunning():
            self._close_after_worker = True
            self.run_button.setEnabled(False)
            self._show_loading_dialog()
            self._append_loading_log("Finishing replay preparation before closing the app...")
            event.ignore()
            return
        self._close_loading_dialog()
        super().closeEvent(event)

    def showEvent(self, event: Any) -> None:  # pragma: no cover - layout polish
        super().showEvent(event)
        if not self._splitter_initialized:
            screen_width = self.screen().availableGeometry().width() if self.screen() is not None else self.width()
            total_width = max(self.width(), 1200)
            left_width = min(int(screen_width * 0.45), int(total_width * 0.60))
            self.main_splitter.setSizes([left_width, max(total_width - left_width, 480)])
            self._splitter_initialized = True

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)

        self.status_bar = QStatusBar(self)
        self.status_bar.setSizeGripEnabled(False)
        self.footer_label = QLabel("(c) 2026 Calvin J. Lomax", self.status_bar)
        self.footer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_bar.addPermanentWidget(self.footer_label, 1)
        self.setStatusBar(self.status_bar)

        outer_layout = QVBoxLayout(root)
        outer_layout.setContentsMargins(10, 10, 10, 10)
        outer_layout.setSpacing(10)

        self.results_header_label = QLabel("Backtest Results: No model selected", root)
        self.results_header_label.setStyleSheet("font-size: 18px; font-weight: 700;")
        outer_layout.addWidget(self.results_header_label)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, root)
        outer_layout.addWidget(self.main_splitter)

        controls_scroll = QScrollArea(self.main_splitter)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setMinimumWidth(420)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        controls_panel = QWidget(controls_scroll)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(8)

        controls_layout.addWidget(self._build_run_group())
        state_controls_row = QWidget(controls_panel)
        state_controls_layout = QHBoxLayout(state_controls_row)
        state_controls_layout.setContentsMargins(0, 0, 0, 0)
        state_controls_layout.setSpacing(10)

        state_panel = QWidget(state_controls_row)
        state_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        state_panel_layout = QVBoxLayout(state_panel)
        state_panel_layout.setContentsMargins(0, 0, 0, 0)
        state_panel_layout.setSpacing(8)
        state_panel_layout.addWidget(self._build_current_state_group())
        self.holdings_button = QPushButton("Show Holdings", state_panel)
        self.holdings_button.clicked.connect(self._show_holdings_dialog)
        state_panel_layout.addWidget(self.holdings_button, stretch=0)
        self.summary_button = QPushButton("Show Summary", state_panel)
        self.summary_button.clicked.connect(self._show_summary_dialog)
        state_panel_layout.addWidget(self.summary_button, stretch=0)
        state_controls_layout.addWidget(state_panel, stretch=2)
        self.playback_group = self._build_playback_group()
        self.playback_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        state_controls_layout.addWidget(self.playback_group, stretch=3)
        controls_layout.addWidget(state_controls_row)
        controls_layout.addWidget(self._build_execution_details_group())
        controls_layout.addStretch(1)
        controls_scroll.setWidget(controls_panel)

        charts_panel = QWidget(self.main_splitter)
        charts_layout = QVBoxLayout(charts_panel)
        charts_layout.setContentsMargins(12, 12, 12, 12)
        charts_layout.setSpacing(10)

        self.equity_canvas = EquityCurveCanvas()
        self.allocation_history_canvas = AllocationHistoryCanvas()
        self.current_allocation_canvas = CurrentAllocationCanvas()
        self.timestamp_heatmap_canvas = TimestampHeatmapCanvas()

        chart_header = QWidget(charts_panel)
        chart_header_layout = QHBoxLayout(chart_header)
        chart_header_layout.setContentsMargins(0, 0, 0, 0)
        chart_header_layout.setSpacing(8)
        self.drawdown_toggle = QCheckBox("Show Drawdown", chart_header)
        self.drawdown_toggle.toggled.connect(self._on_drawdown_toggle_changed)
        self.equal_weight_toggle = QCheckBox("Show Equal Weight", chart_header)
        self.equal_weight_toggle.toggled.connect(self._on_equal_weight_toggle_changed)
        self.markowitz_toggle = QCheckBox("Show Markowitz", chart_header)
        self.markowitz_toggle.setToolTip("Compare the model to a rolling long-only mean-variance baseline on the same rebalance dates.")
        self.markowitz_toggle.toggled.connect(self._on_markowitz_toggle_changed)
        chart_header_layout.addStretch(1)
        chart_header_layout.addWidget(self.drawdown_toggle)
        chart_header_layout.addWidget(self.equal_weight_toggle)
        chart_header_layout.addWidget(self.markowitz_toggle)
        charts_layout.addWidget(chart_header)

        charts_layout.addWidget(self.equity_canvas, stretch=5)
        charts_layout.addWidget(self.allocation_history_canvas, stretch=4)

        bottom_charts = QWidget(charts_panel)
        bottom_layout = QHBoxLayout(bottom_charts)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        bottom_layout.addWidget(self.current_allocation_canvas, stretch=1)
        bottom_layout.addWidget(self.timestamp_heatmap_canvas, stretch=1)
        charts_layout.addWidget(bottom_charts, stretch=3)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal, charts_panel)
        self.timeline_slider.setRange(0, 0)
        self.timeline_slider.setTracking(True)
        self.timeline_slider.valueChanged.connect(self._on_slider_changed)
        charts_layout.addWidget(self.timeline_slider)

        self.timeline_label = QLabel("Replay position: 0 / 0")
        charts_layout.addWidget(self.timeline_label)

        self.main_splitter.addWidget(controls_scroll)
        self.main_splitter.addWidget(charts_panel)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 3)

    def _build_run_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        universe_group = QGroupBox("", group)
        universe_layout = QVBoxLayout(universe_group)
        universe_layout.setSpacing(6)
        self.active_checkpoint_name_label = QLabel("No model selected", universe_group)
        self.active_checkpoint_name_label.setWordWrap(True)
        self.active_checkpoint_name_label.setStyleSheet("font-weight: 600;")
        self.active_checkpoint_subtitle_label = QLabel("Select a trained model to begin backtesting.", universe_group)
        self.active_checkpoint_subtitle_label.setWordWrap(True)
        self.active_checkpoint_subtitle_label.setToolTip("Synthetic 10-slot inference means the model was trained against abstract asset slots and can generalize to arbitrary 10-name baskets.")
        self.select_model_button = QPushButton("Select Model", universe_group)
        self.select_model_button.clicked.connect(self._open_checkpoint_dialog)
        model_row = QHBoxLayout()
        model_text = QVBoxLayout()
        model_text.addWidget(QLabel("Active Model", universe_group))
        model_text.addWidget(self.active_checkpoint_name_label)
        model_text.addWidget(self.active_checkpoint_subtitle_label)
        model_row.addLayout(model_text, stretch=1)
        model_row.addWidget(self.select_model_button, stretch=0)
        universe_layout.addLayout(model_row)

        universe_layout.addWidget(QLabel("Selected Assets", universe_group))
        self.selected_assets_table = QTableWidget(0, ASSET_GRID_COLUMNS, universe_group)
        self.selected_assets_table.horizontalHeader().setVisible(False)
        self.selected_assets_table.verticalHeader().setVisible(False)
        self.selected_assets_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.selected_assets_table.verticalHeader().setDefaultSectionSize(24)
        self.selected_assets_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.selected_assets_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.selected_assets_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.selected_assets_table.setShowGrid(True)
        self._configure_selected_assets_table()
        universe_layout.addWidget(self.selected_assets_table)

        self.ticker_help_label = QLabel("")
        self._update_ticker_help_label()
        universe_layout.addWidget(self.ticker_help_label)
        layout.addWidget(universe_group)

        actions_group = QGroupBox("", group)
        actions_layout = QHBoxLayout(actions_group)
        self.edit_portfolio_button = QPushButton("Edit Portfolio", actions_group)
        self.random_ticker_button = QPushButton("Random Portfolio", actions_group)
        self.save_portfolio_button = QPushButton("Save Portfolio", actions_group)
        self.load_portfolio_button = QPushButton("Load Portfolio", actions_group)
        self.edit_portfolio_button.clicked.connect(self._open_portfolio_editor)
        self.random_ticker_button.clicked.connect(self._set_random_portfolio)
        self.save_portfolio_button.clicked.connect(self._save_portfolio)
        self.load_portfolio_button.clicked.connect(self._load_portfolio)
        actions_layout.addWidget(self.edit_portfolio_button)
        actions_layout.addWidget(self.random_ticker_button)
        actions_layout.addWidget(self.save_portfolio_button)
        actions_layout.addWidget(self.load_portfolio_button)
        layout.addWidget(actions_group)

        self.capital_input = QDoubleSpinBox(group)
        self.capital_input.setRange(100.0, 1_000_000_000.0)
        self.capital_input.setDecimals(2)
        self.capital_input.setSingleStep(10_000.0)
        self.capital_input.setValue(100_000.0)
        self.capital_input.setPrefix("$")

        self.benchmark_combo = QComboBox(group)
        self.benchmark_combo.setEditable(False)
        self.benchmark_combo.addItems(list(CANONICAL_TOP_ETF_UNIVERSE))
        self.benchmark_combo.setCurrentText("SPY")
        self.benchmark_combo.currentTextChanged.connect(self._refresh_selected_assets_table)

        self.duration_combo = QComboBox(group)
        for profile in REPLAY_DURATION_PROFILES:
            self.duration_combo.addItem(profile.label, profile.key)
        self.duration_combo.setCurrentText(DEFAULT_REPLAY_DURATION_KEY)
        self.duration_combo.currentTextChanged.connect(self._on_duration_changed)

        self.start_date_edit = QDateEdit(DEFAULT_START_DATE, group)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.dateChanged.connect(self._on_start_date_changed)

        self.end_date_edit = QDateEdit(QDate.currentDate(), group)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.dateChanged.connect(self._on_end_date_changed)

        capital_group = QGroupBox("", group)
        capital_layout = QGridLayout(capital_group)
        capital_layout.addWidget(QLabel("Starting Capital", capital_group), 0, 0)
        capital_layout.addWidget(QLabel("Benchmark", capital_group), 0, 1)
        capital_layout.addWidget(self.capital_input, 1, 0)
        capital_layout.addWidget(self.benchmark_combo, 1, 1)
        layout.addWidget(capital_group)

        self.rebalance_mode_combo = QComboBox(group)
        self.rebalance_mode_combo.addItem("Auto", "auto")
        self.rebalance_mode_combo.addItem("Manual", "manual")
        self.rebalance_mode_combo.setToolTip("Auto keeps a similar decision count to the 5-year baseline. Manual uses the interval you choose directly.")
        self.rebalance_mode_combo.currentIndexChanged.connect(self._update_interval_controls)

        self.rebalance_combo = QComboBox(group)
        for label, frequency, _approx_days in REBALANCE_INTERVAL_OPTIONS:
            self.rebalance_combo.addItem(label, frequency)
        self.rebalance_combo.setCurrentIndex(0)
        self.rebalance_combo.setToolTip("Manual rebalance interval. 1D is a business-day schedule.")
        self.rebalance_combo.currentIndexChanged.connect(self._update_interval_controls)

        self.effective_interval_label = QLabel("Effective Interval: —", group)
        self.effective_interval_label.setToolTip("Auto keeps a similar decision count as the 5-year baseline.")
        self.estimated_steps_label = QLabel("Estimated Steps: —", group)
        self.rebalance_warning_label = QLabel("", group)
        self.rebalance_warning_label.setWordWrap(True)

        time_group = QGroupBox("", group)
        time_layout = QGridLayout(time_group)
        time_layout.setHorizontalSpacing(8)
        time_layout.setVerticalSpacing(4)
        duration_label = QLabel("Duration", time_group)
        duration_label.setToolTip("Displayed replay horizon. Start and end dates stay synchronized to this duration unless you change them manually.")
        rebalance_label = QLabel("Rebalance Mode", time_group)
        rebalance_label.setToolTip("Business-day shorthand B means the trading-day calendar used by the backtest.")
        manual_label = QLabel("Interval", time_group)
        manual_label.setToolTip("Manual interval used when Rebalance Mode is Manual. B means business day.")
        time_layout.addWidget(duration_label, 0, 0)
        time_layout.addWidget(QLabel("Start Date", time_group), 0, 1)
        time_layout.addWidget(QLabel("End Date", time_group), 0, 2)
        time_layout.addWidget(rebalance_label, 2, 0)
        time_layout.addWidget(manual_label, 2, 1)
        time_layout.addWidget(QLabel("Effective Interval", time_group), 2, 2)
        time_layout.addWidget(self.duration_combo, 1, 0)
        time_layout.addWidget(self.start_date_edit, 1, 1)
        time_layout.addWidget(self.end_date_edit, 1, 2)
        time_layout.addWidget(self.rebalance_mode_combo, 3, 0)
        time_layout.addWidget(self.rebalance_combo, 3, 1)
        time_layout.addWidget(self.effective_interval_label, 3, 2)
        time_layout.addWidget(self.estimated_steps_label, 4, 0, 1, 2)
        time_layout.addWidget(self.rebalance_warning_label, 4, 2)
        layout.addWidget(time_group)

        self.run_button = QPushButton("Run Backtest", group)
        run_font = self.run_button.font()
        if run_font.pointSizeF() > 0:
            run_font.setPointSizeF(run_font.pointSizeF() * 1.2)
        self.run_button.setFont(run_font)
        self.run_button.setMinimumHeight(52)
        self.run_button.clicked.connect(self._start_replay_preparation)
        layout.addWidget(self.run_button)
        return group

    def _build_playback_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        self.play_pause_button = QPushButton("\u25B6", group)
        self.play_pause_button.setMinimumWidth(130)
        self.play_pause_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        play_font = self.play_pause_button.font()
        if play_font.pointSizeF() > 0:
            play_font.setPointSizeF(play_font.pointSizeF() * 2.5)
        self.play_pause_button.setFont(play_font)
        self.play_pause_button.clicked.connect(self._toggle_playback)
        self.restart_button = QPushButton("Reset Simulation", group)
        self.step_back_button = QPushButton("Previous Step", group)
        self.step_forward_button = QPushButton("Next Step", group)
        self.speed_combo = SpeedDisplayComboBox(group)
        self.speed_combo.addItems(list(PLAYBACK_SPEEDS_MS.keys()))
        self.speed_combo.setCurrentText("2x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        self.speed_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        for control in (self.restart_button, self.step_back_button, self.step_forward_button):
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.restart_button.clicked.connect(self._restart)
        self.step_back_button.clicked.connect(self._step_backward)
        self.step_forward_button.clicked.connect(self._step_forward)

        layout.addWidget(self.play_pause_button, 0, 0, 2, 1)
        layout.addWidget(self.restart_button, 0, 1)
        layout.addWidget(self.step_back_button, 0, 2)
        layout.addWidget(self.step_forward_button, 1, 1)
        layout.addWidget(self.speed_combo, 1, 2)
        return group

    def _build_current_state_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(4)

        self.current_date_label = QLabel("—")
        self.current_portfolio_value_label = QLabel("—")
        self.current_benchmark_value_label = QLabel("—")
        self.current_return_label = QLabel("—")
        self.current_benchmark_return_label = QLabel("—")
        self.current_excess_label = QLabel("—")

        left_rows = [("Date", self.current_date_label), ("Portfolio Value", self.current_portfolio_value_label), ("Benchmark Value", self.current_benchmark_value_label)]
        right_rows = [("Portfolio Return", self.current_return_label), ("Benchmark Return", self.current_benchmark_return_label), ("Excess Return", self.current_excess_label)]
        for row_index, (label_text, value_label) in enumerate(left_rows):
            layout.addWidget(QLabel(label_text, group), row_index, 0)
            layout.addWidget(value_label, row_index, 1)
        for row_index, (label_text, value_label) in enumerate(right_rows):
            layout.addWidget(QLabel(label_text, group), row_index, 2)
            layout.addWidget(value_label, row_index, 3)
        return group

    def _build_execution_details_group(self) -> QGroupBox:
        group = QGroupBox("")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(14)
        layout.setVerticalSpacing(4)

        self.current_turnover_label = QLabel("—")
        self.current_rebalance_label = QLabel("—")
        self.current_interval_label = QLabel("—")
        self.current_estimated_steps_label = QLabel("—")

        detail_rows = [
            ("Turnover", self.current_turnover_label),
            ("Rebalanced", self.current_rebalance_label),
            ("Effective Interval", self.current_interval_label),
            ("Estimated Steps", self.current_estimated_steps_label),
        ]
        for row_index, (label_text, value_label) in enumerate(detail_rows):
            layout.addWidget(QLabel(label_text, group), row_index // 2, (row_index % 2) * 2)
            layout.addWidget(value_label, row_index // 2, (row_index % 2) * 2 + 1)
        return group

    def _sync_dates_from_duration(self, *, anchor: str) -> None:
        if self._syncing_dates:
            return
        self._syncing_dates = True
        try:
            duration_key = self._active_duration_key()
            if anchor == "start":
                start_timestamp = pd.Timestamp(self.start_date_edit.date().toString("yyyy-MM-dd"))
                end_timestamp = duration_end_from_start(start_timestamp, duration_key)
                self.end_date_edit.setDate(QDate(end_timestamp.year, end_timestamp.month, end_timestamp.day))
            else:
                end_timestamp = pd.Timestamp(self.end_date_edit.date().toString("yyyy-MM-dd"))
                start_timestamp = duration_start_from_end(end_timestamp, duration_key)
                self.start_date_edit.setDate(QDate(start_timestamp.year, start_timestamp.month, start_timestamp.day))
        finally:
            self._syncing_dates = False
        self._update_interval_controls()

    def _selected_window_business_days(self) -> int:
        start_timestamp = pd.Timestamp(self.start_date_edit.date().toString("yyyy-MM-dd"))
        end_timestamp = pd.Timestamp(self.end_date_edit.date().toString("yyyy-MM-dd"))
        if end_timestamp < start_timestamp:
            start_timestamp, end_timestamp = end_timestamp, start_timestamp
        return max(len(pd.bdate_range(start_timestamp, end_timestamp)), 1)

    def _compute_effective_interval(self) -> tuple[str, str, int, str]:
        mode = str(self.rebalance_mode_combo.currentData() or "auto")
        start_timestamp = pd.Timestamp(self.start_date_edit.date().toString("yyyy-MM-dd"))
        end_timestamp = pd.Timestamp(self.end_date_edit.date().toString("yyyy-MM-dd"))
        if end_timestamp < start_timestamp:
            start_timestamp, end_timestamp = end_timestamp, start_timestamp
        duration_days = self._selected_window_business_days()

        if mode == "manual":
            label = self.rebalance_combo.currentText()
            frequency = str(self.rebalance_combo.currentData() or "B")
        else:
            baseline_index = pd.bdate_range("2020-01-01", periods=REPLAY_DURATION_PROFILES[-1].approximate_business_days)
            baseline_frequency = str(self.rebalance_combo.currentData() or "B")
            target_intervals = max(len(generate_schedule(baseline_index, baseline_frequency)), 1)
            desired_days = duration_days / target_intervals
            label, frequency, _ = min(
                REBALANCE_INTERVAL_OPTIONS,
                key=lambda option: abs(option[2] - desired_days),
            )
        live_index = pd.bdate_range(start_timestamp, end_timestamp)
        estimated_steps = max(len(generate_schedule(live_index, frequency)), 1)
        return label, frequency, estimated_steps, mode

    def _update_interval_controls(self) -> None:
        if not hasattr(self, "rebalance_combo"):
            return
        label, frequency, estimated_steps, mode = self._compute_effective_interval()
        self._current_interval_label = label
        self._current_interval_frequency = frequency
        self._current_interval_mode = mode
        self._current_estimated_steps = estimated_steps

        manual_mode = mode == "manual"
        self.rebalance_combo.setEnabled(manual_mode)
        self.effective_interval_label.setText(
            f"{label} ({'Auto-adjusted' if mode == 'auto' else 'Manual'})"
        )
        self.estimated_steps_label.setText(f"Estimated Steps: {estimated_steps}")
        warning = ""
        if manual_mode and estimated_steps < 20:
            warning = "Manual interval warning: fewer than 20 steps is probably too coarse."
        elif manual_mode and estimated_steps > 500:
            warning = "Manual interval warning: more than 500 steps is likely noisy."
        self.rebalance_warning_label.setText(warning)

    def _load_checkpoints(self) -> None:
        try:
            descriptors = self.checkpoint_service.discover_checkpoints()
        except Exception as exc:
            descriptors = []
            QMessageBox.warning(self, "QuantShield Models", f"Failed to discover models: {exc}")

        self._all_checkpoint_descriptors = descriptors
        if not descriptors:
            self._current_descriptor = None
            self.active_checkpoint_name_label.setText("No model selected")
            self.active_checkpoint_subtitle_label.setText("No saved backtest models were found.")
            self._update_results_header()
            self.run_button.setEnabled(False)
            return

        descriptor = None
        if self._pending_checkpoint_descriptor is not None:
            descriptor = self._match_descriptor(self._pending_checkpoint_descriptor.path)
            self._pending_checkpoint_descriptor = None
        if (
            descriptor is None
            and self._current_descriptor is not None
            and self._current_descriptor.duration_key == self._active_duration_key()
            and self._descriptor_matches_active_portfolio_size(self._current_descriptor)
        ):
            descriptor = self._match_descriptor(self._current_descriptor.path)
        if descriptor is None:
            descriptor = self._default_descriptor_for_duration(self._active_duration_key())
        if descriptor is not None:
            self._apply_descriptor(descriptor, preserve_portfolio=True)
        self.run_button.setEnabled(self._current_descriptor is not None)
        self._update_interval_controls()

    def _on_duration_changed(self, _: str) -> None:
        self._sync_dates_from_duration(anchor="end")
        self._load_checkpoints()

    def _on_start_date_changed(self, _: QDate) -> None:
        if self._syncing_dates:
            return
        self._sync_dates_from_duration(anchor="start")

    def _on_end_date_changed(self, _: QDate) -> None:
        if self._syncing_dates:
            return
        self._sync_dates_from_duration(anchor="end")

    def _active_duration_key(self) -> str:
        data = self.duration_combo.currentData()
        return str(data or DEFAULT_REPLAY_DURATION_KEY)

    def _default_descriptor_for_duration(self, duration_key: str) -> CheckpointDescriptor | None:
        for descriptor in self._all_checkpoint_descriptors:
            if descriptor.duration_key == duration_key and self._descriptor_matches_active_portfolio_size(descriptor):
                return descriptor
        for descriptor in self._all_checkpoint_descriptors:
            if self._descriptor_matches_active_portfolio_size(descriptor):
                return descriptor
        return self._all_checkpoint_descriptors[0] if self._all_checkpoint_descriptors else None

    def _descriptor_matches_active_portfolio_size(self, descriptor: CheckpointDescriptor) -> bool:
        if descriptor.uses_placeholder_tickers:
            return descriptor.slot_count == self._max_portfolio_size
        return descriptor.slot_count <= self._max_portfolio_size

    def _default_universe_for_current_size(self) -> list[str]:
        return list(CANONICAL_TOP_50_UNIVERSE if self._max_portfolio_size > 10 else CANONICAL_TOP_ETF_UNIVERSE)

    def _match_descriptor(self, path: Path) -> CheckpointDescriptor | None:
        for descriptor in self._all_checkpoint_descriptors:
            if descriptor.path == path:
                return descriptor
        return None

    def _apply_descriptor(self, descriptor: CheckpointDescriptor, *, preserve_portfolio: bool) -> None:
        self._current_descriptor = descriptor
        self.active_checkpoint_name_label.setText(descriptor.display_name)
        self.active_checkpoint_subtitle_label.setText(descriptor.display_subtitle)
        self.active_checkpoint_subtitle_label.setToolTip(descriptor.detail_text)
        self._update_results_header()
        if not preserve_portfolio or not self._selected_tickers_state:
            default_tickers = (
                descriptor.inference_default_tickers
                if len(descriptor.inference_default_tickers) >= 5
                else self._default_universe_for_current_size()
            )
            self._set_selected_tickers(default_tickers)

    def _set_selected_tickers(self, tickers: list[str]) -> None:
        normalized: list[str] = []
        seen: set[str] = set()
        for ticker in tickers:
            upper = str(ticker).strip().upper()
            if upper and upper not in seen:
                normalized.append(upper)
                seen.add(upper)
            if len(normalized) >= self._max_portfolio_size:
                break
        self._selected_tickers_state = sorted(normalized)
        self._refresh_selected_assets_table()

    def _configure_selected_assets_table(self) -> None:
        if not hasattr(self, "selected_assets_table"):
            return
        row_count = max(1, (self._max_portfolio_size + ASSET_GRID_COLUMNS - 1) // ASSET_GRID_COLUMNS)
        self.selected_assets_table.setRowCount(row_count)
        self.selected_assets_table.setColumnCount(ASSET_GRID_COLUMNS)
        self.selected_assets_table.setFixedHeight(row_count * 24 + 10)

    def _update_ticker_help_label(self) -> None:
        if hasattr(self, "ticker_help_label"):
            self.ticker_help_label.setText(f"Up to {self._max_portfolio_size} selected tickers are shown below.")

    def _set_max_portfolio_size(self, max_portfolio_size: int, *, refresh_checkpoints: bool = True) -> None:
        normalized_size = 50 if int(max_portfolio_size) > 10 else 10
        size_changed = normalized_size != self._max_portfolio_size
        self._max_portfolio_size = normalized_size
        self._configure_selected_assets_table()
        self._update_ticker_help_label()
        if len(self._selected_tickers_state) > self._max_portfolio_size:
            self._selected_tickers_state = self._selected_tickers_state[: self._max_portfolio_size]
        self._refresh_selected_assets_table()
        if refresh_checkpoints and size_changed and self._all_checkpoint_descriptors:
            self._load_checkpoints()

    def _refresh_selected_assets_table(self) -> None:
        if not hasattr(self, "selected_assets_table"):
            return
        benchmark = self.benchmark_combo.currentText().strip().upper() if hasattr(self, "benchmark_combo") else ""
        self.selected_assets_table.clearContents()
        self._configure_selected_assets_table()
        for slot_index in range(self._max_portfolio_size):
            row_index = slot_index // ASSET_GRID_COLUMNS
            column_index = slot_index % ASSET_GRID_COLUMNS
            ticker = self._selected_tickers_state[slot_index] if slot_index < len(self._selected_tickers_state) else ""
            item = QTableWidgetItem(ticker)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if ticker == benchmark and ticker:
                item.setBackground(QColor("#d9eefc"))
            self.selected_assets_table.setItem(row_index, column_index, item)

    def _update_results_header(self) -> None:
        descriptor_name = self._current_descriptor.display_name if self._current_descriptor is not None else "No model selected"
        self.results_header_label.setText(f"Backtest Results: {descriptor_name}")

    def _selected_tickers(self) -> list[str]:
        return list(self._selected_tickers_state)

    def _open_portfolio_editor(self) -> None:
        default_tickers = (
            self._current_descriptor.inference_default_tickers
            if self._current_descriptor is not None
            else self._default_universe_for_current_size()
        )
        dialog = TickerSearchDialog(
            search_service=self.ticker_search_service,
            portfolio_library_service=self.portfolio_library_service,
            selected_tickers=self._selected_tickers(),
            default_tickers=default_tickers,
            benchmark_ticker=self.benchmark_combo.currentText(),
            duration_key=self._active_duration_key(),
            start_date=self.start_date_edit.date().toString("yyyy-MM-dd"),
            end_date=self.end_date_edit.date().toString("yyyy-MM-dd"),
            max_count=self._max_portfolio_size,
            parent=self,
        )
        if dialog.exec():
            self._set_selected_tickers(dialog.selected_tickers)
        self._load_checkpoints()

    def _set_random_portfolio(self) -> None:
        try:
            tickers = self.ticker_search_service.random_portfolio(size=max(self._max_portfolio_size, 5))
        except Exception as exc:
            self._show_error(f"Could not generate a random baseline: {exc}")
            return
        self._set_selected_tickers(tickers)

    def _save_portfolio(self) -> None:
        tickers = self._selected_tickers()
        if len(tickers) < 5:
            self._open_portfolio_editor()
            return
        dialog = SavePortfolioDialog(
            tickers=tickers,
            details_text=(
                f"Benchmark: {self.benchmark_combo.currentText()} | Capital: ${self.capital_input.value():,.2f}\n"
                f"Window: {self.start_date_edit.date().toString('yyyy-MM-dd')} to {self.end_date_edit.date().toString('yyyy-MM-dd')}"
            ),
            parent=self,
        )
        if not dialog.exec():
            return
        try:
            self.portfolio_library_service.save_configuration(
                dialog.configuration_name,
                tickers,
                starting_capital=float(self.capital_input.value()),
                benchmark_ticker=self.benchmark_combo.currentText(),
                duration_key=self._active_duration_key(),
                start_date=self.start_date_edit.date().toString("yyyy-MM-dd"),
                end_date=self.end_date_edit.date().toString("yyyy-MM-dd"),
                rebalance_mode=self._current_interval_mode,
                rebalance_frequency=self._current_interval_frequency,
                model_path=self._current_descriptor.path.as_posix() if self._current_descriptor is not None else None,
                max_portfolio_size=self._max_portfolio_size,
            )
        except Exception as exc:
            self._show_error(f"Could not save the configuration: {exc}")
            return
        QMessageBox.information(self, "Save Configuration", f"Saved configuration '{dialog.configuration_name}'.")

    def _load_portfolio(self) -> None:
        try:
            configurations = self.portfolio_library_service.list_configurations()
        except Exception as exc:
            self._show_error(f"Could not read saved configurations: {exc}")
            return
        dialog = LoadPortfolioDialog(configurations=configurations, parent=self)
        if dialog.exec() and dialog.selected_configuration is not None:
            self._apply_saved_configuration(dialog.selected_configuration)

    def _apply_saved_configuration(self, configuration) -> None:
        configured_size = configuration.max_portfolio_size or (50 if len(configuration.tickers) > 10 else 10)
        self._set_max_portfolio_size(configured_size, refresh_checkpoints=False)
        self._set_selected_tickers(configuration.tickers)
        if configuration.starting_capital is not None:
            self.capital_input.setValue(configuration.starting_capital)
        if configuration.benchmark_ticker:
            self.benchmark_combo.setCurrentText(configuration.benchmark_ticker)
        if configuration.duration_key:
            self.duration_combo.setCurrentText(configuration.duration_key)
        if configuration.start_date:
            start = pd.Timestamp(configuration.start_date)
            self.start_date_edit.setDate(QDate(start.year, start.month, start.day))
        if configuration.end_date:
            end = pd.Timestamp(configuration.end_date)
            self.end_date_edit.setDate(QDate(end.year, end.month, end.day))
        if configuration.rebalance_mode:
            index = self.rebalance_mode_combo.findData(configuration.rebalance_mode)
            if index >= 0:
                self.rebalance_mode_combo.setCurrentIndex(index)
        if configuration.rebalance_frequency:
            interval_index = self.rebalance_combo.findData(configuration.rebalance_frequency)
            if interval_index >= 0:
                self.rebalance_combo.setCurrentIndex(interval_index)
        if configuration.model_path:
            matched = self._match_descriptor(Path(configuration.model_path))
            if matched is not None:
                self._apply_descriptor(matched, preserve_portfolio=True)
            else:
                self._load_checkpoints()
        else:
            self._load_checkpoints()
        self._update_interval_controls()

    def _open_checkpoint_dialog(self) -> None:
        self._load_checkpoints()
        dialog = CheckpointSelectionDialog(
            descriptors=self._all_checkpoint_descriptors,
            selected_descriptor=self._current_descriptor,
            active_duration_key=self._active_duration_key(),
            active_max_portfolio_size=self._max_portfolio_size,
            parent=self,
        )
        if not dialog.exec() or dialog.selected_descriptor is None:
            return
        descriptor = dialog.selected_descriptor
        self._set_max_portfolio_size(dialog.selected_max_portfolio_size, refresh_checkpoints=False)
        if descriptor.duration_key and descriptor.duration_key != self._active_duration_key():
            confirm = QMessageBox.question(
                self,
                "Select Model",
                (
                    f"'{descriptor.display_name}' belongs to the {descriptor.duration_key} suite.\n\n"
                    "Using it will also switch the active replay duration and linked date span. Continue?"
                ),
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            self._pending_checkpoint_descriptor = descriptor
            self.duration_combo.setCurrentText(descriptor.duration_key)
            return
        self._apply_descriptor(descriptor, preserve_portfolio=True)

    def _set_playback_enabled(self, enabled: bool) -> None:
        for widget in [
            self.play_pause_button,
            self.restart_button,
            self.step_back_button,
            self.step_forward_button,
            self.holdings_button,
            self.summary_button,
            self.speed_combo,
            self.timeline_slider,
        ]:
            widget.setEnabled(enabled)
        self._update_play_pause_button()

    def _start_replay_preparation(self) -> None:
        if self._current_descriptor is None:
            self._show_error("No saved model is available. Train or select a model first.")
            return

        tickers = self._selected_tickers()
        if len(tickers) < 5:
            self._open_portfolio_editor()
            return

        benchmark_ticker = self.benchmark_combo.currentText().strip().upper()
        if not benchmark_ticker:
            self._show_error("Benchmark ticker cannot be empty.")
            return

        interval_label, interval_frequency, estimated_steps, interval_mode = self._compute_effective_interval()

        request = ReplayPreparationRequest(
            checkpoint_path=self._current_descriptor.path,
            portfolio_tickers=tickers,
            start_date=self.start_date_edit.date().toString("yyyy-MM-dd"),
            end_date=self.end_date_edit.date().toString("yyyy-MM-dd"),
            rebalance_frequency=interval_frequency,
            rebalance_label=interval_label,
            rebalance_mode=interval_mode,
            estimated_steps=estimated_steps,
            benchmark_ticker=benchmark_ticker,
            starting_capital=float(self.capital_input.value()),
        )
        self._prepare_replay_async(request)

    def _prepare_replay_async(self, request: ReplayPreparationRequest) -> None:
        self.run_button.setEnabled(False)
        self._set_playback_enabled(False)
        self.playback_timer.stop()
        self._show_loading_dialog()

        self._worker_thread = QThread(self)
        self._worker = ReplayPreparationWorker(
            checkpoint_service=self.checkpoint_service,
            market_data_service=self.market_data_service,
            replay_service=self.replay_service,
            request=request,
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.status.connect(self._append_loading_log)
        self._worker.progress.connect(self._set_loading_progress)
        self._worker.finished.connect(self._on_replay_prepared)
        self._worker.error.connect(self._on_replay_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._cleanup_worker)
        self._worker_thread.start()

    def _cleanup_worker(self) -> None:
        should_close = self._close_after_worker
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._worker_thread is not None:
            self._worker_thread.deleteLater()
            self._worker_thread = None
        self.run_button.setEnabled(self._current_descriptor is not None)
        if should_close:
            self._close_after_worker = False
            QTimer.singleShot(0, self.close)

    def _show_loading_dialog(self) -> None:
        if self._loading_dialog is None:
            self._loading_dialog = ReplayLoadingDialog(self)
        self._loading_dialog.progress_bar.setValue(0)
        self._loading_dialog.log_view.clear()
        self._loading_dialog.status_label.setText("Preparing replay...")
        self._loading_dialog.show()
        self._loading_dialog.raise_()
        self._loading_dialog.activateWindow()

    def _close_loading_dialog(self) -> None:
        if self._loading_dialog is not None:
            self._loading_dialog.hide()

    def _append_loading_log(self, message: str) -> None:
        if self._loading_dialog is not None:
            self._loading_dialog.append_log(message)

    def _set_loading_progress(self, value: int) -> None:
        if self._loading_dialog is not None:
            self._loading_dialog.set_progress(value)

    def _on_replay_prepared(self, replay_result: object) -> None:
        if not isinstance(replay_result, PolicyReplayResult):
            self._close_loading_dialog()
            self._show_error("Replay worker returned an unexpected result payload.")
            return
        self._close_loading_dialog()
        self.replay_result = replay_result
        self.controller.set_frames(replay_result.frames)
        self._set_playback_enabled(True)

        self.timeline_slider.blockSignals(True)
        self.timeline_slider.setRange(0, self.controller.max_index)
        self.timeline_slider.setValue(0)
        self.timeline_slider.blockSignals(False)

        self.equity_canvas.set_result(replay_result)
        self.equity_canvas.set_drawdown_visible(self.drawdown_toggle.isChecked())
        self.equity_canvas.set_equal_weight_visible(self.equal_weight_toggle.isChecked())
        self.equity_canvas.set_markowitz_visible(self.markowitz_toggle.isChecked())
        self.allocation_history_canvas.set_result(replay_result)
        self.current_allocation_canvas.set_result(replay_result)
        self.timestamp_heatmap_canvas.set_result(replay_result)
        self._render_current_frame(self.controller.current_frame())
        self._play()

    def _on_replay_error(self, message: str) -> None:
        self.playback_timer.stop()
        self._set_playback_enabled(False)
        self._close_loading_dialog()
        self._show_error(message)

    def _show_error(self, message: str) -> None:
        self._close_loading_dialog()
        QMessageBox.critical(self, "QuantShield Replay Error", message)

    def _on_drawdown_toggle_changed(self, checked: bool) -> None:
        self.equity_canvas.set_drawdown_visible(checked)

    def _on_equal_weight_toggle_changed(self, checked: bool) -> None:
        self.equity_canvas.set_equal_weight_visible(checked)

    def _on_markowitz_toggle_changed(self, checked: bool) -> None:
        self.equity_canvas.set_markowitz_visible(checked)

    def _on_speed_changed(self, label: str) -> None:
        if self.playback_timer.isActive():
            self.playback_timer.setInterval(PLAYBACK_SPEEDS_MS.get(label, 130))

    def _toggle_playback(self) -> None:
        if self.playback_timer.isActive():
            self._pause()
        else:
            self._play()

    def _play(self) -> None:
        if not self.controller.has_frames:
            return
        interval = PLAYBACK_SPEEDS_MS.get(self.speed_combo.currentText(), 130)
        self.playback_timer.start(interval)
        self._update_play_pause_button()

    def _pause(self) -> None:
        self.playback_timer.stop()
        self._update_play_pause_button()

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
        self._update_play_pause_button()

    def _step_backward(self) -> None:
        if not self.controller.has_frames:
            return
        self.playback_timer.stop()
        frame = self.controller.step_backward()
        self._render_current_frame(frame)
        self._update_play_pause_button()

    def _show_summary_dialog(self) -> None:
        if self.replay_result is None:
            return
        dialog = ReplaySummaryDialog(replay_result=self.replay_result, parent=self)
        dialog.exec()

    def _show_holdings_dialog(self) -> None:
        if self.replay_result is None or not self.controller.has_frames:
            return
        if self._holdings_dialog is None:
            self._holdings_dialog = HoldingsBreakdownDialog(parent=self)
        self._update_holdings_dialog(self.controller.current_frame())
        self._holdings_dialog.show()
        self._holdings_dialog.raise_()
        self._holdings_dialog.activateWindow()

    def _update_play_pause_button(self) -> None:
        self.play_pause_button.setText("\u23F8" if self.playback_timer.isActive() else "\u25B6")

    def _advance_playback(self) -> None:
        if not self.controller.has_frames:
            self.playback_timer.stop()
            self._update_play_pause_button()
            return
        frame = self.controller.step_forward()
        self._render_current_frame(frame)
        if self.controller.current_index >= self.controller.max_index:
            self.playback_timer.stop()
            self._update_play_pause_button()

    def _on_slider_changed(self, index: int) -> None:
        if self._slider_is_syncing or not self.controller.has_frames:
            return
        self.playback_timer.stop()
        frame = self.controller.scrub_to(index)
        self._render_current_frame(frame)

    def _render_current_frame(self, frame) -> None:
        self._slider_is_syncing = True
        self.timeline_slider.setValue(frame.index)
        self._slider_is_syncing = False
        self.timeline_label.setText(f"Simulation step: {frame.index + 1} / {self.controller.max_index + 1}")

        self.current_date_label.setText(frame.date.date().isoformat())
        self.current_portfolio_value_label.setText(f"${frame.portfolio_value:,.2f}")
        self.current_benchmark_value_label.setText(f"${frame.benchmark_value:,.2f}")
        cumulative_portfolio_return = frame.portfolio_value / self.replay_result.starting_capital - 1.0 if self.replay_result is not None else 0.0
        cumulative_benchmark_return = frame.benchmark_value / self.replay_result.starting_capital - 1.0 if self.replay_result is not None else 0.0
        self.current_return_label.setText(format_percent(cumulative_portfolio_return))
        self.current_benchmark_return_label.setText(format_percent(cumulative_benchmark_return))
        self.current_excess_label.setText(format_percent(cumulative_portfolio_return - cumulative_benchmark_return))
        self.current_turnover_label.setText(format_percent(frame.turnover))
        self.current_rebalance_label.setText("Yes" if frame.rebalanced else "No")
        self.current_interval_label.setText(
            self.replay_result.rebalance_label if self.replay_result is not None else self._current_interval_label
        )
        self.current_estimated_steps_label.setText(
            str(self.replay_result.estimated_steps if self.replay_result is not None else self._current_estimated_steps)
        )

        self._update_holdings_dialog(frame)
        self.equity_canvas.update_frame(frame.index)
        self.allocation_history_canvas.update_frame(frame.index)
        self.current_allocation_canvas.update_frame(frame)
        self.timestamp_heatmap_canvas.update_frame(frame.index)

    def _holdings_rows(self, frame) -> list[list[tuple[str, float | int | str]]]:
        if self.replay_result is None:
            return []

        weights = frame.weights.sort_index()
        asset_returns = self.replay_result.asset_returns
        benchmark_returns = self.replay_result.benchmark_returns
        benchmark_path = benchmark_returns.loc[: frame.date]
        benchmark_total_return = float(np.prod(1.0 + benchmark_path.to_numpy(dtype=float)) - 1.0) if not benchmark_path.empty else 0.0
        benchmark_log_return = float(np.log1p(benchmark_path.to_numpy(dtype=float)).sum()) if not benchmark_path.empty else 0.0

        rows: list[list[tuple[str, float | int | str]]] = []
        for ticker, weight_value in weights.items():
            asset_path = asset_returns.loc[: frame.date, ticker]
            price_path = self.replay_result.prices.loc[: frame.date, ticker]
            current_return = float(asset_path.iloc[-1]) if not asset_path.empty else 0.0
            total_return = float(np.prod(1.0 + asset_path.to_numpy(dtype=float)) - 1.0) if not asset_path.empty else 0.0
            normalized_return = total_return - benchmark_total_return
            log_vs_benchmark = (
                float(np.log1p(asset_path.to_numpy(dtype=float)).sum()) - benchmark_log_return if not asset_path.empty else 0.0
            )
            share_count = int(frame.shares.get(ticker, 0))
            latest_price = float(price_path.iloc[-1]) if not price_path.empty else 0.0
            current_value = share_count * latest_price
            rows.append(
                [
                    (str(ticker), str(ticker)),
                    (format_percent(float(weight_value)), float(weight_value)),
                    (f"${current_value:,.2f}", float(current_value)),
                    (f"{share_count:,d}", share_count),
                    (format_percent(total_return), total_return),
                    (format_percent(current_return), current_return),
                    (format_percent(normalized_return), normalized_return),
                    (f"{log_vs_benchmark:.4f}", log_vs_benchmark),
                ]
            )
        return rows

    def _update_holdings_dialog(self, frame) -> None:
        if self._holdings_dialog is None:
            return
        self._holdings_dialog.update_rows(self._holdings_rows(frame))

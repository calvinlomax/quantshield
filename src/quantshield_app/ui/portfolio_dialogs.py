"""Dialogs for saving/loading configurations and inspecting backtest summaries."""

from __future__ import annotations

import math
from pathlib import Path
import re
import sys

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from PySide6.QtCore import QDate, QProcess, Qt, QTimer
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDateEdit,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from quantshield.metrics import drawdown_series
from quantshield.replay_durations import DEFAULT_REPLAY_DURATION_KEY, REPLAY_DURATION_PROFILES, get_replay_duration_profile
from quantshield.universe import CANONICAL_TOP_ETF_UNIVERSE
from quantshield.utils import infer_periods_per_year
from quantshield_app.services import PolicyReplayResult, SavedConfiguration


FIT_REBALANCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1D", "B"),
    ("3D", "3B"),
    ("1W", "W-FRI"),
    ("2W", "2W-FRI"),
    ("1M", "ME"),
)

DEFAULT_DURATION_FREQUENCIES = {
    "1mo": "B",
    "3mo": "3B",
    "6mo": "W-FRI",
    "1y": "2W-FRI",
    "3y": "ME",
    "5y": "ME",
}


def _format_percent(value: float) -> str:
    return f"{value:.2%}"


def _format_currency(value: float) -> str:
    return f"${value:,.2f}"


def _slugify_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "portfolio_fit"


class SaveConfigurationDialog(QDialog):
    """Prompt for a configuration name before saving."""

    def __init__(self, *, tickers: list[str], details_text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Configuration")
        self.resize(480, 230)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Save this configuration for reuse later:\n{', '.join(tickers)}"))
        if details_text:
            preview = QLabel(details_text, self)
            preview.setWordWrap(True)
            layout.addWidget(preview)

        form = QFormLayout()
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("Configuration name")
        form.addRow("Name", self.name_input)
        layout.addLayout(form)

        button_box = QDialogButtonBox(self)
        save_button = button_box.addButton("Save", QDialogButtonBox.ButtonRole.AcceptRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        save_button.clicked.connect(self._accept_if_valid)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

    @property
    def configuration_name(self) -> str:
        return self.name_input.text().strip()

    @property
    def portfolio_name(self) -> str:
        return self.configuration_name

    def _accept_if_valid(self) -> None:
        if not self.configuration_name:
            QMessageBox.warning(self, "Save Configuration", "Enter a name for the configuration before saving.")
            return
        self.accept()


class LoadConfigurationDialog(QDialog):
    """Browse and load a saved backtest configuration."""

    def __init__(
        self,
        *,
        configurations: list[SavedConfiguration],
        window_title: str = "Load Configuration",
        empty_text: str = "No saved configurations were found.",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(window_title)
        self.resize(760, 460)
        self.selected_configuration: SavedConfiguration | None = None
        self.selected_portfolio: SavedConfiguration | None = None
        self._empty_text = empty_text

        layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        self.configuration_list = QListWidget(self)
        self.configuration_list.setMinimumWidth(280)
        self.details = QPlainTextEdit(self)
        self.details.setReadOnly(True)
        content_layout.addWidget(self.configuration_list, stretch=2)
        content_layout.addWidget(self.details, stretch=3)
        layout.addLayout(content_layout, stretch=1)

        for configuration in configurations:
            item = QListWidgetItem(configuration.name)
            item.setData(256, configuration)
            self.configuration_list.addItem(item)

        self.configuration_list.currentItemChanged.connect(self._on_item_changed)
        self.configuration_list.itemDoubleClicked.connect(lambda _item: self._accept_if_valid())
        if self.configuration_list.count() > 0:
            self.configuration_list.setCurrentRow(0)
        else:
            self.details.setPlainText(self._empty_text)

        button_box = QDialogButtonBox(self)
        load_button = button_box.addButton("Load", QDialogButtonBox.ButtonRole.AcceptRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        load_button.clicked.connect(self._accept_if_valid)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

    def _on_item_changed(self, current: QListWidgetItem | None) -> None:
        configuration = current.data(256) if current is not None else None
        if isinstance(configuration, SavedConfiguration):
            self.selected_configuration = configuration
            self.selected_portfolio = configuration
            detail_lines = [
                f"Name: {configuration.name}",
                f"Tickers: {', '.join(configuration.tickers)}",
                f"Source: {configuration.source.title()}",
            ]
            if configuration.notes:
                detail_lines.append(f"Notes: {configuration.notes}")
            if configuration.starting_capital is not None:
                detail_lines.append(f"Starting Capital: {_format_currency(configuration.starting_capital)}")
            if configuration.benchmark_ticker:
                detail_lines.append(f"Benchmark: {configuration.benchmark_ticker}")
            if configuration.duration_key:
                detail_lines.append(f"Training Horizon: {configuration.duration_key}")
            if configuration.start_date and configuration.end_date:
                detail_lines.append(f"Window: {configuration.start_date} to {configuration.end_date}")
            if configuration.rebalance_mode and configuration.rebalance_frequency:
                detail_lines.append(
                    f"Rebalance: {configuration.rebalance_mode.title()} ({configuration.rebalance_frequency})"
                )
            if configuration.model_path:
                detail_lines.append(f"Model Path: {configuration.model_path}")
            self.details.setPlainText("\n".join(detail_lines))
        else:
            self.selected_configuration = None
            self.selected_portfolio = None
            self.details.setPlainText("No configuration selected.")

    def _accept_if_valid(self) -> None:
        if self.selected_configuration is None:
            QMessageBox.warning(self, "Load Configuration", "Choose a saved configuration before loading it.")
            return
        self.accept()


SavePortfolioDialog = SaveConfigurationDialog
LoadPortfolioDialog = LoadConfigurationDialog


class FitProgressCanvas(FigureCanvasQTAgg):
    """Live training diagnostics for a running portfolio fit."""

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(8.4, 6.2))
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.loss_axes = self.figure.add_subplot(311)
        self.return_axes = self.figure.add_subplot(312)
        self.sweep_axes = self.figure.add_subplot(313)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.96, hspace=0.52)
        self.clear()

    def clear(self) -> None:
        self.loss_axes.clear()
        self.return_axes.clear()
        self.sweep_axes.clear()
        self.loss_axes.text(0.5, 0.5, "Waiting for training history", ha="center", va="center")
        self.return_axes.text(0.5, 0.5, "Excess-return diagnostics will appear here", ha="center", va="center")
        self.sweep_axes.text(0.5, 0.5, "Completed candidate scores will appear here", ha="center", va="center")
        for axes in (self.loss_axes, self.return_axes, self.sweep_axes):
            axes.set_xticks([])
            axes.set_yticks([])
        self.draw_idle()

    def refresh_from_output_dir(self, output_dir: Path) -> None:
        self.loss_axes.clear()
        self.return_axes.clear()
        self.sweep_axes.clear()

        history_path, candidate_name = self._current_history_path(output_dir)
        if history_path is not None and history_path.exists():
            history = pd.read_csv(history_path)
            if "epoch" in history.columns:
                history = history.set_index("epoch")
            self._plot_history(history, candidate_name or history_path.parent.name)
        else:
            self.loss_axes.text(0.5, 0.5, "Waiting for candidate diagnostics", ha="center", va="center")
            self.return_axes.text(0.5, 0.5, "Training has not emitted history yet", ha="center", va="center")
            self.loss_axes.set_xticks([])
            self.loss_axes.set_yticks([])
            self.return_axes.set_xticks([])
            self.return_axes.set_yticks([])

        sweep_path = output_dir / "model_sweep.csv"
        if sweep_path.exists():
            try:
                sweep = pd.read_csv(sweep_path)
            except Exception:
                sweep = pd.DataFrame()
            if not sweep.empty and {"candidate", "all_composite_score"}.issubset(sweep.columns):
                self._plot_sweep(sweep)
            else:
                self.sweep_axes.text(0.5, 0.5, "Candidate sweep is still building", ha="center", va="center")
                self.sweep_axes.set_xticks([])
                self.sweep_axes.set_yticks([])
        else:
            self.sweep_axes.text(0.5, 0.5, "No completed candidates yet", ha="center", va="center")
            self.sweep_axes.set_xticks([])
            self.sweep_axes.set_yticks([])

        self.draw_idle()

    @staticmethod
    def _current_history_path(output_dir: Path) -> tuple[Path | None, str | None]:
        current_candidate_path = output_dir / "current_candidate.txt"
        candidate_name = current_candidate_path.read_text(encoding="utf-8").strip() if current_candidate_path.exists() else None
        if candidate_name:
            candidate_history = output_dir / "candidate_models" / candidate_name / "training_history.csv"
            if candidate_history.exists():
                return candidate_history, candidate_name

        candidates_root = output_dir / "candidate_models"
        histories = sorted(
            candidates_root.glob("*/training_history.csv"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if histories:
            return histories[0], histories[0].parent.name
        return None, candidate_name

    def _plot_history(self, history: pd.DataFrame, candidate_name: str) -> None:
        loss_columns = [
            column
            for column in ["train_total_loss", "train_actor_loss", "train_critic_loss", "train_bc_loss"]
            if column in history.columns
        ]
        if loss_columns:
            history[loss_columns].plot(ax=self.loss_axes, linewidth=1.2)
            self.loss_axes.legend(loc="best", fontsize=7)
        else:
            self.loss_axes.text(0.5, 0.5, "No loss columns available", ha="center", va="center")
        self.loss_axes.set_title(f"Training Diagnostics: {candidate_name}", fontsize=9)
        self.loss_axes.set_ylabel("Loss")
        self.loss_axes.grid(alpha=0.2)

        return_columns = [
            column
            for column in ["train_policy_excess_return", "validation_policy_excess_return"]
            if column in history.columns
        ]
        if return_columns:
            history[return_columns].plot(ax=self.return_axes, linewidth=1.3)
        if "train_demo_excess_return" in history.columns:
            self.return_axes.plot(history.index, history["train_demo_excess_return"], linestyle="--", linewidth=1.0, label="train_demo")
        if "validation_demo_excess_return" in history.columns:
            self.return_axes.plot(
                history.index,
                history["validation_demo_excess_return"],
                linestyle="--",
                linewidth=1.0,
                label="validation_demo",
            )
        if return_columns or "train_demo_excess_return" in history.columns or "validation_demo_excess_return" in history.columns:
            self.return_axes.legend(loc="best", fontsize=7)
        else:
            self.return_axes.text(0.5, 0.5, "No excess-return diagnostics available", ha="center", va="center")
        self.return_axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        self.return_axes.set_ylabel("Excess")
        self.return_axes.set_xlabel("Epoch")
        self.return_axes.grid(alpha=0.2)

        if "selected_checkpoint" in history.columns and history["selected_checkpoint"].any():
            selected_epoch = int(history.index[history["selected_checkpoint"].astype(bool)][0])
            self.loss_axes.axvline(selected_epoch, color="#2ca02c", linestyle=":", linewidth=1.0)
            self.return_axes.axvline(selected_epoch, color="#2ca02c", linestyle=":", linewidth=1.0)

    def _plot_sweep(self, sweep: pd.DataFrame) -> None:
        ordered = sweep.sort_values("all_composite_score", ascending=False)
        colors = [
            "#2ca02c" if bool(row.get("validation_beats_all_tickers", False)) else "#ff7f0e"
            for _, row in ordered.iterrows()
        ]
        bars = self.sweep_axes.bar(range(len(ordered)), ordered["all_composite_score"], color=colors)
        self.sweep_axes.set_title("Candidate Sweep", fontsize=9)
        self.sweep_axes.set_ylabel("Composite Score")
        self.sweep_axes.set_xticks(range(len(ordered)))
        self.sweep_axes.set_xticklabels(ordered["candidate"].tolist(), rotation=25, ha="right", fontsize=7)
        self.sweep_axes.grid(axis="y", alpha=0.2)
        for bar, value in zip(bars, ordered["all_composite_score"].tolist(), strict=True):
            self.sweep_axes.text(
                bar.get_x() + bar.get_width() / 2.0,
                float(value),
                f"{float(value):.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )


class FitModelDialog(QDialog):
    """Collect parameters and launch a portfolio-specific model-fit job."""

    def __init__(
        self,
        *,
        tickers: list[str],
        benchmark_ticker: str = "SPY",
        duration_key: str = DEFAULT_REPLAY_DURATION_KEY,
        start_date: str | None = None,
        end_date: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fit Model")
        self.resize(920, 700)
        self._tickers = list(tickers)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._append_stdout)
        self._process.readyReadStandardError.connect(self._append_stderr)
        self._process.finished.connect(self._on_finished)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._refresh_progress_visuals)

        root = Path(__file__).resolve().parents[3]
        self._root = root
        default_name = f"{duration_key}_{'_'.join(self._tickers[:3]).lower()}_fit" if self._tickers else "portfolio_fit"

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Fit a new model specifically for the currently selected portfolio. "
            "This writes a new experiment directory and does not replace the built-in models.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.name_input = QLineEdit(default_name, self)
        self.name_input.textChanged.connect(self._update_output_path_label)
        self.duration_combo = QComboBox(self)
        for profile in REPLAY_DURATION_PROFILES:
            self.duration_combo.addItem(profile.label, profile.key)
        duration_index = self.duration_combo.findData(duration_key)
        self.duration_combo.setCurrentIndex(duration_index if duration_index >= 0 else 0)
        self.duration_combo.currentIndexChanged.connect(self._sync_defaults_from_duration)

        self.start_date_edit = QDateEdit(self)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit = QDateEdit(self)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        if start_date:
            parsed = pd.Timestamp(start_date)
            self.start_date_edit.setDate(QDate(parsed.year, parsed.month, parsed.day))
        else:
            self.start_date_edit.setDate(QDate(2018, 1, 1))
        if end_date:
            parsed = pd.Timestamp(end_date)
            self.end_date_edit.setDate(QDate(parsed.year, parsed.month, parsed.day))
        else:
            self.end_date_edit.setDate(QDate.currentDate())

        self.benchmark_combo = QComboBox(self)
        benchmark_choices = list(CANONICAL_TOP_ETF_UNIVERSE)
        if benchmark_ticker and benchmark_ticker not in benchmark_choices:
            benchmark_choices.append(benchmark_ticker)
        self.benchmark_combo.addItems(benchmark_choices)
        self.benchmark_combo.setCurrentText(benchmark_ticker or "SPY")

        self.rebalance_combo = QComboBox(self)
        for label, frequency in FIT_REBALANCE_OPTIONS:
            self.rebalance_combo.addItem(label, frequency)

        self.candidate_mode_combo = QComboBox(self)
        self.candidate_mode_combo.addItem("Standard", "standard")
        self.candidate_mode_combo.addItem("Experimental", "experimental")
        self.candidate_mode_combo.addItem("Comprehensive", "comprehensive")
        self.candidate_mode_combo.setCurrentIndex(1)

        self.lookback_spin = QSpinBox(self)
        self.lookback_spin.setRange(5, 504)
        self.epochs_spin = QSpinBox(self)
        self.epochs_spin.setRange(16, 512)
        self.epochs_spin.setValue(56)
        self.batch_size_spin = QSpinBox(self)
        self.batch_size_spin.setRange(8, 256)
        self.batch_size_spin.setValue(64)

        self.device_combo = QComboBox(self)
        self.device_combo.addItem("Auto", "")
        self.device_combo.addItem("CPU", "cpu")

        form.addRow("Fit Name", self.name_input)
        form.addRow("Training Horizon", self.duration_combo)
        form.addRow("Training Start", self.start_date_edit)
        form.addRow("Training End", self.end_date_edit)
        form.addRow("Benchmark", self.benchmark_combo)
        form.addRow("Rebalance Interval", self.rebalance_combo)
        form.addRow("Candidate Set", self.candidate_mode_combo)
        form.addRow("Lookback Window", self.lookback_spin)
        form.addRow("Epoch Budget", self.epochs_spin)
        form.addRow("Batch Size", self.batch_size_spin)
        form.addRow("Device", self.device_combo)
        layout.addLayout(form)

        self.output_path_label = QLabel(self)
        self.output_path_label.setWordWrap(True)
        layout.addWidget(self.output_path_label)

        self.progress_status_label = QLabel("Training progress will appear here after the fit starts.", self)
        self.progress_status_label.setWordWrap(True)
        layout.addWidget(self.progress_status_label)

        self.progress_canvas = FitProgressCanvas(self)
        layout.addWidget(self.progress_canvas, stretch=2)

        self.log_view = QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        button_row = QHBoxLayout()
        self.start_button = QPushButton("Start Fit", self)
        self.start_button.clicked.connect(self._start_fit)
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.reject)
        button_row.addWidget(self.start_button)
        button_row.addStretch(1)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self._sync_defaults_from_duration()

    def _sync_defaults_from_duration(self) -> None:
        duration_key = str(self.duration_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY)
        profile = get_replay_duration_profile(duration_key)
        self.lookback_spin.setValue(profile.lookback_window)
        frequency = DEFAULT_DURATION_FREQUENCIES.get(duration_key, "ME")
        interval_index = self.rebalance_combo.findData(frequency)
        if interval_index >= 0:
            self.rebalance_combo.setCurrentIndex(interval_index)
        self._update_output_path_label()

    def _output_dir(self) -> Path:
        duration_key = str(self.duration_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY)
        return self._root / "outputs" / "portfolio_model_fits" / duration_key / _slugify_name(self.name_input.text())

    def _python_executable(self) -> str:
        venv_python = self._root / ".venv" / "bin" / "python"
        return venv_python.as_posix() if venv_python.exists() else sys.executable

    def _update_output_path_label(self) -> None:
        output_dir = self._output_dir()
        self.output_path_label.setText(f"Output Directory: {output_dir.as_posix()}")

    def _refresh_progress_visuals(self) -> None:
        output_dir = self._output_dir()
        self.progress_canvas.refresh_from_output_dir(output_dir)
        current_candidate_path = output_dir / "current_candidate.txt"
        if current_candidate_path.exists():
            candidate_name = current_candidate_path.read_text(encoding="utf-8").strip()
            self.progress_status_label.setText(f"Training candidate: {candidate_name}")
        else:
            sweep_path = output_dir / "model_sweep.csv"
            if sweep_path.exists():
                try:
                    sweep = pd.read_csv(sweep_path)
                except Exception:
                    sweep = pd.DataFrame()
                if not sweep.empty:
                    completed = len(sweep)
                    best_row = sweep.sort_values("all_composite_score", ascending=False).iloc[0]
                    self.progress_status_label.setText(
                        f"Completed candidates: {completed}. "
                        f"Current leader: {best_row['candidate']} ({float(best_row['all_composite_score']):.2f})."
                    )
                    return
            self.progress_status_label.setText("Waiting for training output…")

    def _start_fit(self) -> None:
        if len(self._tickers) < 5:
            QMessageBox.warning(self, "Fit Model", "Select at least 5 tickers before fitting a new model.")
            return
        fit_name = self.name_input.text().strip()
        if not fit_name:
            QMessageBox.warning(self, "Fit Model", "Enter a fit name before starting the model search.")
            return
        if self._process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Fit Model", "A model-fit job is already running.")
            return

        output_dir = self._output_dir()
        duration_key = str(self.duration_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY)
        benchmark = self.benchmark_combo.currentText().strip().upper()
        rebalance_frequency = str(self.rebalance_combo.currentData() or "ME")
        device = str(self.device_combo.currentData() or "")
        arguments = [
            str(self._root / "scripts" / "fit_portfolio_model.py"),
            "--name",
            fit_name,
            "--duration-key",
            duration_key,
            "--start-date",
            self.start_date_edit.date().toString("yyyy-MM-dd"),
            "--end-date",
            self.end_date_edit.date().toString("yyyy-MM-dd"),
            "--benchmark",
            benchmark,
            "--rebalance-frequency",
            rebalance_frequency,
            "--candidate-mode",
            str(self.candidate_mode_combo.currentData() or "experimental"),
            "--lookback-window",
            str(self.lookback_spin.value()),
            "--epochs",
            str(self.epochs_spin.value()),
            "--batch-size",
            str(self.batch_size_spin.value()),
            "--output-dir",
            output_dir.as_posix(),
            "--tickers",
            *self._tickers,
        ]
        if device:
            arguments.extend(["--device", device])

        self.log_view.clear()
        self.progress_canvas.clear()
        self.progress_status_label.setText("Launching fit job…")
        self.log_view.appendPlainText(f"Starting fit for: {', '.join(self._tickers)}")
        self.log_view.appendPlainText(f"Command: {self._python_executable()} {' '.join(arguments)}")
        self.start_button.setEnabled(False)
        self._process.setWorkingDirectory(self._root.as_posix())
        environment = self._process.processEnvironment()
        environment.insert("PYTHONPATH", (self._root / "src").as_posix())
        environment.insert("MPLCONFIGDIR", "/tmp/mpl")
        self._process.setProcessEnvironment(environment)
        self._process.start(self._python_executable(), arguments)
        self._progress_timer.start()
        self._refresh_progress_visuals()

    def _append_stdout(self) -> None:
        text = bytes(self._process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.log_view.appendPlainText(text.rstrip())

    def _append_stderr(self) -> None:
        text = bytes(self._process.readAllStandardError()).decode("utf-8", errors="replace")
        if text:
            self.log_view.appendPlainText(text.rstrip())

    def _on_finished(self, exit_code: int, _exit_status) -> None:
        self.start_button.setEnabled(True)
        self._progress_timer.stop()
        self._refresh_progress_visuals()
        if exit_code == 0:
            self.log_view.appendPlainText("")
            self.log_view.appendPlainText("Model fit completed successfully.")
            self.progress_status_label.setText("Model fit completed successfully.")
        else:
            self.log_view.appendPlainText("")
            self.log_view.appendPlainText(f"Model fit exited with code {exit_code}.")
            self.progress_status_label.setText(f"Model fit exited with code {exit_code}.")


class ReplaySummaryDialog(QDialog):
    """Comprehensive backtest summary shown on demand."""

    def __init__(self, *, replay_result: PolicyReplayResult, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Backtest Summary")
        self.resize(1240, 760)

        layout = QVBoxLayout(self)
        header = QLabel(self._header_text(replay_result), self)
        header.setWordWrap(True)
        layout.addWidget(header)

        summary_table = QTableWidget(
            len(replay_result.summary_table.index),
            len(replay_result.summary_table.columns) + 1,
            self,
        )
        summary_table.setHorizontalHeaderLabels(["Series", *[str(column) for column in replay_result.summary_table.columns]])
        summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        summary_table.verticalHeader().setVisible(False)
        for row_index, (series_name, row) in enumerate(replay_result.summary_table.iterrows()):
            summary_table.setItem(row_index, 0, QTableWidgetItem(str(series_name)))
            for column_index, value in enumerate(row.to_list(), start=1):
                numeric = float(value)
                text = f"{numeric:.4f}" if math.isfinite(numeric) else "—"
                summary_table.setItem(row_index, column_index, QTableWidgetItem(text))
        layout.addWidget(summary_table, stretch=1)

        highlights = QPlainTextEdit(self)
        highlights.setReadOnly(True)
        highlights.setPlainText(self._highlights_text(replay_result))
        layout.addWidget(highlights, stretch=2)

        button_box = QDialogButtonBox(self)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _header_text(replay_result: PolicyReplayResult) -> str:
        start_date = replay_result.frames[0].date.strftime("%Y-%m-%d")
        end_date = replay_result.frames[-1].date.strftime("%Y-%m-%d")
        frame_count = len(replay_result.frames)
        return (
            f"Model: {replay_result.checkpoint.path.as_posix()}\n"
            f"Timeframe: {start_date} to {end_date}\n"
            f"Portfolio: {', '.join(replay_result.requested_tickers)}\n"
            f"Benchmark: {replay_result.benchmark_ticker}\n"
            f"Starting Capital: {_format_currency(replay_result.starting_capital)}\n"
            f"Effective Interval: {replay_result.rebalance_label} ({replay_result.rebalance_mode.title()})\n"
            f"Frames: {frame_count} playback timestamps rendered across the backtest window."
        )

    @staticmethod
    def _highlights_text(replay_result: PolicyReplayResult) -> str:
        comparison = replay_result.comparison_returns.copy()
        periods_per_year = infer_periods_per_year(comparison.index, default=252)
        beta = ReplaySummaryDialog._beta(comparison["portfolio"], comparison["benchmark"])
        tracking_error = float(comparison["excess"].std(ddof=1) * np.sqrt(periods_per_year)) if len(comparison) > 1 else np.nan
        active_vs_equal_weight_tracking_error = (
            float(comparison["active_vs_equal_weight"].std(ddof=1) * np.sqrt(periods_per_year))
            if len(comparison) > 1 and "active_vs_equal_weight" in comparison.columns
            else np.nan
        )
        average_turnover = float(np.mean([frame.turnover for frame in replay_result.frames if frame.rebalanced])) if replay_result.frames else 0.0
        drawdown_duration, recovered = ReplaySummaryDialog._drawdown_duration(comparison["portfolio"])
        equal_weight_drawdown_duration, equal_weight_recovered = ReplaySummaryDialog._drawdown_duration(comparison["equal_weight"])
        interpretation = ReplaySummaryDialog._interpretation(replay_result, beta=beta, tracking_error=tracking_error)
        summary = replay_result.summary_table
        equal_weight_summary = summary.loc["equal_weight"] if "equal_weight" in summary.index else None

        lines = [
            "Timeframe",
            f"- Window: {replay_result.frames[0].date.strftime('%Y-%m-%d')} to {replay_result.frames[-1].date.strftime('%Y-%m-%d')}",
            f"- Estimated decisions: {replay_result.estimated_steps}",
            f"- Frames: {len(replay_result.frames)} daily playback observations; each frame is one rendered timestamp in the simulation.",
            "",
            "Assumptions",
            "- Returns are gross. Transaction costs, slippage, borrow costs, and taxes are not modeled, so gross and net are identical here.",
            ReplaySummaryDialog._risk_free_text(replay_result),
            "",
            "Portfolio Construction",
            f"- Structure: long-only, fully invested policy weights across {len(replay_result.requested_tickers)} selected assets.",
            "- Weighting: model-inferred continuous asset weights normalized at each rebalance date.",
            "- Equal-weight baseline: the same selected assets rebalanced evenly at each backtest decision date.",
            f"- Effective interval: {replay_result.rebalance_label} ({replay_result.rebalance_mode.title()}).",
            f"- Average turnover on rebalance dates: {_format_percent(average_turnover)}.",
            "",
            "Diagnostics",
            f"- Cumulative return: {_format_percent(replay_result.metrics['total_return'])}.",
            f"- Benchmark cumulative return: {_format_percent(replay_result.metrics['benchmark_total_return'])}.",
            f"- Excess return vs benchmark: {_format_percent(replay_result.metrics['excess_total_return'])}.",
            f"- Beta vs benchmark: {beta:.3f}" if math.isfinite(beta) else "- Beta vs benchmark: —",
            f"- Tracking error: {_format_percent(tracking_error)}" if math.isfinite(tracking_error) else "- Tracking error: —",
            f"- Annualized return: {_format_percent(replay_result.metrics['annualized_return'])}.",
            f"- Annualized volatility: {_format_percent(replay_result.metrics['annualized_volatility'])}.",
            f"- Sharpe ratio: {replay_result.metrics['sharpe_ratio']:.3f}.",
            f"- Max drawdown: {_format_percent(replay_result.metrics['max_drawdown'])}.",
            "",
            "Equal-Weight Baseline",
            f"- Equal-weight cumulative return: {_format_percent(replay_result.metrics['equal_weight_total_return'])}.",
            (
                f"- Equal-weight annualized return: {_format_percent(float(equal_weight_summary['annualized_return']))}."
                if equal_weight_summary is not None and math.isfinite(float(equal_weight_summary["annualized_return"]))
                else "- Equal-weight annualized return: —"
            ),
            (
                f"- Equal-weight annualized volatility: {_format_percent(float(equal_weight_summary['annualized_volatility']))}."
                if equal_weight_summary is not None and math.isfinite(float(equal_weight_summary["annualized_volatility"]))
                else "- Equal-weight annualized volatility: —"
            ),
            (
                f"- Equal-weight Sharpe ratio: {float(equal_weight_summary['sharpe_ratio']):.3f}."
                if equal_weight_summary is not None and math.isfinite(float(equal_weight_summary["sharpe_ratio"]))
                else "- Equal-weight Sharpe ratio: —"
            ),
            (
                f"- Equal-weight max drawdown: {_format_percent(float(equal_weight_summary['max_drawdown']))}."
                if equal_weight_summary is not None and math.isfinite(float(equal_weight_summary["max_drawdown"]))
                else "- Equal-weight max drawdown: —"
            ),
            f"- Active weighting vs equal-weight total return: {_format_percent(replay_result.metrics['active_vs_equal_weight_total_return'])}.",
            (
                f"- Active weighting vs equal-weight tracking error: {_format_percent(active_vs_equal_weight_tracking_error)}."
                if math.isfinite(active_vs_equal_weight_tracking_error)
                else "- Active weighting vs equal-weight tracking error: —"
            ),
            "",
            "Drawdown",
            f"- Longest drawdown duration: {drawdown_duration} trading days.",
            f"- Recovery status: {'Recovered by the end of the test window.' if recovered else 'Still below peak at the end of the window.'}",
            f"- Equal-weight drawdown duration: {equal_weight_drawdown_duration} trading days.",
            f"- Equal-weight recovery status: {'Recovered by the end of the test window.' if equal_weight_recovered else 'Still below peak at the end of the window.'}",
            "",
            "Interpretation",
            f"- {interpretation}",
            "",
            "Tag Guidance",
            "- Benchmark+ means the model showed positive excess-return evidence versus SPY.",
            "- Validated emphasizes stability on the held-out validation slice.",
            "- Exploratory indicates weaker statistical evidence and a higher chance of unstable behavior.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _risk_free_text(replay_result: PolicyReplayResult) -> str:
        assumption = replay_result.risk_free_assumption
        if assumption is None:
            return "- Risk-free rate assumption: 0.00%."
        rate_text = _format_percent(assumption.annual_rate)
        if assumption.fallback_used:
            return (
                f"- Risk-free rate assumption: {rate_text} annualized fallback matched to the "
                f"{assumption.maturity_label} because live Treasury data was unavailable."
            )
        as_of_text = assumption.as_of_date.strftime("%Y-%m-%d") if assumption.as_of_date is not None else "latest available date"
        return (
            f"- Risk-free rate assumption: {rate_text} annualized from the {assumption.maturity_label} "
            f"({as_of_text}; source: {assumption.source})."
        )

    @staticmethod
    def _beta(portfolio: pd.Series, benchmark: pd.Series) -> float:
        clean = pd.concat([portfolio.rename("portfolio"), benchmark.rename("benchmark")], axis=1).dropna()
        if len(clean) < 2:
            return np.nan
        variance = float(clean["benchmark"].var(ddof=1))
        if variance == 0.0:
            return np.nan
        covariance = float(clean["portfolio"].cov(clean["benchmark"]))
        return covariance / variance

    @staticmethod
    def _drawdown_duration(returns: pd.Series) -> tuple[int, bool]:
        drawdowns = drawdown_series(returns.dropna())
        if drawdowns.empty:
            return 0, True
        longest = 0
        current = 0
        for value in drawdowns:
            if value < 0.0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        recovered = bool(drawdowns.iloc[-1] >= 0.0)
        return longest, recovered

    @staticmethod
    def _interpretation(replay_result: PolicyReplayResult, *, beta: float, tracking_error: float) -> str:
        excess = replay_result.metrics["excess_total_return"]
        sharpe = replay_result.metrics["sharpe_ratio"]
        max_dd = abs(replay_result.metrics["max_drawdown"])
        if excess > 0.0 and sharpe >= 1.0 and max_dd <= 0.20:
            return "The strategy outperformed the benchmark with solid risk-adjusted returns and a contained drawdown profile."
        if excess > 0.0 and max_dd <= 0.30:
            return "The strategy beat the benchmark, but the path was uneven enough that drawdown tolerance still matters."
        if excess <= 0.0 and math.isfinite(tracking_error) and tracking_error < 0.05:
            return "The strategy tracked the benchmark fairly closely, but it did not add meaningful excess return over this window."
        if excess <= 0.0 and math.isfinite(beta) and beta > 1.1:
            return "The strategy underperformed while taking more benchmark-linked risk than the benchmark itself."
        return "The result is mixed: the model’s return profile needs to be weighed against benchmark-relative risk and drawdown tolerance."


class HoldingsBreakdownDialog(QDialog):
    """Popup window showing the current holdings breakdown table."""

    HEADERS = [
        "Ticker",
        "Weight",
        "Value ($)",
        "Shares",
        "Total Return",
        "Current Return",
        "Relative Return (Linear)",
        "Relative Return (Log)",
    ]

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Holdings Breakdown")
        self.resize(1120, 520)

        layout = QVBoxLayout(self)
        self.table = QTableWidget(0, len(self.HEADERS), self)
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, stretch=1)

        button_box = QDialogButtonBox(self)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

    def update_rows(self, rows: list[list[tuple[str, float | int | str]]]) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for row_index, row_values in enumerate(rows):
            for column_index, (display_value, numeric_value) in enumerate(row_values):
                item = QTableWidgetItem(display_value)
                item.setData(Qt.ItemDataRole.UserRole, numeric_value)
                if column_index == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                else:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row_index, column_index, item)
        self.table.setSortingEnabled(True)

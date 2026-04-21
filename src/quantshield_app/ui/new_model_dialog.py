"""Dialog for configuring, launching, monitoring, and saving new model-training runs."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PySide6.QtCore import QDate, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from quantshield.config import load_config
from quantshield.replay_durations import DEFAULT_REPLAY_DURATION_KEY, REPLAY_DURATION_PROFILES
from quantshield.universe import CANONICAL_TOP_50_UNIVERSE, CANONICAL_TOP_ETF_UNIVERSE
from quantshield_app.services import (
    ModelTrainingRequest,
    ModelTrainingService,
    PortfolioLibraryService,
    SavedConfiguration,
    parse_ticker_input,
)


REBALANCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1D", "B"),
    ("3D", "3B"),
    ("1W", "W-FRI"),
    ("2W", "2W-FRI"),
    ("1M", "ME"),
)
UNIVERSE_SOURCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Current Portfolio", "current_portfolio"),
    ("Saved Preset", "preset"),
    ("Canonical Universe", "canonical"),
    ("Broad Config Universe", "broad_config"),
    ("Manual Entry", "manual"),
    ("File Import", "file"),
)
BENCHMARK_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Ticker", "ticker"),
    ("Equal Weight", "equal_weight"),
    ("Markowitz", "markowitz"),
)


class LineMetricCanvas(FigureCanvasQTAgg):
    """Small reusable canvas for one time-series metric family."""

    def __init__(
        self,
        *,
        title: str,
        ylabel: str,
        empty_message: str,
        parent: QWidget | None = None,
        lock_y_on_first_plot: bool = False,
    ) -> None:
        figure = Figure(figsize=(5.4, 3.2))
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.10, right=0.98, bottom=0.20, top=0.88)
        self._title = title
        self._ylabel = ylabel
        self._empty_message = empty_message
        self._lock_y_on_first_plot = lock_y_on_first_plot
        self._locked_ylim: tuple[float, float] | None = None
        self.reset()

    def reset(self) -> None:
        self.axes.clear()
        self._locked_ylim = None
        self.axes.text(0.5, 0.5, self._empty_message, ha="center", va="center")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_title(self._title, fontsize=9)
        self.draw_idle()

    def plot_columns(self, history: pd.DataFrame, columns: list[str], *, zero_line: bool = False) -> None:
        self.axes.clear()
        plotted = False
        for column in columns:
            if column not in history.columns:
                continue
            self.axes.plot(history["epoch"], history[column], linewidth=1.2, label=column)
            plotted = True
        if not plotted:
            self.axes.text(0.5, 0.5, self._empty_message, ha="center", va="center")
            self.axes.set_xticks([])
            self.axes.set_yticks([])
        else:
            self.axes.set_xlabel("Epoch")
            self.axes.set_ylabel(self._ylabel)
            self.axes.grid(alpha=0.2)
            self.axes.legend(loc="best", fontsize=7)
            if zero_line:
                self.axes.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
            if self._lock_y_on_first_plot:
                if self._locked_ylim is None:
                    plotted_values = []
                    for column in columns:
                        if column not in history.columns:
                            continue
                        series = pd.to_numeric(history[column], errors="coerce").dropna()
                        if not series.empty:
                            plotted_values.extend(series.tolist())
                    if plotted_values:
                        lower = min(plotted_values)
                        upper = max(plotted_values)
                        span = max(upper - lower, 0.25)
                        padding = max(span * 0.18, max(abs(lower), abs(upper), 1.0) * 0.10, 0.15)
                        locked_lower = min(0.0, lower - padding)
                        locked_upper = max(upper + padding, 1.5)
                        self._locked_ylim = (locked_lower, locked_upper)
                if self._locked_ylim is not None:
                    self.axes.set_ylim(*self._locked_ylim)
        self.axes.set_title(self._title, fontsize=9)
        self.draw_idle()


class Training3DCanvas(FigureCanvasQTAgg):
    """Optional 3D visualization for real candidate sweeps."""

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(5.2, 4.2))
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.axes = self.figure.add_subplot(111, projection="3d")
        self.figure.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.92)
        self.reset()

    def reset(self, message: str = "No real optimization surface available yet") -> None:
        self.axes.clear()
        self.axes.text2D(0.08, 0.5, message, transform=self.axes.transAxes)
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_zticks([])
        self.draw_idle()

    def update_points(self, rows: list[dict[str, object]]) -> None:
        if not rows:
            self.reset()
            return
        frame = pd.DataFrame(rows)
        required = {"hidden_dim", "attention_layers", "all_composite_score"}
        if not required.issubset(frame.columns):
            self.reset("3D view unavailable for this run")
            return
        self.axes.clear()
        colors = frame["validation_composite_score"] if "validation_composite_score" in frame.columns else frame["all_composite_score"]
        scatter = self.axes.scatter(
            frame["hidden_dim"].astype(float),
            frame["attention_layers"].astype(float),
            frame["all_composite_score"].astype(float),
            c=colors.astype(float),
            cmap="viridis",
            s=50,
            depthshade=True,
        )
        self.axes.set_title("Reduced Candidate Sweep", fontsize=9)
        self.axes.set_xlabel("Hidden Dim")
        self.axes.set_ylabel("Layers")
        self.axes.set_zlabel("Composite")
        self.figure.colorbar(scatter, ax=self.axes, shrink=0.72, pad=0.08)
        self.draw_idle()


class GradientViewDialog(QDialog):
    """Standalone dialog for the optional 3D optimization surface."""

    def __init__(self, *, candidate_rows: list[dict[str, object]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gradient Descent / Candidate Surface")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        intro = QLabel("This view only appears when the training scripts emitted real sweep coordinates.", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.surface_canvas = Training3DCanvas(self)
        self.surface_canvas.update_points(candidate_rows)
        layout.addWidget(self.surface_canvas, stretch=1)
        button_row = QHBoxLayout()
        button_row.addStretch(1)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)


class TrainingMonitorDialog(QDialog):
    """Dedicated run-monitoring window shown while the training process is active."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Training Monitor")
        self.resize(1260, 860)
        self._running = True
        self._candidate_rows: list[dict[str, object]] = []
        self._candidate_total = 0
        self._completed_candidates = 0

        layout = QVBoxLayout(self)
        header_row = QHBoxLayout()
        self.state_label = QLabel("State: running", self)
        self.view_gradient_button = QPushButton("View Gradient Descent", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.close_button = QPushButton("Close", self)
        self.close_button.setEnabled(False)
        header_row.addWidget(self.state_label)
        header_row.addStretch(1)
        header_row.addWidget(self.view_gradient_button)
        header_row.addWidget(self.cancel_button)
        header_row.addWidget(self.close_button)
        layout.addLayout(header_row)

        self.candidate_status_label = QLabel("This run may train one or more candidate models.", self)
        self.candidate_status_label.setWordWrap(True)
        layout.addWidget(self.candidate_status_label)

        grid = QGridLayout()
        self.loss_canvas = LineMetricCanvas(
            title="Loss",
            ylabel="Loss",
            empty_message="Loss metrics will appear here",
            parent=self,
            lock_y_on_first_plot=True,
        )
        self.reward_canvas = LineMetricCanvas(
            title="Reward / Objective",
            ylabel="Reward",
            empty_message="Reward metrics will appear here",
            parent=self,
        )
        self.relative_canvas = LineMetricCanvas(
            title="Benchmark-Relative",
            ylabel="Relative",
            empty_message="Benchmark-relative metrics will appear here",
            parent=self,
        )
        self.cli_view = QPlainTextEdit(self)
        self.cli_view.setReadOnly(True)
        grid.addWidget(self.loss_canvas, 0, 0)
        grid.addWidget(self.reward_canvas, 0, 1)
        grid.addWidget(self.relative_canvas, 1, 0)
        grid.addWidget(self.cli_view, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        layout.addLayout(grid, stretch=1)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)
        layout.addWidget(self.progress_bar)

        self.view_gradient_button.clicked.connect(self._open_gradient_view)

    def set_running(self, running: bool, *, state_label: str | None = None) -> None:
        self._running = running
        if state_label is not None:
            self.state_label.setText(f"State: {state_label}")
        self.cancel_button.setEnabled(running)
        self.close_button.setEnabled(not running)
        if running:
            if self._candidate_total > 0:
                self.progress_bar.setRange(0, self._candidate_total)
                self.progress_bar.setValue(self._completed_candidates)
            else:
                self.progress_bar.setRange(0, 0)
        else:
            target = max(self._candidate_total, 1)
            self.progress_bar.setRange(0, target)
            self.progress_bar.setValue(target)

    def set_candidate_plan(
        self,
        *,
        total_candidates: int,
        active_candidate: str | None = None,
        completed_candidates: int | None = None,
    ) -> None:
        self._candidate_total = max(int(total_candidates), 0)
        if completed_candidates is not None:
            self._completed_candidates = max(int(completed_candidates), 0)
        active_text = active_candidate or "waiting to start"
        if self._candidate_total > 1:
            self.candidate_status_label.setText(
                f"This run is training {self._candidate_total} separate candidate models sequentially. "
                f"Completed: {self._completed_candidates}/{self._candidate_total}. Active candidate: {active_text}."
            )
        elif self._candidate_total == 1:
            self.candidate_status_label.setText(
                f"This run is training 1 candidate model. Active candidate: {active_text}."
            )
        else:
            self.candidate_status_label.setText("This run may train one or more candidate models.")
        if self._running:
            if self._candidate_total > 0:
                self.progress_bar.setRange(0, self._candidate_total)
                self.progress_bar.setValue(self._completed_candidates)
            else:
                self.progress_bar.setRange(0, 0)

    def set_candidate_rows(self, rows: list[dict[str, object]]) -> None:
        self._candidate_rows = list(rows)
        self._completed_candidates = len(self._candidate_rows)
        if self._candidate_total > 0 and self._running:
            self.progress_bar.setRange(0, self._candidate_total)
            self.progress_bar.setValue(min(self._completed_candidates, self._candidate_total))

    def append_log(self, line: str) -> None:
        self.cli_view.appendPlainText(line)

    def reset_metric_views(self) -> None:
        self.loss_canvas.reset()
        self.reward_canvas.reset()
        self.relative_canvas.reset()

    def update_history(self, history: pd.DataFrame) -> None:
        self.loss_canvas.plot_columns(
            history,
            ["train_total_loss", "train_actor_loss", "train_critic_loss", "train_bc_loss", "validation_loss"],
        )
        self.reward_canvas.plot_columns(
            history,
            ["train_policy_training_reward", "validation_policy_training_reward", "train_policy_excess_return", "validation_policy_excess_return"],
        )
        self.relative_canvas.plot_columns(
            history,
            [
                "validation_policy_excess_vs_equal_weight",
                "validation_policy_excess_vs_restricted_random",
                "validation_policy_excess_vs_markowitz",
                "train_policy_excess_vs_equal_weight",
                "train_policy_excess_vs_restricted_random",
                "train_policy_excess_vs_markowitz",
            ],
            zero_line=True,
        )

    def _open_gradient_view(self) -> None:
        dialog = GradientViewDialog(candidate_rows=self._candidate_rows, parent=self)
        dialog.exec()

    def reject(self) -> None:
        if self._running:
            return
        super().reject()


class GraphResultsDialog(QDialog):
    """Read-only dialog showing the completed run's full graph set and CLI output."""

    def __init__(self, *, history: pd.DataFrame, candidate_rows: list[dict[str, object]], cli_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Training Graphs")
        self.resize(1260, 860)

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        self.loss_canvas = LineMetricCanvas(
            title="Loss",
            ylabel="Loss",
            empty_message="No loss metrics available",
            parent=self,
            lock_y_on_first_plot=True,
        )
        self.reward_canvas = LineMetricCanvas(title="Reward / Objective", ylabel="Reward", empty_message="No reward metrics available", parent=self)
        self.relative_canvas = LineMetricCanvas(
            title="Benchmark-Relative",
            ylabel="Relative",
            empty_message="No benchmark-relative metrics available",
            parent=self,
        )
        self.cli_view = QPlainTextEdit(self)
        self.cli_view.setReadOnly(True)
        self.cli_view.setPlainText(cli_text)
        grid.addWidget(self.loss_canvas, 0, 0)
        grid.addWidget(self.reward_canvas, 0, 1)
        grid.addWidget(self.relative_canvas, 1, 0)
        grid.addWidget(self.cli_view, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        layout.addLayout(grid, stretch=1)

        actions = QHBoxLayout()
        view_gradient = QPushButton("View Gradient Descent", self)
        close_button = QPushButton("Close", self)
        actions.addStretch(1)
        actions.addWidget(view_gradient)
        actions.addWidget(close_button)
        layout.addLayout(actions)

        if not history.empty:
            self.loss_canvas.plot_columns(
                history,
                ["train_total_loss", "train_actor_loss", "train_critic_loss", "train_bc_loss", "validation_loss"],
            )
            self.reward_canvas.plot_columns(
                history,
                ["train_policy_training_reward", "validation_policy_training_reward", "train_policy_excess_return", "validation_policy_excess_return"],
            )
            self.relative_canvas.plot_columns(
                history,
                [
                    "validation_policy_excess_vs_equal_weight",
                    "validation_policy_excess_vs_restricted_random",
                    "validation_policy_excess_vs_markowitz",
                    "train_policy_excess_vs_equal_weight",
                    "train_policy_excess_vs_restricted_random",
                    "train_policy_excess_vs_markowitz",
                ],
                zero_line=True,
            )

        view_gradient.clicked.connect(lambda: GradientViewDialog(candidate_rows=candidate_rows, parent=self).exec())
        close_button.clicked.connect(self.accept)


class AdvancedOptionsDialog(QDialog):
    """Popup for advanced hyperparameters and launch preview."""

    def __init__(self, *, initial_values: dict[str, object], preview_callback, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Options")
        self.resize(980, 760)
        self._preview_callback = preview_callback

        layout = QVBoxLayout(self)
        groups_layout = QHBoxLayout()

        advanced_group = QGroupBox("Advanced Hyperparameters", self)
        advanced_form = QFormLayout(advanced_group)
        self.hidden_dim_spin = QSpinBox(advanced_group)
        self.hidden_dim_spin.setRange(32, 1024)
        self.attention_heads_spin = QSpinBox(advanced_group)
        self.attention_heads_spin.setRange(1, 32)
        self.attention_layers_spin = QSpinBox(advanced_group)
        self.attention_layers_spin.setRange(1, 16)
        self.dropout_spin = QDoubleSpinBox(advanced_group)
        self.dropout_spin.setDecimals(3)
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.01)
        self.weight_decay_spin = QDoubleSpinBox(advanced_group)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setRange(0.0, 1.0)
        self.weight_decay_spin.setSingleStep(0.0001)
        self.actor_bc_spin = QDoubleSpinBox(advanced_group)
        self.actor_bc_spin.setDecimals(4)
        self.actor_bc_spin.setRange(0.0, 25.0)
        self.entropy_spin = QDoubleSpinBox(advanced_group)
        self.entropy_spin.setDecimals(6)
        self.entropy_spin.setRange(0.0, 1.0)
        self.validation_fraction_spin = QDoubleSpinBox(advanced_group)
        self.validation_fraction_spin.setDecimals(2)
        self.validation_fraction_spin.setRange(0.05, 0.50)
        self.validation_fraction_spin.setSingleStep(0.05)
        self.optimizer_combo = QComboBox(advanced_group)
        self.optimizer_combo.addItem("AdamW", "adamw")
        self.optimizer_combo.addItem("Adam", "adam")
        self.checkpoint_frequency_spin = QSpinBox(advanced_group)
        self.checkpoint_frequency_spin.setRange(0, 512)
        self.early_stopping_spin = QSpinBox(advanced_group)
        self.early_stopping_spin.setRange(0, 256)
        self.reward_benchmark_spin = QDoubleSpinBox(advanced_group)
        self.reward_benchmark_spin.setDecimals(3)
        self.reward_benchmark_spin.setRange(0.0, 5.0)
        self.reward_equal_weight_spin = QDoubleSpinBox(advanced_group)
        self.reward_equal_weight_spin.setDecimals(3)
        self.reward_equal_weight_spin.setRange(0.0, 5.0)
        self.reward_random_spin = QDoubleSpinBox(advanced_group)
        self.reward_random_spin.setDecimals(3)
        self.reward_random_spin.setRange(0.0, 5.0)
        self.reward_markowitz_spin = QDoubleSpinBox(advanced_group)
        self.reward_markowitz_spin.setDecimals(3)
        self.reward_markowitz_spin.setRange(0.0, 5.0)
        self.reward_raw_spin = QDoubleSpinBox(advanced_group)
        self.reward_raw_spin.setDecimals(3)
        self.reward_raw_spin.setRange(0.0, 5.0)
        advanced_form.addRow("Hidden Dim", self.hidden_dim_spin)
        advanced_form.addRow("Attention Heads", self.attention_heads_spin)
        advanced_form.addRow("Attention Layers", self.attention_layers_spin)
        advanced_form.addRow("Dropout", self.dropout_spin)
        advanced_form.addRow("Weight Decay", self.weight_decay_spin)
        advanced_form.addRow("Actor BC Weight", self.actor_bc_spin)
        advanced_form.addRow("Entropy Weight", self.entropy_spin)
        advanced_form.addRow("Validation Split", self.validation_fraction_spin)
        advanced_form.addRow("Optimizer", self.optimizer_combo)
        advanced_form.addRow("Checkpoint Every", self.checkpoint_frequency_spin)
        advanced_form.addRow("Early Stop Patience", self.early_stopping_spin)
        advanced_form.addRow("Reward vs Benchmark", self.reward_benchmark_spin)
        advanced_form.addRow("Reward vs Equal Weight", self.reward_equal_weight_spin)
        advanced_form.addRow("Reward vs Random", self.reward_random_spin)
        advanced_form.addRow("Reward vs Markowitz", self.reward_markowitz_spin)
        advanced_form.addRow("Reward Raw", self.reward_raw_spin)
        groups_layout.addWidget(advanced_group, stretch=1)

        preview_group = QGroupBox("Launch Preview", self)
        preview_layout = QVBoxLayout(preview_group)
        self.preview_summary_label = QLabel(preview_group)
        self.preview_summary_label.setWordWrap(True)
        self.preview_output_path_label = QLabel(preview_group)
        self.preview_output_path_label.setWordWrap(True)
        self.resolved_values_preview = QPlainTextEdit(preview_group)
        self.resolved_values_preview.setReadOnly(True)
        self.command_preview = QPlainTextEdit(preview_group)
        self.command_preview.setReadOnly(True)
        preview_layout.addWidget(self.preview_summary_label)
        preview_layout.addWidget(self.preview_output_path_label)
        preview_layout.addWidget(QLabel("Resolved Hyperparameters", preview_group))
        preview_layout.addWidget(self.resolved_values_preview, stretch=1)
        preview_layout.addWidget(QLabel("Exact Command", preview_group))
        preview_layout.addWidget(self.command_preview, stretch=1)
        groups_layout.addWidget(preview_group, stretch=1)

        layout.addLayout(groups_layout, stretch=1)

        buttons = QHBoxLayout()
        apply_button = QPushButton("Apply", self)
        close_button = QPushButton("Close", self)
        buttons.addStretch(1)
        buttons.addWidget(apply_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        self._set_values(initial_values)
        for widget in (
            self.hidden_dim_spin,
            self.attention_heads_spin,
            self.attention_layers_spin,
            self.dropout_spin,
            self.weight_decay_spin,
            self.actor_bc_spin,
            self.entropy_spin,
            self.validation_fraction_spin,
            self.checkpoint_frequency_spin,
            self.early_stopping_spin,
            self.reward_benchmark_spin,
            self.reward_equal_weight_spin,
            self.reward_random_spin,
            self.reward_markowitz_spin,
            self.reward_raw_spin,
        ):
            widget.valueChanged.connect(self._refresh_preview)
        self.optimizer_combo.currentIndexChanged.connect(self._refresh_preview)
        apply_button.clicked.connect(self.accept)
        close_button.clicked.connect(self.reject)
        self._refresh_preview()

    def values(self) -> dict[str, object]:
        return {
            "hidden_dim": int(self.hidden_dim_spin.value()),
            "attention_heads": int(self.attention_heads_spin.value()),
            "attention_layers": int(self.attention_layers_spin.value()),
            "dropout": float(self.dropout_spin.value()),
            "weight_decay": float(self.weight_decay_spin.value()),
            "actor_bc_weight": float(self.actor_bc_spin.value()),
            "entropy_weight": float(self.entropy_spin.value()),
            "validation_fraction": float(self.validation_fraction_spin.value()),
            "optimizer": str(self.optimizer_combo.currentData() or "adamw"),
            "checkpoint_frequency": int(self.checkpoint_frequency_spin.value()),
            "early_stopping_patience": int(self.early_stopping_spin.value()),
            "reward_weight_vs_benchmark": float(self.reward_benchmark_spin.value()),
            "reward_weight_vs_equal_weight": float(self.reward_equal_weight_spin.value()),
            "reward_weight_vs_restricted_random": float(self.reward_random_spin.value()),
            "reward_weight_vs_markowitz": float(self.reward_markowitz_spin.value()),
            "reward_weight_raw": float(self.reward_raw_spin.value()),
        }

    def _set_values(self, values: dict[str, object]) -> None:
        self.hidden_dim_spin.setValue(int(values["hidden_dim"]))
        self.attention_heads_spin.setValue(int(values["attention_heads"]))
        self.attention_layers_spin.setValue(int(values["attention_layers"]))
        self.dropout_spin.setValue(float(values["dropout"]))
        self.weight_decay_spin.setValue(float(values["weight_decay"]))
        self.actor_bc_spin.setValue(float(values["actor_bc_weight"]))
        self.entropy_spin.setValue(float(values["entropy_weight"]))
        self.validation_fraction_spin.setValue(float(values["validation_fraction"]))
        optimizer_index = max(0, self.optimizer_combo.findData(str(values["optimizer"])))
        self.optimizer_combo.setCurrentIndex(optimizer_index)
        self.checkpoint_frequency_spin.setValue(int(values["checkpoint_frequency"]))
        self.early_stopping_spin.setValue(int(values["early_stopping_patience"]))
        self.reward_benchmark_spin.setValue(float(values["reward_weight_vs_benchmark"]))
        self.reward_equal_weight_spin.setValue(float(values["reward_weight_vs_equal_weight"]))
        self.reward_random_spin.setValue(float(values["reward_weight_vs_restricted_random"]))
        self.reward_markowitz_spin.setValue(float(values["reward_weight_vs_markowitz"]))
        self.reward_raw_spin.setValue(float(values["reward_weight_raw"]))

    def _refresh_preview(self) -> None:
        summary, output_path, resolved_lines, command_text = self._preview_callback(self.values())
        self.preview_summary_label.setText(summary)
        self.preview_output_path_label.setText(output_path)
        self.resolved_values_preview.setPlainText(resolved_lines)
        self.command_preview.setPlainText(command_text)


class ModelSummaryDialog(QDialog):
    """Simple dialog for the completed run summary."""

    def __init__(self, *, summary_text: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Summary")
        self.resize(760, 540)
        layout = QVBoxLayout(self)
        text = QPlainTextEdit(self)
        text.setReadOnly(True)
        text.setPlainText(summary_text)
        layout.addWidget(text, stretch=1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.accept)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)


class NewModelDialog(QDialog):
    """Configure, launch, monitor, and optionally save a new training run."""

    def __init__(
        self,
        *,
        current_portfolio_tickers: list[str],
        current_benchmark_ticker: str,
        current_duration_key: str,
        current_start_date: str,
        current_end_date: str,
        current_max_portfolio_size: int,
        portfolio_library_service: PortfolioLibraryService | None = None,
        refresh_models_callback=None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Model")
        self.resize(1120, 760)

        self._root = Path(__file__).resolve().parents[3]
        self._current_portfolio_tickers = [ticker.strip().upper() for ticker in current_portfolio_tickers if ticker.strip()]
        self._refresh_models_callback = refresh_models_callback
        self._portfolio_library_service = portfolio_library_service or PortfolioLibraryService()
        self._training_service = ModelTrainingService(self._root, self)
        self._current_launch = None
        self._current_request: ModelTrainingRequest | None = None
        self._metric_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
        self._candidate_rows: list[dict[str, object]] = []
        self._active_candidate = "training_run"
        self.completed_model_path: Path | None = None
        self._last_output_dir: Path | None = None
        self._pending_output_dir: Path | None = None
        self._last_universe_resolution_error: str | None = None
        self._all_logs: list[str] = []
        self._run_complete = False
        self._save_completed = False
        self._advanced_values: dict[str, object] = {}
        self._monitor_dialog: TrainingMonitorDialog | None = None
        self._broad_config_tickers = load_config(self._root / "config" / "broad_universe_config.yaml").data.tickers

        self._training_service.state_changed.connect(self._on_state_changed)
        self._training_service.log_received.connect(self._append_log)
        self._training_service.event_received.connect(self._handle_event)
        self._training_service.run_finished.connect(self._on_run_finished)

        root_layout = QVBoxLayout(self)
        intro = QLabel(
            "Configure a new training run. Training launches one of the existing scripts, then moves into a dedicated monitor window.",
            self,
        )
        intro.setWordWrap(True)
        root_layout.addWidget(intro)

        top_grid = QGridLayout()
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 1)
        top_grid.setColumnStretch(2, 1)

        universe_group = QGroupBox("Training Universe", self)
        universe_layout = QVBoxLayout(universe_group)
        universe_form = QFormLayout()
        self.universe_source_combo = QComboBox(universe_group)
        for label, value in UNIVERSE_SOURCE_OPTIONS:
            self.universe_source_combo.addItem(label, value)
        self.preset_combo = QComboBox(universe_group)
        self.manual_tickers_edit = QPlainTextEdit(universe_group)
        self.manual_tickers_edit.setPlaceholderText("Enter tickers separated by commas, spaces, or new lines.")
        self.file_path_input = QLineEdit(universe_group)
        self.file_path_input.setPlaceholderText("Import .txt or .csv with ticker symbols")
        self.browse_button = QPushButton("Browse…", universe_group)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path_input, stretch=1)
        file_row.addWidget(self.browse_button)
        universe_form.addRow("Source", self.universe_source_combo)
        universe_form.addRow("Preset", self.preset_combo)
        universe_layout.addLayout(universe_form)
        universe_layout.addWidget(QLabel("Manual / Imported Tickers", universe_group))
        universe_layout.addWidget(self.manual_tickers_edit, stretch=1)
        universe_layout.addLayout(file_row)
        self.resolved_universe_label = QLabel(universe_group)
        self.resolved_universe_label.setWordWrap(True)
        universe_layout.addWidget(self.resolved_universe_label)
        top_grid.addWidget(universe_group, 0, 0)

        basic_group = QGroupBox("Basic Hyperparameters", self)
        basic_layout = QVBoxLayout(basic_group)
        basic_form = QFormLayout()
        self.model_size_combo = QComboBox(basic_group)
        self.model_size_combo.addItem("10", 10)
        self.model_size_combo.addItem("50", 50)
        self.model_size_combo.setCurrentIndex(1 if int(current_max_portfolio_size) > 10 else 0)
        self.training_mode_combo = QComboBox(basic_group)
        self.training_mode_combo.addItem("Portfolio Fit", "portfolio_fit")
        self.training_mode_combo.addItem("Experiment", "experiment")
        self.training_mode_combo.addItem("RL Policy", "rl_policy")
        self.lookback_spin = QSpinBox(basic_group)
        self.lookback_spin.setRange(5, 756)
        self.epochs_spin = QSpinBox(basic_group)
        self.epochs_spin.setRange(8, 1024)
        self.batch_spin = QSpinBox(basic_group)
        self.batch_spin.setRange(8, 512)
        self.seed_spin = QSpinBox(basic_group)
        self.seed_spin.setRange(0, 1_000_000)
        self.lr_spin = QDoubleSpinBox(basic_group)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.000001, 1.0)
        self.lr_spin.setSingleStep(0.0001)
        self.device_combo = QComboBox(basic_group)
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        self.candidate_mode_combo = QComboBox(basic_group)
        self.candidate_mode_combo.addItem("Standard", "standard")
        self.candidate_mode_combo.addItem("Experimental", "experimental")
        self.candidate_mode_combo.addItem("Comprehensive", "comprehensive")
        self.random_universes_spin = QSpinBox(basic_group)
        self.random_universes_spin.setRange(8, 4096)
        self.candidate_pool_spin = QSpinBox(basic_group)
        self.candidate_pool_spin.setRange(5, 1024)
        self.objectives_input = QLineEdit("min_variance, mean_variance, risk_parity, equal_weight", basic_group)
        basic_form.addRow("Model Size", self.model_size_combo)
        basic_form.addRow("Training Mode", self.training_mode_combo)
        basic_form.addRow("Lookback", self.lookback_spin)
        basic_form.addRow("Epochs", self.epochs_spin)
        basic_form.addRow("Batch Size", self.batch_spin)
        basic_form.addRow("Seed", self.seed_spin)
        basic_form.addRow("Learning Rate", self.lr_spin)
        basic_form.addRow("Device", self.device_combo)
        basic_form.addRow("Candidate Set", self.candidate_mode_combo)
        basic_form.addRow("Random Universes", self.random_universes_spin)
        basic_form.addRow("Candidate Pool", self.candidate_pool_spin)
        basic_form.addRow("Objectives", self.objectives_input)
        basic_layout.addLayout(basic_form)
        advanced_row = QHBoxLayout()
        advanced_row.addStretch(1)
        self.advanced_button = QPushButton("Advanced", basic_group)
        advanced_row.addWidget(self.advanced_button)
        basic_layout.addLayout(advanced_row)
        top_grid.addWidget(basic_group, 0, 1)

        dates_group = QGroupBox("Dates, Horizon, Benchmark", self)
        dates_form = QFormLayout(dates_group)
        self.duration_combo = QComboBox(dates_group)
        for profile in REPLAY_DURATION_PROFILES:
            self.duration_combo.addItem(profile.label, profile.key)
        duration_index = max(0, self.duration_combo.findData(current_duration_key))
        self.duration_combo.setCurrentIndex(duration_index)
        self.start_date_edit = QDateEdit(dates_group)
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        start_ts = pd.Timestamp(current_start_date)
        self.start_date_edit.setDate(QDate(start_ts.year, start_ts.month, start_ts.day))
        self.end_date_edit = QDateEdit(dates_group)
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        end_ts = pd.Timestamp(current_end_date)
        self.end_date_edit.setDate(QDate(end_ts.year, end_ts.month, end_ts.day))
        self.rebalance_combo = QComboBox(dates_group)
        for label, value in REBALANCE_OPTIONS:
            self.rebalance_combo.addItem(label, value)
        self.benchmark_mode_combo = QComboBox(dates_group)
        for label, value in BENCHMARK_MODE_OPTIONS:
            self.benchmark_mode_combo.addItem(label, value)
        self.benchmark_ticker_input = QLineEdit(current_benchmark_ticker, dates_group)
        self.equal_weight_scope_combo = QComboBox(dates_group)
        self.equal_weight_scope_combo.addItem("Training Universe", "training_universe")
        dates_form.addRow("Horizon", self.duration_combo)
        dates_form.addRow("Start", self.start_date_edit)
        dates_form.addRow("End", self.end_date_edit)
        dates_form.addRow("Interval", self.rebalance_combo)
        dates_form.addRow("Benchmark", self.benchmark_mode_combo)
        dates_form.addRow("Ticker", self.benchmark_ticker_input)
        dates_form.addRow("Equal-Weight Scope", self.equal_weight_scope_combo)
        top_grid.addWidget(dates_group, 0, 2)

        root_layout.addLayout(top_grid)

        post_run_group = QGroupBox("Completed Run", self)
        post_run_layout = QVBoxLayout(post_run_group)
        actions_row = QHBoxLayout()
        self.save_model_button = QPushButton("Save Model", post_run_group)
        self.view_all_graphs_button = QPushButton("View All Graphs", post_run_group)
        self.view_model_summary_button = QPushButton("View Model Summary", post_run_group)
        actions_row.addWidget(self.save_model_button)
        actions_row.addWidget(self.view_all_graphs_button)
        actions_row.addWidget(self.view_model_summary_button)
        actions_row.addStretch(1)
        post_run_layout.addLayout(actions_row)
        graph_row = QHBoxLayout()
        self.completed_loss_canvas = LineMetricCanvas(
            title="Loss",
            ylabel="Loss",
            empty_message="Loss graph will appear after training completes",
            parent=post_run_group,
            lock_y_on_first_plot=True,
        )
        self.completed_surface_canvas = Training3DCanvas(post_run_group)
        graph_row.addWidget(self.completed_loss_canvas, stretch=1)
        graph_row.addWidget(self.completed_surface_canvas, stretch=1)
        post_run_layout.addLayout(graph_row, stretch=1)
        self.post_run_group = post_run_group
        self.post_run_group.setVisible(False)
        root_layout.addWidget(post_run_group, stretch=1)

        self.metadata_tabs = QTabWidget(self)
        metadata_page = QWidget(self.metadata_tabs)
        metadata_form = QFormLayout(metadata_page)
        self.save_name_input = QLineEdit(metadata_page)
        self.save_description_input = QLineEdit(metadata_page)
        self.save_tags_input = QLineEdit(metadata_page)
        self.save_output_category_combo = QComboBox(metadata_page)
        self.auto_select_checkbox = QCheckBox("Auto-select when complete", metadata_page)
        self.auto_select_checkbox.setChecked(True)
        self.save_status_label = QLabel(metadata_page)
        self.save_status_label.setWordWrap(True)
        metadata_form.addRow("Name", self.save_name_input)
        metadata_form.addRow("Description", self.save_description_input)
        metadata_form.addRow("Tags", self.save_tags_input)
        metadata_form.addRow("Save To", self.save_output_category_combo)
        metadata_form.addRow("", self.auto_select_checkbox)
        metadata_form.addRow("", self.save_status_label)
        self.metadata_tabs.addTab(metadata_page, "Metadata")
        self.metadata_tabs.setVisible(False)
        root_layout.addWidget(self.metadata_tabs)

        controls_row = QHBoxLayout()
        self.state_label = QLabel("State: idle", self)
        self.start_button = QPushButton("Start", self)
        self.close_button = QPushButton("Close", self)
        self.confirm_save_button = QPushButton("Confirm Save", self)
        self.hide_metadata_button = QPushButton("Hide Metadata", self)
        controls_row.addWidget(self.state_label)
        controls_row.addStretch(1)
        controls_row.addWidget(self.start_button)
        controls_row.addWidget(self.confirm_save_button)
        controls_row.addWidget(self.hide_metadata_button)
        controls_row.addWidget(self.close_button)
        root_layout.addLayout(controls_row)

        self.advanced_button.clicked.connect(self._open_advanced_dialog)
        self.start_button.clicked.connect(self._start_training)
        self.close_button.clicked.connect(self.reject)
        self.confirm_save_button.clicked.connect(self._save_model)
        self.hide_metadata_button.clicked.connect(lambda: self.metadata_tabs.setVisible(False))
        self.save_model_button.clicked.connect(self._show_metadata_tab)
        self.view_all_graphs_button.clicked.connect(self._view_all_graphs)
        self.view_model_summary_button.clicked.connect(self._view_model_summary)
        self.browse_button.clicked.connect(self._browse_file)

        for widget in (
            self.file_path_input,
            self.manual_tickers_edit,
            self.benchmark_ticker_input,
            self.objectives_input,
        ):
            signal = widget.textChanged if hasattr(widget, "textChanged") else None
            if signal is not None:
                signal.connect(self._update_preview_labels)
        for combo in (
            self.model_size_combo,
            self.training_mode_combo,
            self.universe_source_combo,
            self.preset_combo,
            self.duration_combo,
            self.rebalance_combo,
            self.benchmark_mode_combo,
            self.equal_weight_scope_combo,
            self.device_combo,
            self.candidate_mode_combo,
        ):
            combo.currentIndexChanged.connect(self._update_preview_labels)
        for spin in (
            self.lookback_spin,
            self.epochs_spin,
            self.batch_spin,
            self.seed_spin,
            self.lr_spin,
            self.random_universes_spin,
            self.candidate_pool_spin,
        ):
            spin.valueChanged.connect(self._update_preview_labels)

        self.training_mode_combo.currentIndexChanged.connect(self._reset_defaults_for_mode)
        self.duration_combo.currentIndexChanged.connect(self._reset_defaults_for_mode)
        self.model_size_combo.currentIndexChanged.connect(self._on_model_size_changed)
        self.universe_source_combo.currentIndexChanged.connect(self._on_universe_source_changed)
        self.benchmark_mode_combo.currentIndexChanged.connect(self._on_benchmark_mode_changed)

        self._populate_presets()
        self._reset_defaults_for_mode()
        self._on_universe_source_changed()
        self._on_benchmark_mode_changed()
        self._set_pre_run_state()
        self._update_preview_labels()

    def _selected_model_size(self) -> int:
        return int(self.model_size_combo.currentData() or 10)

    def _selected_mode(self) -> str:
        return str(self.training_mode_combo.currentData() or "portfolio_fit")

    def _populate_presets(self) -> None:
        self.preset_combo.clear()
        for configuration in self._portfolio_library_service.list_preset_configurations(max_portfolio_size=self._selected_model_size()):
            self.preset_combo.addItem(configuration.name, configuration)

    def _reset_defaults_for_mode(self) -> None:
        self._pending_output_dir = None
        defaults = self._training_service.default_hyperparameters(
            mode=self._selected_mode(),
            duration_key=str(self.duration_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY),
        )
        self.lookback_spin.setValue(int(defaults["lookback_window"]))
        self.epochs_spin.setValue(int(defaults["epochs"]))
        self.batch_spin.setValue(int(defaults["batch_size"]))
        self.seed_spin.setValue(int(defaults["seed"]))
        self.lr_spin.setValue(float(defaults["learning_rate"]))
        self.device_combo.setCurrentIndex(max(0, self.device_combo.findData(str(defaults.get("device", "auto")))))
        self.candidate_mode_combo.setCurrentIndex(max(0, self.candidate_mode_combo.findData(str(defaults.get("candidate_mode", "experimental")))))
        self.random_universes_spin.setValue(int(defaults.get("random_universes", 256)))
        self.candidate_pool_spin.setValue(int(defaults.get("candidate_pool_size", 80)))
        self.objectives_input.setText(", ".join(str(item) for item in defaults.get("objectives", ["min_variance", "mean_variance", "risk_parity", "equal_weight"])))
        self._advanced_values = {
            "hidden_dim": int(defaults["hidden_dim"]),
            "attention_heads": int(defaults["attention_heads"]),
            "attention_layers": int(defaults["attention_layers"]),
            "dropout": float(defaults["dropout"]),
            "weight_decay": float(defaults["weight_decay"]),
            "actor_bc_weight": float(defaults["actor_bc_weight"]),
            "entropy_weight": float(defaults["entropy_weight"]),
            "validation_fraction": float(defaults["validation_fraction"]),
            "optimizer": str(defaults["optimizer"]),
            "checkpoint_frequency": int(defaults["checkpoint_frequency"]),
            "early_stopping_patience": int(defaults["early_stopping_patience"]),
            "reward_weight_vs_benchmark": float(defaults["reward_weight_vs_benchmark"]),
            "reward_weight_vs_equal_weight": float(defaults["reward_weight_vs_equal_weight"]),
            "reward_weight_vs_restricted_random": float(defaults["reward_weight_vs_restricted_random"]),
            "reward_weight_vs_markowitz": float(defaults["reward_weight_vs_markowitz"]),
            "reward_weight_raw": float(defaults["reward_weight_raw"]),
        }
        self._sync_mode_visibility()
        self._populate_save_categories()
        self._update_preview_labels()

    def _sync_mode_visibility(self) -> None:
        mode = self._selected_mode()
        self.candidate_mode_combo.setVisible(mode == "portfolio_fit")
        self.random_universes_spin.setVisible(mode == "experiment")
        self.candidate_pool_spin.setVisible(mode == "experiment")
        self.objectives_input.setVisible(mode == "rl_policy")

    def _on_model_size_changed(self) -> None:
        self._pending_output_dir = None
        self._populate_presets()
        self._update_preview_labels()

    def _on_universe_source_changed(self) -> None:
        source = str(self.universe_source_combo.currentData() or "current_portfolio")
        self.preset_combo.setEnabled(source == "preset")
        manual_enabled = source == "manual"
        file_enabled = source == "file"
        self.manual_tickers_edit.setEnabled(manual_enabled or file_enabled)
        self.file_path_input.setEnabled(file_enabled)
        self.browse_button.setEnabled(file_enabled)
        self._update_preview_labels()

    def _on_benchmark_mode_changed(self) -> None:
        mode = str(self.benchmark_mode_combo.currentData() or "ticker")
        self.benchmark_ticker_input.setEnabled(mode == "ticker")
        self.equal_weight_scope_combo.setEnabled(mode == "equal_weight")
        self._update_preview_labels()

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Tickers",
            self._root.as_posix(),
            "Ticker Files (*.txt *.csv);;All Files (*)",
        )
        if not path:
            return
        self.file_path_input.setText(path)
        try:
            tickers = ModelTrainingService.load_tickers_from_file(path)
        except Exception as exc:
            QMessageBox.warning(self, "Import Tickers", str(exc))
            return
        self.manual_tickers_edit.setPlainText(", ".join(tickers))
        self._update_preview_labels()

    def _resolved_tickers(self) -> list[str]:
        self._last_universe_resolution_error = None
        source = str(self.universe_source_combo.currentData() or "current_portfolio")
        if source == "current_portfolio":
            return list(self._current_portfolio_tickers)
        if source == "preset":
            preset = self.preset_combo.currentData()
            if isinstance(preset, SavedConfiguration):
                return list(preset.tickers)
            return []
        if source == "canonical":
            base = CANONICAL_TOP_50_UNIVERSE if self._selected_model_size() > 10 else CANONICAL_TOP_ETF_UNIVERSE
            return list(base[: self._selected_model_size()])
        if source == "broad_config":
            return [ticker.strip().upper() for ticker in self._broad_config_tickers if ticker.strip()][: self._selected_model_size()]
        if source == "file":
            path = self.file_path_input.text().strip()
            if not path:
                return parse_ticker_input(self.manual_tickers_edit.toPlainText())
            try:
                return ModelTrainingService.load_tickers_from_file(path)
            except Exception as exc:
                self._last_universe_resolution_error = str(exc)
                return []
        return parse_ticker_input(self.manual_tickers_edit.toPlainText())

    def _build_request(self, *, for_preview: bool = False, advanced_values: dict[str, object] | None = None) -> ModelTrainingRequest:
        if self._last_output_dir is not None:
            temp_output_dir = self._last_output_dir
        else:
            if self._pending_output_dir is None:
                self._pending_output_dir = self._training_service.build_temporary_output_dir(
                    mode=self._selected_mode(),
                    duration_key=str(self.duration_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY),
                )
            temp_output_dir = self._pending_output_dir
        temp_name = temp_output_dir.name
        values = {**self._advanced_values, **(advanced_values or {})}
        objectives = [item.strip() for item in self.objectives_input.text().split(",") if item.strip()]
        hyperparameters: dict[str, object] = {
            "lookback_window": int(self.lookback_spin.value()),
            "epochs": int(self.epochs_spin.value()),
            "batch_size": int(self.batch_spin.value()),
            "seed": int(self.seed_spin.value()),
            "learning_rate": float(self.lr_spin.value()),
            "device": str(self.device_combo.currentData() or "auto"),
            **values,
        }
        mode = self._selected_mode()
        if mode == "portfolio_fit":
            hyperparameters["candidate_mode"] = str(self.candidate_mode_combo.currentData() or "experimental")
        if mode == "experiment":
            hyperparameters["random_universes"] = int(self.random_universes_spin.value())
            hyperparameters["candidate_pool_size"] = int(self.candidate_pool_spin.value())
        if mode == "rl_policy":
            hyperparameters["objectives"] = objectives

        return ModelTrainingRequest(
            name=temp_name,
            description="",
            tags=[],
            model_size=self._selected_model_size(),
            training_mode=mode,
            universe_source=str(self.universe_source_combo.currentData() or "current_portfolio"),
            tickers=self._resolved_tickers(),
            start_date=self.start_date_edit.date().toString("yyyy-MM-dd"),
            end_date=self.end_date_edit.date().toString("yyyy-MM-dd"),
            duration_key=str(self.duration_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY),
            rebalance_frequency=str(self.rebalance_combo.currentData() or "ME"),
            benchmark_mode=str(self.benchmark_mode_combo.currentData() or "ticker"),
            benchmark_ticker=self.benchmark_ticker_input.text().strip(),
            equal_weight_scope=str(self.equal_weight_scope_combo.currentData() or "training_universe"),
            output_category=self._training_service.output_categories_for_mode(mode)[0],
            output_dir_override=temp_output_dir.as_posix(),
            hyperparameters=hyperparameters,
        )

    def _preview_payload(self, advanced_values: dict[str, object] | None = None) -> tuple[str, str, str, str]:
        request = self._build_request(for_preview=True, advanced_values=advanced_values)
        resolved_tickers = request.tickers
        summary_lines = []
        if self._last_universe_resolution_error:
            summary_lines.append(self._last_universe_resolution_error)
        try:
            launch = self._training_service.resolve_request(request)
        except Exception as exc:
            return (
                "\n".join(summary_lines + [str(exc)]),
                f"Output Path: {request.output_dir_override or '—'}",
                "",
                "",
            )
        summary_lines.extend(
            [
                f"Universe: {len(resolved_tickers)} tickers",
                f"Benchmark: {launch.benchmark_label}",
                f"Mode: {request.training_mode}",
                f"Horizon: {request.duration_key}",
            ]
        )
        resolved_lines = [f"{key}: {value}" for key, value in sorted(launch.resolved_hyperparameters.items(), key=lambda item: item[0])]
        return (
            "\n".join(summary_lines),
            f"Output Path: {launch.output_dir.as_posix()}",
            "\n".join(resolved_lines),
            launch.command_text,
        )

    def _update_preview_labels(self) -> None:
        tickers = self._resolved_tickers()
        self.resolved_universe_label.setText(
            f"Resolved Universe ({len(tickers)}): {', '.join(tickers) if tickers else '—'}"
        )

    def _open_advanced_dialog(self) -> None:
        dialog = AdvancedOptionsDialog(
            initial_values=self._advanced_values,
            preview_callback=self._preview_payload,
            parent=self,
        )
        if dialog.exec():
            self._advanced_values = dialog.values()
        self._update_preview_labels()

    def _set_pre_run_state(self) -> None:
        self._run_complete = False
        self._save_completed = False
        self._pending_output_dir = None
        self.post_run_group.setVisible(False)
        self.metadata_tabs.setVisible(False)
        self.confirm_save_button.setVisible(False)
        self.hide_metadata_button.setVisible(False)
        self.save_model_button.setVisible(False)
        self.view_all_graphs_button.setVisible(False)
        self.view_model_summary_button.setVisible(False)
        self.start_button.setVisible(True)
        self.start_button.setEnabled(True)
        self.state_label.setText("State: idle")

    def _set_post_run_state(self) -> None:
        self.post_run_group.setVisible(True)
        self.save_model_button.setVisible(True)
        self.view_all_graphs_button.setVisible(True)
        self.view_model_summary_button.setVisible(True)
        self.start_button.setVisible(False)
        self.completed_loss_canvas.plot_columns(
            self._active_history_frame(),
            ["train_total_loss", "train_actor_loss", "train_critic_loss", "train_bc_loss", "validation_loss"],
        )
        self.completed_surface_canvas.update_points(self._candidate_rows)

    def _start_training(self) -> None:
        request = self._build_request()
        try:
            launch = self._training_service.start_training(request)
        except Exception as exc:
            QMessageBox.warning(self, "New Model", str(exc))
            return
        self._current_request = request
        self._current_launch = launch
        self._last_output_dir = launch.output_dir
        self._pending_output_dir = None
        self._metric_rows.clear()
        self._candidate_rows.clear()
        self._active_candidate = "training_run"
        self._all_logs.clear()
        self.completed_model_path = None
        self.metadata_tabs.setVisible(False)
        self._populate_save_categories()

        self._monitor_dialog = TrainingMonitorDialog(self)
        self._monitor_dialog.cancel_button.clicked.connect(self._cancel_training)
        self._monitor_dialog.close_button.clicked.connect(self._monitor_dialog.accept)
        self._monitor_dialog.set_running(True, state_label="running")
        self.hide()
        self._monitor_dialog.exec()
        self.show()
        self.raise_()
        self.activateWindow()
        if self._run_complete:
            self._set_post_run_state()
        else:
            self._set_pre_run_state()

    def _cancel_training(self) -> None:
        if not self._training_service.is_running:
            return
        confirm = QMessageBox.question(self, "Cancel Training", "Cancel the active training run?")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._training_service.cancel()

    def _append_log(self, line: str, _stream: str) -> None:
        self._all_logs.append(line)
        if self._monitor_dialog is not None:
            self._monitor_dialog.append_log(line)

    def _handle_event(self, event: dict[str, object]) -> None:
        event_type = str(event.get("event") or "")
        if event_type == "run_initialized":
            if self._monitor_dialog is not None:
                total_candidates = int(event.get("candidate_total") or 0)
                self._monitor_dialog.set_candidate_plan(
                    total_candidates=total_candidates,
                    active_candidate="waiting to start",
                    completed_candidates=0,
                )
        elif event_type == "candidate_started":
            candidate = str(event.get("candidate") or "training_run")
            candidate_changed = candidate != self._active_candidate
            self._active_candidate = candidate
            if self._monitor_dialog is not None:
                if candidate_changed:
                    self._monitor_dialog.reset_metric_views()
                total_candidates = int(event.get("total_candidates") or event.get("candidate_total") or 0)
                candidate_index = int(event.get("candidate_index") or 0)
                self._monitor_dialog.set_candidate_plan(
                    total_candidates=total_candidates,
                    active_candidate=candidate,
                    completed_candidates=max(candidate_index - 1, len(self._candidate_rows)),
                )
        elif event_type == "epoch_metrics":
            candidate = str(event.get("candidate") or "training_run")
            metrics = event.get("metrics")
            if isinstance(metrics, dict):
                self._metric_rows[candidate].append(metrics)
                self._active_candidate = candidate
                history = pd.DataFrame(self._metric_rows[candidate]).sort_values("epoch")
                if self._monitor_dialog is not None:
                    self._monitor_dialog.update_history(history)
        elif event_type == "candidate_completed":
            self._candidate_rows.append(event)
            if self._monitor_dialog is not None:
                self._monitor_dialog.set_candidate_rows(self._candidate_rows)
                total_candidates = int(event.get("total_candidates") or self._monitor_dialog._candidate_total)
                self._monitor_dialog.set_candidate_plan(
                    total_candidates=total_candidates,
                    active_candidate="waiting for next candidate",
                    completed_candidates=len(self._candidate_rows),
                )
        elif event_type == "run_complete":
            model_path = event.get("model_path")
            if model_path:
                self.completed_model_path = Path(str(model_path))
            if self._monitor_dialog is not None and self._monitor_dialog._candidate_total > 0:
                self._monitor_dialog.set_candidate_plan(
                    total_candidates=self._monitor_dialog._candidate_total,
                    active_candidate="completed",
                    completed_candidates=max(len(self._candidate_rows), self._monitor_dialog._candidate_total),
                )

    def _on_state_changed(self, state: str) -> None:
        self.state_label.setText(f"State: {state}")
        if self._monitor_dialog is not None:
            self._monitor_dialog.set_running(state == "running", state_label=state)

    def _on_run_finished(self, result: dict[str, object]) -> None:
        success = bool(result.get("success"))
        cancelled = bool(result.get("cancelled"))
        self._run_complete = success
        if self._monitor_dialog is not None:
            self._monitor_dialog.set_running(False, state_label="done" if success else ("cancelled" if cancelled else "failed"))
        if not success and not cancelled:
            QMessageBox.warning(self, "New Model", f"Training failed with exit code {result.get('exit_code')}.")

    def _active_history_frame(self) -> pd.DataFrame:
        rows = self._metric_rows.get(self._active_candidate) or next(iter(self._metric_rows.values()), [])
        if not rows:
            return pd.DataFrame(columns=["epoch"])
        return pd.DataFrame(rows).sort_values("epoch")

    def _view_all_graphs(self) -> None:
        dialog = GraphResultsDialog(
            history=self._active_history_frame(),
            candidate_rows=self._candidate_rows,
            cli_text="\n".join(self._all_logs),
            parent=self,
        )
        dialog.exec()

    def _view_model_summary(self) -> None:
        summary_lines = []
        if self._current_request is not None:
            summary_lines.extend(
                [
                    f"Training Mode: {self._current_request.training_mode}",
                    f"Model Size: {self._current_request.model_size}",
                    f"Horizon: {self._current_request.duration_key}",
                    f"Universe: {', '.join(self._current_request.tickers)}",
                ]
            )
        if self._current_launch is not None:
            summary_lines.extend(
                [
                    f"Benchmark: {self._current_launch.benchmark_label}",
                    f"Training Output: {self._current_launch.output_dir.as_posix()}",
                ]
            )
        if self.completed_model_path is not None:
            summary_lines.append(f"Promoted Model Path: {self.completed_model_path.as_posix()}")
        if self._candidate_rows:
            frame = pd.DataFrame(self._candidate_rows)
            best = frame.sort_values("all_composite_score", ascending=False).iloc[0]
            summary_lines.extend(
                [
                    "",
                    "Best Candidate",
                    f"Candidate: {best.get('candidate', 'unknown')}",
                    f"Composite Score: {best.get('all_composite_score', '—')}",
                    f"Selected Epoch: {best.get('selected_epoch', '—')}",
                ]
            )
        metadata_path = (self._last_output_dir / "model_metadata.json") if self._last_output_dir is not None else None
        if metadata_path is not None and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            summary_lines.extend(["", "Metadata", json.dumps(metadata, indent=2, sort_keys=True)])
        dialog = ModelSummaryDialog(summary_text="\n".join(summary_lines), parent=self)
        dialog.exec()

    def _populate_save_categories(self) -> None:
        self.save_output_category_combo.clear()
        for category in self._training_service.output_categories_for_mode(self._selected_mode()):
            self.save_output_category_combo.addItem(category, category)

    def _show_metadata_tab(self) -> None:
        self.metadata_tabs.setVisible(True)
        self.confirm_save_button.setVisible(True)
        self.hide_metadata_button.setVisible(True)
        default_name = f"{self.duration_combo.currentData()}_{'_'.join(self._resolved_tickers()[:3]).lower()}_{self._selected_mode()}"
        if not self.save_name_input.text().strip():
            self.save_name_input.setText(default_name)
        self.save_status_label.setText("Save the trained run into one of the discovered model roots.")

    def _save_model(self) -> None:
        if self._save_completed:
            return
        if self._last_output_dir is None or self._current_request is None:
            QMessageBox.warning(self, "Save Model", "No completed training run is available to save.")
            return
        name = self.save_name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Save Model", "Model name is required.")
            return
        tags = [tag.strip() for tag in self.save_tags_input.text().split(",") if tag.strip()]
        try:
            model_path = self._training_service.save_trained_model(
                training_output_dir=self._last_output_dir,
                training_mode=self._current_request.training_mode,
                model_size=self._current_request.model_size,
                duration_key=self._current_request.duration_key,
                name=name,
                description=self.save_description_input.text().strip(),
                tags=tags,
                output_category=str(self.save_output_category_combo.currentData() or self._training_service.output_categories_for_mode(self._current_request.training_mode)[0]),
            )
        except Exception as exc:
            QMessageBox.warning(self, "Save Model", str(exc))
            return
        self._save_completed = True
        self.completed_model_path = model_path
        self.save_status_label.setText(f"Saved to {model_path.as_posix()}")
        if self._refresh_models_callback is not None:
            self._refresh_models_callback(model_path if self.auto_select_checkbox.isChecked() else None)
        QMessageBox.information(self, "Save Model", f"Saved model to {model_path.as_posix()}.")
        self.accept()

    def reject(self) -> None:
        if self._training_service.is_running:
            QMessageBox.information(self, "New Model", "Close the monitor or cancel the run before closing this window.")
            return
        super().reject()

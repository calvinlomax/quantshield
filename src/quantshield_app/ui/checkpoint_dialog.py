"""Dialog for browsing and selecting trained replay models."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from quantshield.metrics import drawdown_series
from quantshield.replay_durations import DEFAULT_REPLAY_DURATION_KEY, REPLAY_DURATION_PROFILES
from quantshield_app.services import CheckpointDescriptor, PortfolioLibraryService
from quantshield_app.ui.new_model_dialog import NewModelDialog

ALLOWED_PORTFOLIO_SIZES = (10, 50)


class SortableTableWidgetItem(QTableWidgetItem):
    """Table item that sorts by an explicit hidden key when available."""

    SORT_ROLE = Qt.ItemDataRole.UserRole + 1

    def __lt__(self, other: object) -> bool:
        if isinstance(other, QTableWidgetItem):
            left = self.data(self.SORT_ROLE)
            right = other.data(self.SORT_ROLE)
            if left is not None and right is not None:
                return left < right
        return super().__lt__(other)


class ModelPreviewCanvas(FigureCanvasQTAgg):
    """Mini equity and drawdown preview for a checkpoint descriptor."""

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(4.8, 2.2))
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.equity_axes = self.figure.add_subplot(121)
        self.drawdown_axes = self.figure.add_subplot(122)
        self.figure.subplots_adjust(left=0.08, right=0.98, bottom=0.28, top=0.86, wspace=0.30)

    def set_descriptor(self, descriptor: CheckpointDescriptor | None) -> None:
        self.equity_axes.clear()
        self.drawdown_axes.clear()
        if descriptor is None or not descriptor.preview_policy_predictions_path.exists():
            self.equity_axes.text(0.5, 0.5, "No preview available", ha="center", va="center")
            self.equity_axes.axis("off")
            self.drawdown_axes.axis("off")
            self.draw_idle()
            return
        preview = pd.read_csv(descriptor.preview_policy_predictions_path)
        if "rebalance_date" not in preview.columns or "policy_raw_return" not in preview.columns:
            self.equity_axes.text(0.5, 0.5, "No preview available", ha="center", va="center")
            self.equity_axes.axis("off")
            self.drawdown_axes.axis("off")
            self.draw_idle()
            return

        preview["rebalance_date"] = pd.to_datetime(preview["rebalance_date"])
        preview = preview.sort_values("rebalance_date")
        returns = preview["policy_raw_return"].astype(float)
        equity = (1.0 + returns).cumprod()
        drawdowns = drawdown_series(returns)

        self.equity_axes.plot(preview["rebalance_date"], equity, linewidth=1.5, color="#1f77b4")
        self.equity_axes.set_title("Mini Equity", fontsize=9)
        self.equity_axes.tick_params(axis="x", labelsize=7, rotation=25)
        self.equity_axes.tick_params(axis="y", labelsize=7)
        self.equity_axes.grid(alpha=0.2)

        self.drawdown_axes.plot(preview["rebalance_date"], drawdowns, linewidth=1.5, color="#d62728")
        self.drawdown_axes.fill_between(preview["rebalance_date"], drawdowns.to_numpy(dtype=float), 0.0, alpha=0.16, color="#d62728")
        self.drawdown_axes.set_title("Mini Drawdown", fontsize=9)
        self.drawdown_axes.tick_params(axis="x", labelsize=7, rotation=25)
        self.drawdown_axes.tick_params(axis="y", labelsize=7)
        self.drawdown_axes.grid(alpha=0.2)
        self.draw_idle()


class ModelComparisonDialog(QDialog):
    """Full comparison analysis for a selected set of models."""

    def __init__(self, *, descriptors: list[CheckpointDescriptor], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare Models")
        self.resize(1220, 760)
        self._descriptors = list(descriptors)

        layout = QVBoxLayout(self)
        names = ", ".join(descriptor.display_name for descriptor in descriptors)
        intro = QLabel(
            "Selected models are compared across core metadata and performance signals. "
            "Greener cells are more desirable for that metric, while redder cells are less desirable.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        summary_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(summary_splitter, stretch=1)

        comparison_panel = QWidget(summary_splitter)
        comparison_layout = QVBoxLayout(comparison_panel)
        self.analysis_label = QLabel(self._analysis_text(names), comparison_panel)
        self.analysis_label.setWordWrap(True)
        comparison_layout.addWidget(self.analysis_label)

        self.table = QTableWidget(comparison_panel)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.setRowCount(0)
        self.table.setColumnCount(len(descriptors) + 1)
        self.table.setHorizontalHeaderLabels(["Metric", *[descriptor.display_name for descriptor in descriptors]])
        comparison_layout.addWidget(self.table, stretch=1)
        summary_splitter.addWidget(comparison_panel)

        best_panel = QWidget(summary_splitter)
        best_layout = QVBoxLayout(best_panel)
        best_group = QGroupBox("Global Best Model", best_panel)
        best_group_layout = QVBoxLayout(best_group)
        best_text = QLabel(self._best_model_text(descriptors), best_group)
        best_text.setWordWrap(True)
        best_group_layout.addWidget(best_text)
        best_layout.addWidget(best_group)

        notes_group = QGroupBox("How To Read This", best_panel)
        notes_layout = QVBoxLayout(notes_group)
        notes_label = QLabel(
            "The winner is chosen from a composite built from quality tag, validation excess return, "
            "all-sample excess return, and all-sample t-statistic. Missing metrics are penalized.",
            notes_group,
        )
        notes_label.setWordWrap(True)
        notes_layout.addWidget(notes_label)
        best_layout.addWidget(notes_group)
        best_layout.addStretch(1)
        summary_splitter.addWidget(best_panel)
        summary_splitter.setSizes([900, 320])

        self._populate_table()

        button_box = QDialogButtonBox(self)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

    @staticmethod
    def _analysis_text(names: str) -> str:
        return (
            f"Comparing {names}. This table mixes metadata with benchmark-relative performance so the differences are visible "
            "without opening each model individually."
        )

    @staticmethod
    def _quality_score(descriptor: CheckpointDescriptor) -> float:
        if descriptor.quality_tag == "Validated":
            return 3.0
        if descriptor.quality_tag == "Benchmark+":
            return 2.0
        return 1.0

    @classmethod
    def _global_score(cls, descriptor: CheckpointDescriptor) -> float:
        metrics = [
            cls._quality_score(descriptor),
            descriptor.validation_mean_excess_return if descriptor.validation_mean_excess_return is not None else -1.0,
            descriptor.all_mean_excess_return if descriptor.all_mean_excess_return is not None else -1.0,
            descriptor.all_t_statistic if descriptor.all_t_statistic is not None else -10.0,
        ]
        return float(sum(metrics))

    @classmethod
    def _best_model_text(cls, descriptors: list[CheckpointDescriptor]) -> str:
        ranked = sorted(descriptors, key=cls._global_score, reverse=True)
        best = ranked[0]
        strengths: list[str] = []
        if best.validation_mean_excess_return is not None:
            strengths.append(f"validation excess {best.validation_mean_excess_return:.2%}")
        if best.all_mean_excess_return is not None:
            strengths.append(f"all-sample excess {best.all_mean_excess_return:.2%}")
        if best.all_t_statistic is not None:
            strengths.append(f"t-stat {best.all_t_statistic:.3f}")
        strengths_text = ", ".join(strengths) if strengths else "metadata-led ranking"
        return "\n".join(
            [
                best.display_name,
                "",
                f"Type: {best.model_type_label}",
                f"Group: {best.model_group_label}",
                f"Tag: {best.quality_tag}",
                f"Horizon: {best.duration_key or 'custom'}",
                f"Portfolio Size: {best.supported_portfolio_size}",
                f"Best because it leads on {strengths_text}.",
                f"Source path: {best.path.parent.as_posix()}",
            ]
        )

    @staticmethod
    def _gradient_color(score: float) -> QColor:
        clamped = max(0.0, min(1.0, score))
        red = int(214 - (214 - 74) * clamped)
        green = int(96 + (170 - 96) * clamped)
        blue = int(96 - (96 - 80) * clamped)
        return QColor(red, green, blue, 180)

    def _populate_table(self) -> None:
        metric_specs: list[tuple[str, Callable[[CheckpointDescriptor], object], bool | None]] = [
            ("Model", lambda descriptor: descriptor.display_name, None),
            ("Group", lambda descriptor: descriptor.model_group_label, None),
            ("Type", lambda descriptor: descriptor.model_type_label, None),
            ("Variant", lambda descriptor: descriptor.variant_label, None),
            ("Tag", lambda descriptor: descriptor.quality_tag, True),
            ("Horizon", lambda descriptor: descriptor.duration_key or "custom", None),
            ("Updated", lambda descriptor: descriptor.updated_label, None),
            ("Portfolio Size", lambda descriptor: descriptor.supported_portfolio_size, None),
            ("Lookback", lambda descriptor: descriptor.lookback_window, None),
            ("Depth", lambda descriptor: descriptor.depth_label, None),
            ("Selected Epoch", lambda descriptor: descriptor.selected_epoch, None),
            ("Validation Mean Excess", lambda descriptor: descriptor.validation_mean_excess_return, True),
            ("All-Sample Mean Excess", lambda descriptor: descriptor.all_mean_excess_return, True),
            ("All-Sample t-statistic", lambda descriptor: descriptor.all_t_statistic, True),
            ("Benchmark", lambda descriptor: descriptor.benchmark_label or "unknown", None),
            ("Universe", lambda descriptor: descriptor.universe_label or "unknown", None),
        ]
        self.table.setRowCount(len(metric_specs))
        for row_index, (metric_name, accessor, higher_is_better) in enumerate(metric_specs):
            metric_item = QTableWidgetItem(metric_name)
            self.table.setItem(row_index, 0, metric_item)
            values = [accessor(descriptor) for descriptor in self._descriptors]
            numeric_values: list[float | None] = []
            if higher_is_better is not None:
                for value in values:
                    if value is None:
                        numeric_values.append(None)
                    elif isinstance(value, str):
                        normalized = value.casefold()
                        if metric_name == "Tag":
                            numeric_values.append({"exploratory": 1.0, "benchmark+": 2.0, "validated": 3.0}.get(normalized, None))
                        else:
                            numeric_values.append(None)
                    else:
                        numeric_values.append(float(value))
                finite_values = [value for value in numeric_values if value is not None]
                min_value = min(finite_values) if finite_values else None
                max_value = max(finite_values) if finite_values else None
            else:
                min_value = None
                max_value = None
            for column_index, value in enumerate(values, start=1):
                if metric_name.endswith("Excess") and value is not None:
                    display = f"{float(value):.2%}"
                elif metric_name == "All-Sample t-statistic" and value is not None:
                    display = f"{float(value):.3f}"
                else:
                    display = "—" if value is None else str(value)
                item = QTableWidgetItem(display)
                if higher_is_better is not None:
                    numeric_value = numeric_values[column_index - 1]
                    if numeric_value is None or min_value is None or max_value is None:
                        item.setBackground(QColor(110, 110, 110, 110))
                    elif math.isclose(max_value, min_value):
                        item.setBackground(QColor(156, 156, 110, 120))
                    else:
                        score = (numeric_value - min_value) / (max_value - min_value)
                        if not higher_is_better:
                            score = 1.0 - score
                        item.setBackground(self._gradient_color(score))
                self.table.setItem(row_index, column_index, item)
        self.table.resizeColumnsToContents()


class CompareModelPickerDialog(QDialog):
    """Choose any two to five models for comparison."""

    def __init__(
        self,
        *,
        descriptors: list[CheckpointDescriptor],
        preselected_paths: set[Path] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Choose Models To Compare")
        self.resize(980, 620)
        self.selected_descriptors: list[CheckpointDescriptor] = []

        layout = QVBoxLayout(self)
        intro = QLabel("Select any 2 to 5 models. The comparison view will compute a best overall model and color-code every metric row.", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, 5, self)
        self.table.setHorizontalHeaderLabels(["Model", "Horizon", "Type", "Tag", "Portfolio Size"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._update_controls)
        layout.addWidget(self.table, stretch=1)

        self.status_label = QLabel("Select 2 to 5 models.", self)
        layout.addWidget(self.status_label)

        button_box = QDialogButtonBox(self)
        self.compare_button = button_box.addButton("Compare", QDialogButtonBox.ButtonRole.AcceptRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self.compare_button.clicked.connect(self._accept_selection)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

        self._descriptors = list(descriptors)
        selected_paths = preselected_paths or set()
        self.table.setRowCount(len(self._descriptors))
        for row_index, descriptor in enumerate(self._descriptors):
            values = [
                descriptor.display_name,
                descriptor.duration_key or "custom",
                descriptor.model_type_label,
                descriptor.quality_tag,
                str(descriptor.supported_portfolio_size),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, descriptor)
                self.table.setItem(row_index, column_index, item)
        for row_index, descriptor in enumerate(self._descriptors):
            if descriptor.path in selected_paths:
                for column_index in range(self.table.columnCount()):
                    item = self.table.item(row_index, column_index)
                    if item is not None:
                        item.setSelected(True)
        self.table.resizeColumnsToContents()
        self._update_controls()

    def _selected(self) -> list[CheckpointDescriptor]:
        rows = sorted({item.row() for item in self.table.selectedItems()})
        selected: list[CheckpointDescriptor] = []
        for row in rows:
            item = self.table.item(row, 0)
            descriptor = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(descriptor, CheckpointDescriptor):
                selected.append(descriptor)
        return selected

    def _update_controls(self) -> None:
        selected = self._selected()
        if len(selected) < 2:
            self.status_label.setText("Select at least 2 models.")
        elif len(selected) > 5:
            self.status_label.setText("Select no more than 5 models.")
        else:
            self.status_label.setText(f"Ready to compare {len(selected)} models.")
        self.compare_button.setEnabled(2 <= len(selected) <= 5)

    def _accept_selection(self) -> None:
        self.selected_descriptors = self._selected()
        if not 2 <= len(self.selected_descriptors) <= 5:
            return
        self.accept()


class CheckpointSelectionDialog(QDialog):
    """Browse checkpoints as models and pick one for replay."""

    def __init__(
        self,
        *,
        descriptors: list[CheckpointDescriptor],
        selected_descriptor: CheckpointDescriptor | None = None,
        active_duration_key: str = DEFAULT_REPLAY_DURATION_KEY,
        active_max_portfolio_size: int = 10,
        current_portfolio_tickers: list[str] | None = None,
        current_benchmark_ticker: str = "SPY",
        current_start_date: str = "2018-01-01",
        current_end_date: str = "2024-01-01",
        portfolio_library_service: PortfolioLibraryService | None = None,
        refresh_descriptors_callback: Callable[[Path | None], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Model & Training Horizon")
        self.resize(1180, 840)

        self._all_descriptors = list(descriptors)
        self._visible_descriptors: list[CheckpointDescriptor] = []
        self.selected_descriptor: CheckpointDescriptor | None = None
        self.selected_max_portfolio_size = 50 if int(active_max_portfolio_size) > 10 else 10
        self._selected_descriptor = selected_descriptor
        self._current_portfolio_tickers = list(current_portfolio_tickers or [])
        self._current_benchmark_ticker = current_benchmark_ticker
        self._current_start_date = current_start_date
        self._current_end_date = current_end_date
        self._portfolio_library_service = portfolio_library_service
        self._refresh_descriptors_callback = refresh_descriptors_callback
        self._new_model_dialog: NewModelDialog | None = None
        self._new_model_restore_modality = self.windowModality()

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Benchmark+ = excess return, Validated = stability, Exploratory = higher risk. "
            "Choose one model to apply, or open Compare Models to analyze any 2 to 5 saved models.",
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls_row = QHBoxLayout()
        controls_row.addWidget(QLabel("Training Horizon", self))
        self.horizon_combo = QComboBox(self)
        self.horizon_combo.setToolTip("Choose the training horizon group to browse models built for that window.")
        for profile in REPLAY_DURATION_PROFILES:
            self.horizon_combo.addItem(profile.label, profile.key)
        # Backward-compatible handle for tests and legacy callers that still
        # drive horizon changes through a `tab_widget` attribute.
        self.tab_widget = self.horizon_combo
        current_index = max(
            0,
            next((index for index, profile in enumerate(REPLAY_DURATION_PROFILES) if profile.key == active_duration_key), 0),
        )
        self.horizon_combo.setCurrentIndex(current_index)
        self.horizon_combo.currentIndexChanged.connect(self._refresh_table)
        controls_row.addWidget(self.horizon_combo)

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Search by model name, variant, or tag")
        self.search_input.textChanged.connect(self._refresh_table)
        controls_row.addWidget(self.search_input, stretch=1)

        self.portfolio_size_combo = QComboBox(self)
        for size in ALLOWED_PORTFOLIO_SIZES:
            self.portfolio_size_combo.addItem(str(size), size)
        initial_size_index = max(0, self.portfolio_size_combo.findData(self.selected_max_portfolio_size))
        self.portfolio_size_combo.setCurrentIndex(initial_size_index)
        self.portfolio_size_combo.setToolTip("Maximum supported portfolio size for the selected model family.")
        self.portfolio_size_combo.currentIndexChanged.connect(self._on_portfolio_size_changed)
        controls_row.addWidget(self.portfolio_size_combo)

        self.variant_filter = QComboBox(self)
        self.variant_filter.addItem("All Variants", "")
        for variant in sorted({descriptor.variant_label for descriptor in descriptors}):
            self.variant_filter.addItem(variant, variant)
        self.variant_filter.currentIndexChanged.connect(self._refresh_table)
        controls_row.addWidget(self.variant_filter)

        self.type_filter = QComboBox(self)
        self.type_filter.addItem("All Types", "")
        for value in sorted({descriptor.model_type_label for descriptor in descriptors}):
            self.type_filter.addItem(value, value)
        self.type_filter.currentIndexChanged.connect(self._refresh_table)
        controls_row.addWidget(self.type_filter)

        self.source_filter = QComboBox(self)
        self.source_filter.addItem("All Sources", "")
        for value in sorted({descriptor.source_label for descriptor in descriptors}):
            self.source_filter.addItem(value, value)
        self.source_filter.currentIndexChanged.connect(self._refresh_table)
        controls_row.addWidget(self.source_filter)
        layout.addLayout(controls_row)

        content_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(content_splitter, stretch=1)

        left_panel = QWidget(content_splitter)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Models", left_panel))
        self.model_table = QTableWidget(0, 5, left_panel)
        self.model_table.setHorizontalHeaderLabels(["Updated", "Type", "Variant", "Depth", "Tag"])
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.model_table.setSortingEnabled(True)
        self.model_table.horizontalHeader().setStretchLastSection(True)
        self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.itemSelectionChanged.connect(self._on_selection_changed)
        self.model_table.itemDoubleClicked.connect(lambda _item: self._accept_if_selection())
        left_layout.addWidget(self.model_table, stretch=1)
        content_splitter.addWidget(left_panel)

        right_panel = QWidget(content_splitter)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.summary_group = QGroupBox("Summary", right_panel)
        self.summary_group_layout = QVBoxLayout(self.summary_group)
        self.summary_label = QLabel("Select a model to inspect details.", self.summary_group)
        self.summary_label.setWordWrap(True)
        self.summary_group_layout.addWidget(self.summary_label)
        right_layout.addWidget(self.summary_group)

        self.performance_group = QGroupBox("Performance", right_panel)
        self.performance_layout = QVBoxLayout(self.performance_group)
        self.performance_label = QLabel("—", self.performance_group)
        self.performance_label.setWordWrap(True)
        self.performance_layout.addWidget(self.performance_label)
        right_layout.addWidget(self.performance_group)

        self.characteristics_group = QGroupBox("Characteristics", right_panel)
        self.characteristics_layout = QVBoxLayout(self.characteristics_group)
        self.characteristics_label = QLabel("—", self.characteristics_group)
        self.characteristics_label.setWordWrap(True)
        self.characteristics_layout.addWidget(self.characteristics_label)
        right_layout.addWidget(self.characteristics_group)

        self.behavior_group = QGroupBox("Behavior", right_panel)
        self.behavior_layout = QVBoxLayout(self.behavior_group)
        self.behavior_label = QLabel("—", self.behavior_group)
        self.behavior_label.setWordWrap(True)
        self.behavior_layout.addWidget(self.behavior_label)
        right_layout.addWidget(self.behavior_group)

        self.visuals_group = QGroupBox("Visuals", right_panel)
        self.visuals_layout = QVBoxLayout(self.visuals_group)
        self.preview_canvas = ModelPreviewCanvas(self.visuals_group)
        self.visuals_layout.addWidget(self.preview_canvas)
        right_layout.addWidget(self.visuals_group, stretch=1)
        content_splitter.addWidget(right_panel)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 2)
        content_splitter.setSizes([708, 472])

        button_box = QDialogButtonBox(self)
        self.new_model_button = button_box.addButton("New Model", QDialogButtonBox.ButtonRole.ActionRole)
        self.compare_button = button_box.addButton("Compare Models", QDialogButtonBox.ButtonRole.ActionRole)
        self.select_button = button_box.addButton("Select Model", QDialogButtonBox.ButtonRole.AcceptRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        self.new_model_button.clicked.connect(self._open_new_model_dialog)
        self.compare_button.clicked.connect(self._show_comparison)
        self.select_button.clicked.connect(self._accept_if_selection)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)

        self._refresh_table()
        if self._selected_descriptor is not None:
            self._select_descriptor(self._selected_descriptor)
        self._on_selection_changed()

    def _active_duration_key(self) -> str:
        return str(self.horizon_combo.currentData() or DEFAULT_REPLAY_DURATION_KEY)

    def _active_portfolio_size(self) -> int:
        return int(self.portfolio_size_combo.currentData() or 10)

    def _on_portfolio_size_changed(self) -> None:
        self.selected_max_portfolio_size = self._active_portfolio_size()
        self._refresh_table()

    def _refresh_table(self) -> None:
        horizon_key = self._active_duration_key()
        selected_portfolio_size = self._active_portfolio_size()
        search = self.search_input.text().strip().casefold()
        variant_filter = str(self.variant_filter.currentData() or "")
        type_filter = str(self.type_filter.currentData() or "")
        source_filter = str(self.source_filter.currentData() or "")

        def _matches(descriptor: CheckpointDescriptor) -> bool:
            if descriptor.duration_key != horizon_key:
                return False
            if descriptor.uses_placeholder_tickers:
                if descriptor.slot_count != selected_portfolio_size:
                    return False
            elif descriptor.slot_count > selected_portfolio_size:
                return False
            if variant_filter and descriptor.variant_label != variant_filter:
                return False
            if type_filter and descriptor.model_type_label != type_filter:
                return False
            if source_filter and descriptor.source_label != source_filter:
                return False
            if search:
                haystack = " ".join(
                    [
                        descriptor.display_name,
                        descriptor.variant_label,
                        descriptor.quality_tag,
                        descriptor.model_type_label,
                        descriptor.source_label,
                    ]
                ).casefold()
                if search not in haystack:
                    return False
            return True

        self._visible_descriptors = [descriptor for descriptor in self._all_descriptors if _matches(descriptor)]
        self.model_table.setSortingEnabled(False)
        self.model_table.setRowCount(len(self._visible_descriptors))
        for row_index, descriptor in enumerate(self._visible_descriptors):
            row_values = [
                descriptor.updated_label,
                descriptor.model_type_label,
                descriptor.variant_label,
                descriptor.depth_label,
                descriptor.quality_tag,
            ]
            for column_index, value in enumerate(row_values):
                item = SortableTableWidgetItem(value)
                if column_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, descriptor)
                    item.setData(SortableTableWidgetItem.SORT_ROLE, descriptor.updated_sort_value)
                elif column_index == 3:
                    item.setData(SortableTableWidgetItem.SORT_ROLE, descriptor.hidden_dim * 1_000_000 + descriptor.attention_heads * 1_000 + descriptor.attention_layers)
                elif column_index == 4:
                    item.setData(
                        SortableTableWidgetItem.SORT_ROLE,
                        {"Exploratory": 1, "Benchmark+": 2, "Validated": 3}.get(descriptor.quality_tag, 0),
                    )
                else:
                    item.setData(SortableTableWidgetItem.SORT_ROLE, value)
                self.model_table.setItem(row_index, column_index, item)

        self.model_table.setSortingEnabled(True)
        self.model_table.sortItems(0, Qt.SortOrder.DescendingOrder)
        if self.model_table.rowCount() > 0:
            preferred_path = self.selected_descriptor.path if self.selected_descriptor is not None else None
            selected_row = 0
            if preferred_path is not None:
                for row_index in range(self.model_table.rowCount()):
                    item = self.model_table.item(row_index, 0)
                    descriptor = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
                    if isinstance(descriptor, CheckpointDescriptor) and descriptor.path == preferred_path:
                        selected_row = row_index
                        break
            self.model_table.selectRow(selected_row)
            self._on_selection_changed()
        if self.model_table.rowCount() == 0:
            self.summary_label.setText("No model is available for the active filters.")
            self.performance_label.setText("—")
            self.characteristics_label.setText("—")
            self.behavior_label.setText("—")
            self.preview_canvas.set_descriptor(None)
            self.selected_descriptor = None
            self.select_button.setEnabled(False)

    def _selected_descriptors(self) -> list[CheckpointDescriptor]:
        rows = sorted({item.row() for item in self.model_table.selectedItems()})
        descriptors: list[CheckpointDescriptor] = []
        for row in rows:
            item = self.model_table.item(row, 0)
            descriptor = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(descriptor, CheckpointDescriptor):
                descriptors.append(descriptor)
        return descriptors

    def _select_descriptor(self, descriptor: CheckpointDescriptor) -> None:
        if descriptor.duration_key is None:
            return
        horizon_index = next(
            (index for index, profile in enumerate(REPLAY_DURATION_PROFILES) if profile.key == descriptor.duration_key),
            0,
        )
        self.horizon_combo.setCurrentIndex(horizon_index)
        for row_index in range(self.model_table.rowCount()):
            item = self.model_table.item(row_index, 0)
            visible = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(visible, CheckpointDescriptor) and visible.path == descriptor.path:
                self.model_table.selectRow(row_index)
                break

    def _on_selection_changed(self) -> None:
        selected = self._selected_descriptors()
        self.select_button.setEnabled(len(selected) == 1)
        self.selected_descriptor = selected[0] if len(selected) == 1 else None
        self._update_details(selected)

    def _update_details(self, selected: list[CheckpointDescriptor]) -> None:
        descriptor = selected[0] if selected else None
        if descriptor is None:
            self.summary_label.setText("Select a model to inspect details.")
            self.performance_label.setText("—")
            self.characteristics_label.setText("—")
            self.behavior_label.setText("—")
            self.preview_canvas.set_descriptor(None)
            return
        self.summary_label.setText(
            "\n".join(
                [
                    f"Name: {descriptor.display_name}",
                    f"Training Horizon: {descriptor.duration_key or 'custom'}",
                    f"Updated: {descriptor.updated_label}",
                    f"Data Window: {descriptor.data_window_label}",
                    f"Source: {descriptor.source_label}",
                ]
            )
        )
        self.performance_label.setText(
            "\n".join(
                [
                    f"Excess vs SPY: {descriptor.all_mean_excess_return:.2%}" if descriptor.all_mean_excess_return is not None else "Excess vs SPY: —",
                    f"Sharpe Proxy: {descriptor.validation_mean_excess_return:.2%}" if descriptor.validation_mean_excess_return is not None else "Sharpe Proxy: —",
                    "Max Drawdown: preview chart only",
                    f"t-stat: {descriptor.all_t_statistic:.3f}" if descriptor.all_t_statistic is not None else "t-stat: —",
                ]
            )
        )
        holdings_scope = (
            f"Synthetic {descriptor.slot_count}-slot inference"
            if descriptor.uses_placeholder_tickers
            else f"{len(descriptor.tickers)}-ticker policy"
        )
        self.characteristics_label.setText(
            "\n".join(
                [
                    f"Lookback Window: {descriptor.lookback_window} observations",
                    f"Depth: {descriptor.depth_label}",
                    f"Holdings Scope: {holdings_scope}",
                    "Portfolio Style: long-only, fully invested",
                    "Turnover: depends on the live backtest interval and asset basket",
                ]
            )
        )
        self.behavior_label.setText(self._behavior_text(descriptor))
        self.preview_canvas.set_descriptor(descriptor)

    @staticmethod
    def _behavior_text(descriptor: CheckpointDescriptor) -> str:
        if descriptor.quality_tag == "Validated":
            return (
                f"{descriptor.variant_label} emphasizes stability for the {descriptor.duration_key or 'custom'} horizon. "
                "Use it when you want smoother, more dependable benchmark-relative behavior."
            )
        if descriptor.quality_tag == "Benchmark+":
            return (
                f"{descriptor.variant_label} showed positive excess-return evidence versus SPY. "
                "Use it when benchmark-relative upside matters more than strict validation stability."
            )
        return (
            f"{descriptor.variant_label} is exploratory. It may capture interesting behavior, "
            "but it carries a higher risk of unstable performance outside the training window."
        )

    def _show_comparison(self) -> None:
        preselected_paths = {descriptor.path for descriptor in self._selected_descriptors()}
        chooser = CompareModelPickerDialog(
            descriptors=self._all_descriptors,
            preselected_paths=preselected_paths,
            parent=self,
        )
        if not chooser.exec():
            return
        selected = chooser.selected_descriptors
        if not 2 <= len(selected) <= 5:
            QMessageBox.information(self, "Compare Models", "Select any 2 to 5 models to compare them.")
            return
        dialog = ModelComparisonDialog(descriptors=selected, parent=self)
        dialog.exec()

    def _open_new_model_dialog(self) -> None:
        if self._new_model_dialog is not None:
            self._new_model_dialog.show()
            self._new_model_dialog.raise_()
            self._new_model_dialog.activateWindow()
            return

        self._new_model_restore_modality = self.windowModality()
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.hide()

        dialog = NewModelDialog(
            current_portfolio_tickers=self._current_portfolio_tickers,
            current_benchmark_ticker=self._current_benchmark_ticker,
            current_duration_key=self._active_duration_key(),
            current_start_date=self._current_start_date,
            current_end_date=self._current_end_date,
            current_max_portfolio_size=self._active_portfolio_size(),
            portfolio_library_service=self._portfolio_library_service,
            refresh_models_callback=self._refresh_models,
            parent=self,
        )
        self._new_model_dialog = dialog
        dialog.finished.connect(self._on_new_model_dialog_closed)
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_new_model_dialog_closed(self, _result: int) -> None:
        dialog = self._new_model_dialog
        self._new_model_dialog = None
        self.setWindowModality(self._new_model_restore_modality)
        self.show()
        self.raise_()
        self.activateWindow()
        if dialog is not None:
            dialog.deleteLater()

    def _refresh_models(self, preferred_model_path: Path | None = None) -> None:
        if self._refresh_descriptors_callback is None:
            return
        refreshed = self._refresh_descriptors_callback(preferred_model_path)
        matched: CheckpointDescriptor | None = None
        descriptors: list[CheckpointDescriptor] | None = None
        if isinstance(refreshed, tuple) and len(refreshed) == 2:
            candidate_descriptors, candidate_match = refreshed
            if isinstance(candidate_descriptors, list):
                descriptors = candidate_descriptors
            if isinstance(candidate_match, CheckpointDescriptor):
                matched = candidate_match
        elif isinstance(refreshed, list):
            descriptors = refreshed
        if descriptors is None:
            return

        self._all_descriptors = list(descriptors)
        if matched is None and preferred_model_path is not None:
            matched = next((descriptor for descriptor in self._all_descriptors if descriptor.path == preferred_model_path), None)
        if matched is not None:
            portfolio_size_index = self.portfolio_size_combo.findData(matched.supported_portfolio_size)
            if portfolio_size_index >= 0:
                self.portfolio_size_combo.setCurrentIndex(portfolio_size_index)
            self._selected_descriptor = matched
        self._refresh_table()
        if matched is not None:
            self._select_descriptor(matched)
            self.selected_descriptor = matched
            self._on_selection_changed()

    def _accept_if_selection(self) -> None:
        if self.selected_descriptor is None:
            return
        self.accept()

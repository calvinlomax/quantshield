"""Dialog for browsing and selecting trained replay models."""

from __future__ import annotations

import math

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
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
from quantshield_app.services import CheckpointDescriptor

ALLOWED_PORTFOLIO_SIZES = (10, 50)


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
    """Side-by-side comparison for up to two model descriptors."""

    def __init__(self, *, descriptors: list[CheckpointDescriptor], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare Models")
        self.resize(780, 420)

        layout = QVBoxLayout(self)
        names = " vs ".join(descriptor.display_name for descriptor in descriptors)
        intro = QLabel(f"Side-by-side comparison for {names}", self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        table = QTableWidget(10, len(descriptors) + 1, self)
        table.setHorizontalHeaderLabels(["Metric", *[descriptor.display_name for descriptor in descriptors]])
        metrics = [
            ("Group", [descriptor.model_group_label for descriptor in descriptors]),
            ("Type", [descriptor.model_type_label for descriptor in descriptors]),
            ("Variant", [descriptor.variant_label for descriptor in descriptors]),
            ("Tag", [descriptor.quality_tag for descriptor in descriptors]),
            ("Updated", [descriptor.updated_label for descriptor in descriptors]),
            ("Horizon", [descriptor.duration_key or "custom" for descriptor in descriptors]),
            ("Depth", [descriptor.depth_label for descriptor in descriptors]),
            (
                "Excess vs SPY",
                [
                    f"{descriptor.all_mean_excess_return:.2%}" if descriptor.all_mean_excess_return is not None else "—"
                    for descriptor in descriptors
                ],
            ),
            (
                "t-stat",
                [f"{descriptor.all_t_statistic:.3f}" if descriptor.all_t_statistic is not None else "—" for descriptor in descriptors],
            ),
            ("Lookback", [str(descriptor.lookback_window) for descriptor in descriptors]),
        ]
        for row_index, (metric_name, values) in enumerate(metrics):
            table.setItem(row_index, 0, QTableWidgetItem(metric_name))
            for column_index, value in enumerate(values, start=1):
                table.setItem(row_index, column_index, QTableWidgetItem(value))
        layout.addWidget(table)

        button_box = QDialogButtonBox(self)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
        close_button.clicked.connect(self.reject)
        layout.addWidget(button_box)


class CheckpointSelectionDialog(QDialog):
    """Browse checkpoints as models and pick one for replay."""

    def __init__(
        self,
        *,
        descriptors: list[CheckpointDescriptor],
        selected_descriptor: CheckpointDescriptor | None = None,
        active_duration_key: str = DEFAULT_REPLAY_DURATION_KEY,
        active_max_portfolio_size: int = 10,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Model & Training Horizon")
        self.resize(1180, 700)

        self._all_descriptors = list(descriptors)
        self._visible_descriptors: list[CheckpointDescriptor] = []
        self.selected_descriptor: CheckpointDescriptor | None = None
        self.selected_max_portfolio_size = 50 if int(active_max_portfolio_size) > 10 else 10
        self._selected_descriptor = selected_descriptor

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Benchmark+ = excess return, Validated = stability, Exploratory = higher risk. "
            "Choose one model to apply, or select up to two to compare side by side.",
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
        self.model_table = QTableWidget(0, 4, left_panel)
        self.model_table.setHorizontalHeaderLabels(["Type", "Variant", "Depth", "Tag"])
        self.model_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self.model_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
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
        content_splitter.setSizes([520, 640])

        button_box = QDialogButtonBox(self)
        self.compare_button = button_box.addButton("Compare Models", QDialogButtonBox.ButtonRole.ActionRole)
        self.select_button = button_box.addButton("Select Model", QDialogButtonBox.ButtonRole.AcceptRole)
        close_button = button_box.addButton("Close", QDialogButtonBox.ButtonRole.RejectRole)
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
        self.model_table.clearSelection()
        self.model_table.setRowCount(len(self._visible_descriptors))
        for row_index, descriptor in enumerate(self._visible_descriptors):
            row_values = [
                descriptor.model_type_label,
                descriptor.variant_label,
                descriptor.depth_label,
                descriptor.quality_tag,
            ]
            for column_index, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                if column_index == 0:
                    item.setData(Qt.ItemDataRole.UserRole, descriptor)
                self.model_table.setItem(row_index, column_index, item)

        self.model_table.resizeColumnsToContents()
        if self.model_table.rowCount() > 0:
            preferred_path = self.selected_descriptor.path if self.selected_descriptor is not None else None
            selected_row = next(
                (
                    row_index
                    for row_index, descriptor in enumerate(self._visible_descriptors)
                    if preferred_path is not None and descriptor.path == preferred_path
                ),
                0,
            )
            self.model_table.selectRow(selected_row)
            self._on_selection_changed()
        if self.model_table.rowCount() == 0:
            self.summary_label.setText("No model is available for the active filters.")
            self.performance_label.setText("—")
            self.characteristics_label.setText("—")
            self.behavior_label.setText("—")
            self.preview_canvas.set_descriptor(None)
            self.selected_descriptor = None
            self.compare_button.setEnabled(False)
            self.select_button.setEnabled(False)

    def _selected_descriptors(self) -> list[CheckpointDescriptor]:
        rows = sorted({item.row() for item in self.model_table.selectedItems()})
        descriptors: list[CheckpointDescriptor] = []
        for row in rows:
            item = self.model_table.item(row, 0)
            descriptor = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if isinstance(descriptor, CheckpointDescriptor):
                descriptors.append(descriptor)
        return descriptors[:2]

    def _select_descriptor(self, descriptor: CheckpointDescriptor) -> None:
        if descriptor.duration_key is None:
            return
        horizon_index = next(
            (index for index, profile in enumerate(REPLAY_DURATION_PROFILES) if profile.key == descriptor.duration_key),
            0,
        )
        self.horizon_combo.setCurrentIndex(horizon_index)
        for row_index, visible in enumerate(self._visible_descriptors):
            if visible.path == descriptor.path:
                self.model_table.selectRow(row_index)
                break

    def _on_selection_changed(self) -> None:
        selected = self._selected_descriptors()
        if len(selected) > 2:
            self._trim_selection_to_two()
            selected = self._selected_descriptors()
        self.compare_button.setEnabled(len(selected) == 2)
        self.select_button.setEnabled(len(selected) == 1)
        self.selected_descriptor = selected[0] if len(selected) == 1 else None
        self._update_details(selected)

    def _trim_selection_to_two(self) -> None:
        rows = sorted({item.row() for item in self.model_table.selectedItems()})
        for row in rows[2:]:
            self.model_table.selectRow(row)

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
        selected = self._selected_descriptors()
        if len(selected) != 2:
            QMessageBox.information(self, "Compare Models", "Select exactly two models to compare them.")
            return
        dialog = ModelComparisonDialog(descriptors=selected, parent=self)
        dialog.exec()

    def _accept_if_selection(self) -> None:
        if self.selected_descriptor is None:
            return
        self.accept()

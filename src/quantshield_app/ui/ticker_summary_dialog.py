"""Ticker summary popup for the portfolio editor."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from quantshield_app.services import TickerSummary


class PriceHistoryCanvas(FigureCanvasQTAgg):
    """Compact recent-price chart used in the company summary popup."""

    def __init__(self, parent: QWidget | None = None) -> None:
        figure = Figure(figsize=(4.3, 2.5))
        super().__init__(figure)
        self.setParent(parent)
        self.figure = figure
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.15, right=0.97, bottom=0.22, top=0.86)
        self.clear()

    def clear(self) -> None:
        self.axes.clear()
        self.axes.text(0.5, 0.5, "No recent price history", ha="center", va="center")
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.draw_idle()

    def set_history(self, symbol: str, price_history: list[tuple[str, float]]) -> None:
        self.axes.clear()
        if not price_history:
            self.clear()
            return

        history_frame = pd.DataFrame(price_history, columns=["date", "price"])
        history_frame["date"] = pd.to_datetime(history_frame["date"])
        history_frame["price"] = pd.to_numeric(history_frame["price"])

        dates = history_frame["date"].to_list()
        prices = history_frame["price"].to_list()
        self.axes.plot(dates, prices, color="#2563eb", linewidth=1.8)
        self.axes.scatter(dates[-1], prices[-1], color="#1d4ed8", s=18, zorder=3)
        self.axes.grid(alpha=0.18)
        self.axes.set_title(f"Recent Price: {symbol}", fontsize=9)
        self.axes.tick_params(axis="y", labelsize=8)
        self.axes.tick_params(axis="x", labelsize=8, rotation=0)
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)

        tick_dates = [dates[0], dates[len(dates) // 2], dates[-1]] if len(dates) >= 3 else dates
        self.axes.set_xticks(tick_dates)
        self.axes.set_xticklabels([pd.Timestamp(value).strftime("%m/%d") for value in tick_dates])
        self.draw_idle()


class TickerSummaryDialog(QDialog):
    """Display a general yfinance summary for a selected ticker."""

    def __init__(self, *, summary: TickerSummary, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"About {summary.symbol}")
        self.resize(1040, 640)

        layout = QVBoxLayout(self)

        header_row = QHBoxLayout()
        title = QLabel(f"{summary.symbol}  {summary.name}", self)
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        header_row.addWidget(title, stretch=1)

        yahoo_button = QPushButton(self)
        yahoo_button.setText("Open in Yahoo Finance")
        yahoo_button.setToolTip("Open this company or fund on Yahoo Finance")
        yahoo_button.setIcon(self._yahoo_icon())
        yahoo_button.setIconSize(QSize(22, 22))
        yahoo_button.setFixedHeight(38)
        yahoo_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(summary.yahoo_finance_url)))
        header_row.addWidget(yahoo_button, stretch=0)
        layout.addLayout(header_row)

        top_row = QHBoxLayout()

        profile_column = QVBoxLayout()
        profile_label = QLabel("Company Profile", self)
        profile_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        profile_column.addWidget(profile_label)

        details = QLabel("\n".join(summary.detail_lines), self)
        details.setWordWrap(True)
        details.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        profile_column.addWidget(details, stretch=1)
        top_row.addLayout(profile_column, stretch=5)

        analytics_column = QVBoxLayout()
        price_label = QLabel("Recent Price", self)
        price_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        analytics_column.addWidget(price_label)

        price_canvas = PriceHistoryCanvas(self)
        price_canvas.setMinimumHeight(210)
        price_canvas.set_history(summary.symbol, summary.price_history)
        analytics_column.addWidget(price_canvas)

        table_label = QLabel("Technicals & Analyst Ratings", self)
        table_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        analytics_column.addWidget(table_label)

        stats_table = QTableWidget(len(summary.statistics_rows), 3, self)
        stats_table.setHorizontalHeaderLabels(["Group", "Metric", "Value"])
        stats_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        stats_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        stats_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        stats_table.verticalHeader().setVisible(False)
        stats_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        stats_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        stats_table.setAlternatingRowColors(True)
        for row_index, (group, metric, value) in enumerate(summary.statistics_rows):
            group_item = QTableWidgetItem(group)
            metric_item = QTableWidgetItem(metric)
            value_item = QTableWidgetItem(value)
            group_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            metric_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            value_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            stats_table.setItem(row_index, 0, group_item)
            stats_table.setItem(row_index, 1, metric_item)
            stats_table.setItem(row_index, 2, value_item)
        analytics_column.addWidget(stats_table, stretch=1)

        top_row.addLayout(analytics_column, stretch=4)
        layout.addLayout(top_row, stretch=1)

        description_label = QLabel("Business Summary", self)
        description_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(description_label)

        summary_text = QPlainTextEdit(self)
        summary_text.setReadOnly(True)
        summary_text.setPlainText(summary.description)
        layout.addWidget(summary_text, stretch=1)

    @staticmethod
    def _yahoo_icon() -> QIcon:
        asset_path = Path(__file__).resolve().parents[3] / "assets" / "yf.png"
        return QIcon(str(asset_path))

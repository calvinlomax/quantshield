"""Popup dialog for searching and adding tickers to the desktop app universe."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from quantshield_app.services import TickerSearchService, TickerSuggestion


class TickerSearchDialog(QDialog):
    """Search dialog with per-letter suggestions backed by yfinance and local fallbacks."""

    def __init__(
        self,
        *,
        search_service: TickerSearchService | None = None,
        selected_tickers: list[str] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Ticker")
        self.resize(720, 460)
        self.search_service = search_service or TickerSearchService()
        self.selected_tickers = self._normalize_tickers(selected_tickers or [])

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Search a ticker symbol or fund name. Add as many as you want, then choose Save and Close."))

        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("Type a ticker or name, for example VOO or NVIDIA")
        layout.addWidget(self.search_input)

        picker_layout = QGridLayout()
        picker_layout.addWidget(QLabel("Suggestions"), 0, 0)
        picker_layout.addWidget(QLabel("Selected Portfolio Tickers"), 0, 2)

        self.results_list = QListWidget(self)
        picker_layout.addWidget(self.results_list, 1, 0)

        middle_button_column = QVBoxLayout()
        self.add_selected_button = QPushButton("Add ->", self)
        self.add_selected_button.clicked.connect(self._add_selected_item)
        self.use_typed_button = QPushButton("Add Typed", self)
        self.use_typed_button.clicked.connect(self._add_typed_symbol)
        self.remove_selected_button = QPushButton("<- Remove", self)
        self.remove_selected_button.clicked.connect(self._remove_selected_tickers)
        middle_button_column.addStretch(1)
        middle_button_column.addWidget(self.add_selected_button)
        middle_button_column.addWidget(self.use_typed_button)
        middle_button_column.addWidget(self.remove_selected_button)
        middle_button_column.addStretch(1)
        picker_layout.addLayout(middle_button_column, 1, 1)

        self.selected_list = QListWidget(self)
        picker_layout.addWidget(self.selected_list, 1, 2)
        layout.addLayout(picker_layout, stretch=1)

        button_row = QHBoxLayout()
        self.selection_count_label = QLabel()
        button_row.addWidget(self.selection_count_label)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.button_box = QDialogButtonBox(self)
        self.save_and_close_button = self.button_box.addButton("Save and Close", QDialogButtonBox.ButtonRole.AcceptRole)
        self.cancel_button = self.button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self._accept_and_close)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self.results_list.itemDoubleClicked.connect(lambda _item: self._add_selected_item())

        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.timeout.connect(self._refresh_results)
        self.search_input.textChanged.connect(self._queue_refresh)

        self._refresh_selected_list()
        self._refresh_results()

    def _queue_refresh(self) -> None:
        self._search_timer.start(175)

    @staticmethod
    def _normalize_tickers(tickers: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for ticker in tickers:
            upper = ticker.strip().upper()
            if upper and upper not in seen:
                normalized.append(upper)
                seen.add(upper)
        return normalized

    def _refresh_results(self) -> None:
        suggestions = self.search_service.search(self.search_input.text(), limit=12)
        self.results_list.clear()
        for suggestion in suggestions:
            item = QListWidgetItem(suggestion.display_text)
            item.setData(Qt.ItemDataRole.UserRole, suggestion.symbol)
            self.results_list.addItem(item)
        if self.results_list.count() > 0:
            self.results_list.setCurrentRow(0)

    def _refresh_selected_list(self) -> None:
        self.selected_list.clear()
        for ticker in self.selected_tickers:
            self.selected_list.addItem(QListWidgetItem(ticker))
        self.selection_count_label.setText(f"Selected: {len(self.selected_tickers)}")

    def _add_ticker(self, ticker: str) -> None:
        normalized = ticker.strip().upper()
        if not normalized or normalized in self.selected_tickers:
            return
        self.selected_tickers.append(normalized)
        self._refresh_selected_list()

    def _add_selected_item(self) -> None:
        item = self.results_list.currentItem()
        if item is None:
            self._add_typed_symbol()
            return
        self._add_ticker(str(item.data(Qt.ItemDataRole.UserRole)).upper())

    def _add_typed_symbol(self) -> None:
        typed_symbol = self.search_input.text().strip().upper()
        if not typed_symbol:
            return
        self._add_ticker(typed_symbol)

    def _remove_selected_tickers(self) -> None:
        to_remove = {item.text().strip().upper() for item in self.selected_list.selectedItems()}
        if not to_remove:
            return
        self.selected_tickers = [ticker for ticker in self.selected_tickers if ticker not in to_remove]
        self._refresh_selected_list()

    def _accept_and_close(self) -> None:
        self.selected_tickers = self._normalize_tickers(self.selected_tickers)
        self.accept()

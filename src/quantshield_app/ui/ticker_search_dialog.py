"""Popup dialog for searching and adding tickers to the desktop app universe."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from quantshield_app.services import PortfolioLibraryService, TickerInfoService, TickerSearchService, TickerSuggestion
from quantshield_app.ui.portfolio_dialogs import FitModelDialog, LoadPortfolioDialog
from quantshield_app.ui.ticker_summary_dialog import TickerSummaryDialog


class TickerSearchDialog(QDialog):
    """Search dialog with per-letter suggestions backed by yfinance and local fallbacks."""

    def __init__(
        self,
        *,
        search_service: TickerSearchService | None = None,
        info_service: TickerInfoService | None = None,
        portfolio_library_service: PortfolioLibraryService | None = None,
        selected_tickers: list[str] | None = None,
        default_tickers: list[str] | None = None,
        benchmark_ticker: str = "SPY",
        duration_key: str = "1y",
        start_date: str | None = None,
        end_date: str | None = None,
        minimum_count: int = 5,
        max_count: int = 10,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Change Portfolio")
        self.resize(720, 460)
        self.search_service = search_service or TickerSearchService()
        self.info_service = info_service or TickerInfoService()
        self.portfolio_library_service = portfolio_library_service or PortfolioLibraryService()
        self.selected_tickers = self._normalize_tickers(selected_tickers or [])
        self.default_tickers = self._normalize_tickers(default_tickers or [])
        self.benchmark_ticker = benchmark_ticker
        self.duration_key = duration_key
        self.start_date = start_date
        self.end_date = end_date
        self.minimum_count = minimum_count
        self.max_count = max(5, int(max_count))
        self.selected_tickers = self.selected_tickers[: self.max_count]
        self.default_tickers = self.default_tickers[: self.max_count]

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Search a ticker symbol or fund name. Build a portfolio of at least 5 tickers, then choose Save and Close."))

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
        self.summary_button = QPushButton("About Company", self)
        self.summary_button.clicked.connect(self._show_ticker_summary)
        self.remove_selected_button = QPushButton("<- Remove", self)
        self.remove_selected_button.clicked.connect(self._remove_selected_tickers)
        middle_button_column.addStretch(1)
        middle_button_column.addWidget(self.add_selected_button)
        middle_button_column.addWidget(self.use_typed_button)
        middle_button_column.addWidget(self.summary_button)
        middle_button_column.addWidget(self.remove_selected_button)
        middle_button_column.addStretch(1)
        picker_layout.addLayout(middle_button_column, 1, 1)

        self.selected_list = QListWidget(self)
        picker_layout.addWidget(self.selected_list, 1, 2)
        layout.addLayout(picker_layout, stretch=1)

        button_row = QHBoxLayout()
        self.reset_defaults_button = QPushButton("Reset to Defaults", self)
        self.reset_defaults_button.clicked.connect(self._reset_to_defaults)
        button_row.addWidget(self.reset_defaults_button)
        self.presets_button = QPushButton("Presets", self)
        self.presets_button.clicked.connect(self._load_preset_portfolio)
        button_row.addWidget(self.presets_button)
        self.fit_model_button = QPushButton("Fit Model", self)
        self.fit_model_button.clicked.connect(self._open_fit_model_dialog)
        button_row.addWidget(self.fit_model_button)
        self.selection_count_label = QLabel()
        button_row.addWidget(self.selection_count_label)
        button_row.addStretch(1)
        self.save_and_close_button = QPushButton("Save and Close", self)
        self.save_and_close_button.clicked.connect(self._accept_and_close)
        self.close_button = QPushButton("Close", self)
        self.close_button.clicked.connect(self.reject)
        button_row.addWidget(self.save_and_close_button)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

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
        self.selection_count_label.setText(f"Selected: {len(self.selected_tickers)} / {self.minimum_count}+ (Max {self.max_count})")

    def _add_ticker(self, ticker: str) -> None:
        normalized = ticker.strip().upper()
        if not normalized or normalized in self.selected_tickers:
            return
        if len(self.selected_tickers) >= self.max_count:
            QMessageBox.information(
                self,
                "Change Portfolio",
                f"This model mode supports up to {self.max_count} tickers. Remove one before adding another.",
            )
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

    def _selected_summary_symbol(self) -> str | None:
        selected_item = self.selected_list.currentItem()
        if selected_item is not None and selected_item.text().strip():
            return selected_item.text().strip().upper()
        result_item = self.results_list.currentItem()
        if result_item is not None:
            symbol = str(result_item.data(Qt.ItemDataRole.UserRole)).strip().upper()
            if symbol:
                return symbol
        typed_symbol = self.search_input.text().strip().upper()
        return typed_symbol or None

    def _show_ticker_summary(self) -> None:
        symbol = self._selected_summary_symbol()
        if not symbol:
            QMessageBox.warning(self, "About Company", "Select a ticker or type a symbol first.")
            return
        try:
            summary = self.info_service.fetch_summary(symbol)
        except Exception as exc:
            QMessageBox.warning(self, "About Company", str(exc))
            return
        dialog = TickerSummaryDialog(summary=summary, parent=self)
        dialog.exec()

    def _reset_to_defaults(self) -> None:
        if not self.default_tickers:
            return
        self.selected_tickers = list(self.default_tickers)
        self._refresh_selected_list()

    def _load_preset_portfolio(self) -> None:
        presets = self.portfolio_library_service.list_preset_configurations(max_portfolio_size=self.max_count)
        dialog = LoadPortfolioDialog(
            configurations=presets,
            window_title="Portfolio Presets",
            empty_text="No preset portfolios are available.",
            parent=self,
        )
        if dialog.exec() and dialog.selected_configuration is not None:
            self.selected_tickers = list(dialog.selected_configuration.tickers[: self.max_count])
            self._refresh_selected_list()

    def _open_fit_model_dialog(self) -> None:
        dialog = FitModelDialog(
            tickers=self._normalize_tickers(self.selected_tickers),
            benchmark_ticker=self.benchmark_ticker,
            duration_key=self.duration_key,
            start_date=self.start_date,
            end_date=self.end_date,
            parent=self,
        )
        dialog.exec()

    def _accept_and_close(self) -> None:
        self.selected_tickers = self._normalize_tickers(self.selected_tickers)[: self.max_count]
        if len(self.selected_tickers) < self.minimum_count:
            QMessageBox.warning(
                self,
                "Change Portfolio",
                f"Select at least {self.minimum_count} tickers before saving the portfolio.",
            )
            return
        self.accept()

"""Ticker search helpers for the desktop app."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from quantshield.universe import SEARCH_SEED_TICKERS


@dataclass(slots=True)
class TickerSuggestion:
    """Search suggestion shown in the add-ticker popup."""

    symbol: str
    name: str
    exchange: str = ""

    @property
    def display_text(self) -> str:
        suffix = f" ({self.exchange})" if self.exchange else ""
        return f"{self.symbol} — {self.name}{suffix}"


class TickerSearchService:
    """Search yfinance for ticker suggestions with a local fallback seed list."""

    def __init__(self, seed_tickers: Iterable[str] = SEARCH_SEED_TICKERS) -> None:
        self.seed_tickers = [ticker.strip().upper() for ticker in seed_tickers if ticker.strip()]

    def search(self, query: str, *, limit: int = 12) -> list[TickerSuggestion]:
        normalized = query.strip().upper()
        suggestions: list[TickerSuggestion] = []
        seen: set[str] = set()

        def add(symbol: str, name: str, exchange: str = "") -> None:
            upper_symbol = symbol.strip().upper()
            if not upper_symbol or upper_symbol in seen:
                return
            suggestions.append(TickerSuggestion(symbol=upper_symbol, name=name, exchange=exchange))
            seen.add(upper_symbol)

        for ticker in self.seed_tickers:
            if not normalized or normalized in ticker:
                add(ticker, "Seed universe")
            if len(suggestions) >= limit:
                return suggestions[:limit]

        if not normalized:
            return suggestions[:limit]

        try:
            import yfinance as yf

            search = yf.Search(query=normalized, max_results=limit, news_count=0)
            for quote in getattr(search, "quotes", []) or []:
                symbol = str(quote.get("symbol", "")).upper()
                if not symbol:
                    continue
                add(
                    symbol,
                    str(quote.get("shortname") or quote.get("longname") or "yfinance result"),
                    str(quote.get("exchange") or quote.get("exchDisp") or ""),
                )
                if len(suggestions) >= limit:
                    break
        except Exception:
            pass

        if not suggestions:
            add(normalized, "Manual symbol entry")
        return suggestions[:limit]

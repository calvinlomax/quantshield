"""Ticker information lookups for the desktop portfolio editor."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class TickerSummary:
    """Human-readable ticker summary pulled from yfinance."""

    symbol: str
    name: str
    description: str
    detail_lines: list[str]
    yahoo_finance_url: str
    price_history: list[tuple[str, float]]
    statistics_rows: list[tuple[str, str, str]]


class TickerInfoService:
    """Fetch general ticker information for the desktop app."""

    def fetch_summary(self, symbol: str) -> TickerSummary:
        """Return a lightweight summary for a ticker symbol."""
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("Ticker symbol cannot be empty.")

        try:
            import yfinance as yf

            ticker = yf.Ticker(normalized)
            info = getattr(ticker, "info", {}) or {}
            fast_info = getattr(ticker, "fast_info", {}) or {}
        except Exception as exc:
            raise ValueError(f"Could not fetch yfinance summary for {normalized}: {exc}") from exc

        price_history = self._fetch_price_history(ticker)
        analyst_snapshot = self._fetch_analyst_snapshot(ticker)
        current_price = (
            fast_info.get("lastPrice")
            or info.get("currentPrice")
            or info.get("regularMarketPrice")
            or (price_history[-1][1] if price_history else None)
        )

        name = str(info.get("longName") or info.get("shortName") or normalized)
        description = str(
            info.get("longBusinessSummary")
            or info.get("description")
            or info.get("shortBusinessSummary")
            or f"No detailed yfinance summary is available for {normalized}."
        )

        detail_pairs = [
            ("Quote Type", info.get("quoteType")),
            ("Exchange", info.get("exchange") or info.get("fullExchangeName")),
            ("Currency", info.get("currency")),
            ("Current Price", current_price),
            ("Previous Close", fast_info.get("previousClose") or info.get("previousClose")),
            ("Market Cap", fast_info.get("marketCap") or info.get("marketCap")),
            ("Sector / Category", info.get("sector") or info.get("category")),
            ("Industry / Family", info.get("industry") or info.get("fundFamily")),
            ("Country", info.get("country")),
            ("Employees", info.get("fullTimeEmployees")),
            ("Beta", info.get("beta")),
            ("Trailing P/E", info.get("trailingPE")),
            ("Dividend Yield", self._format_ratio(info.get("dividendYield"))),
            ("Expense Ratio", self._format_ratio(info.get("annualReportExpenseRatio") or info.get("expenseRatio"))),
            ("52 Week Range", self._format_range(info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh"))),
            ("Average Volume", fast_info.get("tenDayAverageVolume") or info.get("averageVolume")),
            ("Website", info.get("website")),
        ]
        detail_lines = [f"{label}: {self._format_value(value)}" for label, value in detail_pairs if value not in (None, "", [])]

        return TickerSummary(
            symbol=normalized,
            name=name,
            description=description,
            detail_lines=detail_lines,
            yahoo_finance_url=f"https://finance.yahoo.com/quote/{normalized}",
            price_history=price_history,
            statistics_rows=self._build_statistics_rows(
                info=info,
                fast_info=fast_info,
                current_price=current_price,
                analyst_snapshot=analyst_snapshot,
            ),
        )

    @staticmethod
    def _format_value(value: object) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            absolute = abs(float(value))
            for divisor, suffix in ((1_000_000_000.0, "B"), (1_000_000.0, "M"), (1_000.0, "k")):
                if absolute >= divisor:
                    return f"{float(value) / divisor:.2f}".rstrip("0").rstrip(".") + suffix
            return f"{float(value):,.2f}".rstrip("0").rstrip(".")
        return str(value)

    @staticmethod
    def _format_range(low: object, high: object) -> str | None:
        if low in (None, "") or high in (None, ""):
            return None
        return f"{TickerInfoService._format_value(low)} to {TickerInfoService._format_value(high)}"

    @staticmethod
    def _format_ratio(value: object) -> str | None:
        if value in (None, ""):
            return None
        return f"{float(value):.2%}"

    def _fetch_price_history(self, ticker: object) -> list[tuple[str, float]]:
        try:
            history = ticker.history(period="3mo", interval="1d", auto_adjust=False)
        except Exception:
            return []
        if history is None or history.empty:
            return []
        price_column = "Close" if "Close" in history.columns else ("Adj Close" if "Adj Close" in history.columns else None)
        if price_column is None:
            return []
        series = history[price_column].dropna().tail(60)
        return [(pd.Timestamp(index_value).strftime("%Y-%m-%d"), float(price)) for index_value, price in series.items()]

    @staticmethod
    def _fetch_analyst_snapshot(ticker: object) -> dict[str, object]:
        try:
            summary = getattr(ticker, "recommendations_summary", None)
        except Exception:
            return {}
        if summary is None or getattr(summary, "empty", True):
            return {}
        try:
            latest = summary.iloc[-1].to_dict()
        except Exception:
            return {}
        return dict(latest)

    def _build_statistics_rows(
        self,
        *,
        info: dict[str, object],
        fast_info: dict[str, object],
        current_price: object,
        analyst_snapshot: dict[str, object],
    ) -> list[tuple[str, str, str]]:
        rows: list[tuple[str, str, str]] = []
        technical_pairs = [
            ("Current Price", current_price),
            ("50-Day Average", info.get("fiftyDayAverage")),
            ("200-Day Average", info.get("twoHundredDayAverage")),
            ("52-Week High", info.get("fiftyTwoWeekHigh")),
            ("52-Week Low", info.get("fiftyTwoWeekLow")),
            ("Average Volume", fast_info.get("tenDayAverageVolume") or info.get("averageVolume")),
            ("Market Cap", fast_info.get("marketCap") or info.get("marketCap")),
            ("Beta", info.get("beta")),
            ("Trailing P/E", info.get("trailingPE")),
            ("Forward P/E", info.get("forwardPE")),
        ]
        for label, value in technical_pairs:
            if value in (None, "", []):
                continue
            rows.append(("Technicals", label, self._format_value(value)))

        analyst_pairs = [
            ("Consensus", self._format_recommendation(info.get("recommendationKey"))),
            ("Analyst Opinions", info.get("numberOfAnalystOpinions")),
            ("Target Mean", info.get("targetMeanPrice")),
            ("Target Median", info.get("targetMedianPrice")),
            ("Target High", info.get("targetHighPrice")),
            ("Target Low", info.get("targetLowPrice")),
        ]
        for label, value in analyst_pairs:
            if value in (None, "", []):
                continue
            rows.append(("Analyst Ratings", label, self._format_value(value)))

        target_mean = info.get("targetMeanPrice")
        if current_price not in (None, "", []) and target_mean not in (None, "", []):
            try:
                upside = (float(target_mean) / float(current_price)) - 1.0
            except (TypeError, ValueError, ZeroDivisionError):
                upside = None
            if upside is not None:
                rows.append(("Analyst Ratings", "Upside To Mean Target", self._format_ratio(upside) or ""))

        breakdown_columns = ["strongBuy", "buy", "hold", "sell", "strongSell"]
        breakdown_values = [analyst_snapshot.get(column) for column in breakdown_columns]
        if any(value not in (None, "", []) for value in breakdown_values):
            period = analyst_snapshot.get("period")
            suffix = f" ({period})" if period not in (None, "", []) else ""
            formatted = " / ".join(str(int(value)) if value not in (None, "", []) else "-" for value in breakdown_values)
            rows.append(("Analyst Ratings", f"Strong Buy / Buy / Hold / Sell / Strong Sell{suffix}", formatted))

        return rows

    @staticmethod
    def _format_recommendation(value: object) -> str | None:
        if value in (None, "", []):
            return None
        normalized = str(value).replace("_", " ").strip()
        return normalized.title() if normalized else None

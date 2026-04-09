"""Local persistence for named desktop-app configurations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from quantshield.universe import (
    CANONICAL_LARGE_PORTFOLIO_UNIVERSE_A,
    CANONICAL_LARGE_PORTFOLIO_UNIVERSE_B,
    CANONICAL_LARGE_PORTFOLIO_UNIVERSE_C,
    CANONICAL_TOP_50_UNIVERSE,
    CANONICAL_TOP_ETF_UNIVERSE,
)


DEFAULT_PORTFOLIO_LIBRARY_PATH = Path("outputs/app_state/portfolios.json")


@dataclass(frozen=True, slots=True)
class SavedConfiguration:
    """Named backtest configuration stored for reuse in the desktop app."""

    name: str
    tickers: list[str]
    starting_capital: float | None = None
    benchmark_ticker: str | None = None
    duration_key: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    rebalance_mode: str | None = None
    rebalance_frequency: str | None = None
    model_path: str | None = None
    max_portfolio_size: int | None = None
    source: str = "saved"
    notes: str | None = None


SavedPortfolio = SavedConfiguration


PRESET_CONFIGURATION_DEFINITIONS_10: tuple[tuple[str, list[str], str], ...] = (
    (
        "Technology Leaders",
        ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "NOW", "QCOM"],
        "Large-cap software, semis, and platform companies.",
    ),
    (
        "Financial Compounders",
        ["JPM", "BRK-B", "BAC", "GS", "MS", "BLK", "PGR", "SCHW", "AXP", "C"],
        "Banks, brokers, insurers, and diversified financial franchises.",
    ),
    (
        "Healthcare Quality",
        ["LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "DHR", "ABT", "ISRG", "AMGN"],
        "Pharma, managed care, diagnostics, and medical-device leaders.",
    ),
    (
        "Energy & Industrials",
        ["XOM", "CVX", "SLB", "CAT", "DE", "GE", "ETN", "RTX", "UNP", "HON"],
        "Energy producers plus industrial and capital-equipment names.",
    ),
    (
        "Consumer Staples & Brands",
        ["AMZN", "COST", "WMT", "HD", "MCD", "KO", "PG", "TJX", "SBUX", "NKE"],
        "Retail, household products, and global consumer brands.",
    ),
    (
        "AI & Semiconductors",
        ["NVDA", "AVGO", "AMD", "QCOM", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MRVL"],
        "Semiconductor designers, foundry equipment, and AI infrastructure.",
    ),
)

PRESET_CONFIGURATION_DEFINITIONS_50: tuple[tuple[str, list[str], str], ...] = (
    (
        "Expanded Core 50",
        list(CANONICAL_TOP_50_UNIVERSE),
        "Ten liquid core ETFs plus the long-history 40-name default equity basket.",
    ),
    (
        "Expanded Quality 50",
        [*CANONICAL_TOP_ETF_UNIVERSE, *CANONICAL_LARGE_PORTFOLIO_UNIVERSE_A],
        "Core ETFs paired with an industrial, infrastructure, and quality-growth stock mix.",
    ),
    (
        "Expanded Growth 50",
        [*CANONICAL_TOP_ETF_UNIVERSE, *CANONICAL_LARGE_PORTFOLIO_UNIVERSE_B],
        "Core ETFs paired with a higher-beta growth and quality stock mix.",
    ),
)


class PortfolioLibraryService:
    """Store and load named desktop configurations from a local JSON file."""

    def __init__(self, storage_path: str | Path = DEFAULT_PORTFOLIO_LIBRARY_PATH) -> None:
        self.storage_path = Path(storage_path)

    def list_configurations(self) -> list[SavedConfiguration]:
        """Return all saved configurations sorted by name."""
        payload = self._load_payload()
        raw_configurations = payload.get("configurations") or payload.get("portfolios") or {}
        configurations = [self._configuration_from_payload(name, details) for name, details in raw_configurations.items()]
        return sorted(configurations, key=lambda configuration: configuration.name.casefold())

    def list_preset_configurations(self, *, max_portfolio_size: int = 10) -> list[SavedConfiguration]:
        """Return the built-in preset portfolios available in the desktop app."""
        definitions = PRESET_CONFIGURATION_DEFINITIONS_50 if int(max_portfolio_size) > 10 else PRESET_CONFIGURATION_DEFINITIONS_10
        presets = [
            SavedConfiguration(
                name=name,
                tickers=self._normalize_tickers(tickers),
                benchmark_ticker="SPY",
                duration_key="1y",
                max_portfolio_size=int(max_portfolio_size),
                source="preset",
                notes=notes,
            )
            for name, tickers, notes in definitions
        ]
        return sorted(presets, key=lambda configuration: configuration.name.casefold())

    def save_configuration(
        self,
        name: str,
        tickers: list[str],
        *,
        starting_capital: float | None = None,
        benchmark_ticker: str | None = None,
        duration_key: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        rebalance_mode: str | None = None,
        rebalance_frequency: str | None = None,
        model_path: str | None = None,
        max_portfolio_size: int | None = None,
    ) -> SavedConfiguration:
        """Persist a named configuration, overwriting any existing entry with the same name."""
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Configuration name cannot be empty.")
        normalized_tickers = self._normalize_tickers(tickers)
        if len(normalized_tickers) < 1:
            raise ValueError("A saved configuration must include at least one ticker.")

        configuration = SavedConfiguration(
            name=normalized_name,
            tickers=normalized_tickers,
            starting_capital=float(starting_capital) if starting_capital is not None else None,
            benchmark_ticker=self._normalize_optional_symbol(benchmark_ticker),
            duration_key=self._normalize_optional_text(duration_key),
            start_date=self._normalize_optional_text(start_date),
            end_date=self._normalize_optional_text(end_date),
            rebalance_mode=self._normalize_optional_text(rebalance_mode),
            rebalance_frequency=self._normalize_optional_text(rebalance_frequency),
            model_path=self._normalize_optional_text(model_path),
            max_portfolio_size=self._coerce_optional_int(max_portfolio_size),
            source="saved",
        )

        payload = self._load_payload()
        payload["version"] = 2
        payload.setdefault("configurations", {})[normalized_name] = self._serialize_configuration(configuration)
        if "portfolios" in payload:
            payload.pop("portfolios", None)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return configuration

    def load_configuration(self, name: str) -> SavedConfiguration:
        """Return a saved configuration by name."""
        normalized_name = name.strip()
        for configuration in self.list_configurations():
            if configuration.name == normalized_name:
                return configuration
        raise ValueError(f"Saved configuration '{normalized_name}' was not found.")

    def list_portfolios(self) -> list[SavedConfiguration]:
        """Backward-compatible alias for configuration listing."""
        return self.list_configurations()

    def save_portfolio(self, name: str, tickers: list[str]) -> SavedConfiguration:
        """Backward-compatible alias for saving only ticker selections."""
        return self.save_configuration(name, tickers)

    def load_portfolio(self, name: str) -> SavedConfiguration:
        """Backward-compatible alias for configuration loading."""
        return self.load_configuration(name)

    def _load_payload(self) -> dict:
        if not self.storage_path.exists():
            return {"version": 2, "configurations": {}}
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Configuration library is not valid JSON: {self.storage_path}") from exc

    def _configuration_from_payload(self, name: str, details: object) -> SavedConfiguration:
        payload = details if isinstance(details, dict) else {}
        return SavedConfiguration(
            name=name,
            tickers=self._normalize_tickers(payload.get("tickers", [])),
            starting_capital=self._coerce_optional_float(payload.get("starting_capital")),
            benchmark_ticker=self._normalize_optional_symbol(payload.get("benchmark_ticker")),
            duration_key=self._normalize_optional_text(payload.get("duration_key")),
            start_date=self._normalize_optional_text(payload.get("start_date")),
            end_date=self._normalize_optional_text(payload.get("end_date")),
            rebalance_mode=self._normalize_optional_text(payload.get("rebalance_mode")),
            rebalance_frequency=self._normalize_optional_text(payload.get("rebalance_frequency")),
            model_path=self._normalize_optional_text(payload.get("model_path")),
            max_portfolio_size=self._coerce_optional_int(payload.get("max_portfolio_size")),
            source=self._normalize_optional_text(payload.get("source")) or "saved",
            notes=self._normalize_optional_text(payload.get("notes")),
        )

    @staticmethod
    def _serialize_configuration(configuration: SavedConfiguration) -> dict[str, object]:
        return {
            "tickers": configuration.tickers,
            "starting_capital": configuration.starting_capital,
            "benchmark_ticker": configuration.benchmark_ticker,
            "duration_key": configuration.duration_key,
            "start_date": configuration.start_date,
            "end_date": configuration.end_date,
            "rebalance_mode": configuration.rebalance_mode,
            "rebalance_frequency": configuration.rebalance_frequency,
            "model_path": configuration.model_path,
            "max_portfolio_size": configuration.max_portfolio_size,
            "source": configuration.source,
            "notes": configuration.notes,
        }

    @staticmethod
    def _normalize_tickers(tickers: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for ticker in tickers:
            upper = str(ticker).strip().upper()
            if upper and upper not in seen:
                normalized.append(upper)
                seen.add(upper)
        return normalized

    @staticmethod
    def _normalize_optional_symbol(value: object) -> str | None:
        if value in (None, ""):
            return None
        normalized = str(value).strip().upper()
        return normalized or None

    @staticmethod
    def _normalize_optional_text(value: object) -> str | None:
        if value in (None, ""):
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _coerce_optional_int(value: object) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

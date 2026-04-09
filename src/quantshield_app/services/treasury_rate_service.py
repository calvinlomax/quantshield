"""Duration-matched Treasury rate lookup for replay summaries and metrics."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Callable

import pandas as pd
import requests

from quantshield.utils import ensure_directory

HttpGetter = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class TreasuryMaturityProfile:
    """Treasury maturity used for risk-free assumptions."""

    business_days: int
    column_name: str
    display_name: str


@dataclass(frozen=True, slots=True)
class TreasuryRateAssumption:
    """Resolved risk-free assumption for a replay window."""

    annual_rate: float
    maturity_label: str
    maturity_column_name: str
    source: str
    as_of_date: pd.Timestamp | None
    fallback_used: bool = False


TREASURY_MATURITY_PROFILES: tuple[TreasuryMaturityProfile, ...] = (
    TreasuryMaturityProfile(21, "1 Mo", "1-Month Treasury"),
    TreasuryMaturityProfile(63, "3 Mo", "3-Month Treasury"),
    TreasuryMaturityProfile(126, "6 Mo", "6-Month Treasury"),
    TreasuryMaturityProfile(252, "1 Yr", "1-Year Treasury"),
    TreasuryMaturityProfile(504, "2 Yr", "2-Year Treasury"),
    TreasuryMaturityProfile(756, "3 Yr", "3-Year Treasury"),
    TreasuryMaturityProfile(1260, "5 Yr", "5-Year Treasury"),
)


def _default_http_get(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


class TreasuryRateService:
    """Resolve a duration-matched U.S. Treasury yield for replay analytics."""

    TREASURY_SOURCE = "U.S. Treasury daily par yield curve"

    def __init__(
        self,
        *,
        cache_dir: str | Path = "data/raw/treasury_rates",
        allow_online: bool = False,
        http_get: HttpGetter | None = None,
    ) -> None:
        self.cache_dir = ensure_directory(cache_dir)
        self.allow_online = allow_online
        self.http_get = http_get or _default_http_get

    def resolve_for_window(
        self,
        *,
        business_days: int,
        as_of_date: str | pd.Timestamp,
    ) -> TreasuryRateAssumption:
        """Return the nearest Treasury maturity and annual yield for a replay window."""
        maturity = min(
            TREASURY_MATURITY_PROFILES,
            key=lambda profile: (abs(profile.business_days - int(max(business_days, 1))), profile.business_days),
        )
        as_of_timestamp = pd.Timestamp(as_of_date).normalize()
        try:
            annual_rate, resolved_date = self._lookup_rate(
                column_name=maturity.column_name,
                as_of_date=as_of_timestamp,
            )
            return TreasuryRateAssumption(
                annual_rate=annual_rate,
                maturity_label=maturity.display_name,
                maturity_column_name=maturity.column_name,
                source=self.TREASURY_SOURCE,
                as_of_date=resolved_date,
                fallback_used=False,
            )
        except Exception:
            return TreasuryRateAssumption(
                annual_rate=0.0,
                maturity_label=maturity.display_name,
                maturity_column_name=maturity.column_name,
                source=f"{self.TREASURY_SOURCE} (fallback unavailable)",
                as_of_date=None,
                fallback_used=True,
            )

    def _lookup_rate(self, *, column_name: str, as_of_date: pd.Timestamp) -> tuple[float, pd.Timestamp]:
        for months_back in range(0, 25):
            period = (as_of_date.to_period("M") - months_back)
            month_frame = self._load_monthly_rates(period.strftime("%Y%m"))
            if column_name not in month_frame.columns:
                continue
            if months_back == 0:
                eligible = month_frame.loc[month_frame.index <= as_of_date, column_name].dropna()
                if eligible.empty:
                    eligible = month_frame.loc[month_frame.index >= as_of_date, column_name].dropna().head(1)
            else:
                eligible = month_frame[column_name].dropna()
            if eligible.empty:
                continue
            resolved_date = pd.Timestamp(eligible.index[-1] if len(eligible.index) > 1 else eligible.index[0]).normalize()
            return float(eligible.iloc[-1]) / 100.0, resolved_date
        raise ValueError(f"No Treasury yield data was found for column {column_name}.")

    def _load_monthly_rates(self, year_month: str) -> pd.DataFrame:
        cache_path = self.cache_dir / f"daily_treasury_yield_curve_{year_month}.csv"
        if cache_path.exists():
            frame = pd.read_csv(cache_path, parse_dates=["Date"])
            frame["Date"] = pd.to_datetime(frame["Date"])
            frame = frame.set_index("Date").sort_index()
            for column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            return frame

        if not self.allow_online:
            raise FileNotFoundError(f"No cached Treasury data is available for {year_month}.")

        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value_month={year_month}"
        )
        html = self.http_get(url)
        tables = pd.read_html(StringIO(html))
        yield_table = self._extract_yield_table(tables)
        yield_table.to_csv(cache_path, index_label="Date")
        return yield_table

    @staticmethod
    def _extract_yield_table(tables: list[pd.DataFrame]) -> pd.DataFrame:
        required_columns = {"Date", "1 Mo", "3 Mo", "6 Mo", "1 Yr", "2 Yr", "3 Yr", "5 Yr"}
        for table in tables:
            normalized = table.copy()
            normalized.columns = [TreasuryRateService._normalize_column_name(column) for column in normalized.columns]
            if "Date" not in normalized.columns or not required_columns.issubset(set(normalized.columns)):
                continue
            normalized["Date"] = pd.to_datetime(normalized["Date"], format="%m/%d/%Y", errors="coerce")
            normalized = normalized.dropna(subset=["Date"]).set_index("Date").sort_index()
            for column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
            return normalized
        raise ValueError("Treasury yield table could not be parsed from the Treasury response.")

    @staticmethod
    def _normalize_column_name(column: object) -> str:
        if isinstance(column, tuple):
            parts = [str(part).strip() for part in column if str(part).strip() and str(part).strip().lower() != "nan"]
            return " ".join(parts).replace("\xa0", " ").strip()
        return str(column).replace("\xa0", " ").strip()

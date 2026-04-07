"""Canonical ETF universes and asset-class metadata used across QuantShield."""

from __future__ import annotations

CANONICAL_TOP_ETF_UNIVERSE = [
    "VOO",
    "IVV",
    "SPY",
    "VTI",
    "QQQ",
    "VEA",
    "VUG",
    "GLD",
    "IEFA",
    "VTV",
]

CANONICAL_TOP_ETF_ASSET_CLASS_MAP = {
    "VOO": "equity",
    "IVV": "equity",
    "SPY": "equity",
    "VTI": "equity",
    "QQQ": "equity",
    "VEA": "equity",
    "VUG": "equity",
    "GLD": "commodity",
    "IEFA": "equity",
    "VTV": "equity",
}

SEARCH_SEED_TICKERS = [
    *CANONICAL_TOP_ETF_UNIVERSE,
    "IWM",
    "EFA",
    "EEM",
    "TLT",
    "LQD",
    "VNQ",
    "XLF",
    "XLK",
    "XLE",
    "SMH",
    "DIA",
]

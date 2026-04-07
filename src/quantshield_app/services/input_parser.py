"""Input parsing helpers for the QuantShield desktop app."""

from __future__ import annotations


def parse_ticker_input(raw_text: str) -> list[str]:
    """Parse a comma-separated ticker string into a unique ordered list."""
    tokens = [token.strip().upper() for token in raw_text.replace("\n", ",").split(",")]
    tickers: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token:
            continue
        if token not in seen:
            tickers.append(token)
            seen.add(token)
    if not tickers:
        raise ValueError("Enter at least one ticker.")
    return tickers

from __future__ import annotations

import pandas as pd

from quantshield.data_loader import MarketDataLoader, extract_adjusted_close


def test_extract_adjusted_close_multiindex() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    columns = pd.MultiIndex.from_product([["Adj Close", "Volume"], ["SPY", "QQQ"]])
    frame = pd.DataFrame(
        [
            [100.0, 200.0, 1_000_000, 2_000_000],
            [101.0, 202.0, 1_100_000, 1_900_000],
            [102.0, 203.0, 1_200_000, 1_800_000],
        ],
        index=index,
        columns=columns,
    )
    prices = extract_adjusted_close(frame, ["SPY", "QQQ"])
    assert list(prices.columns) == ["SPY", "QQQ"]
    assert prices.iloc[-1]["SPY"] == 102.0


def test_loader_uses_cache(tmp_path) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="B")
    columns = pd.MultiIndex.from_tuples([("Adj Close", "SPY"), ("Adj Close", "QQQ")])
    raw = pd.DataFrame(
        [[100.0, 200.0], [101.0, 201.0], [102.0, 202.0]],
        index=index,
        columns=columns,
    )

    calls = {"count": 0}

    def provider(**_: object) -> pd.DataFrame:
        calls["count"] += 1
        return raw

    loader = MarketDataLoader(cache_dir=tmp_path, provider=provider)
    first = loader.fetch_prices(["SPY", "QQQ"], "2024-01-01", "2024-01-10", use_cache=True, force_refresh=False)
    second = loader.fetch_prices(["SPY", "QQQ"], "2024-01-01", "2024-01-10", use_cache=True, force_refresh=False)

    assert calls["count"] == 1
    pd.testing.assert_frame_equal(first, second)

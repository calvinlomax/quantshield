"""Fetch and cache market data locally from yfinance."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from quantshield.config import load_config
from quantshield.data_loader import MarketDataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache adjusted close prices from yfinance.")
    parser.add_argument("--config", default="config/default_config.yaml", help="Path to YAML configuration file.")
    parser.add_argument("--tickers", nargs="+", help="Optional ticker override.")
    parser.add_argument("--start-date", help="Optional start date override in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional end date override in YYYY-MM-DD format.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cache and refetch from yfinance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    tickers = args.tickers or config.data.tickers
    start_date = args.start_date or config.data.start_date
    end_date = args.end_date if args.end_date is not None else config.data.end_date
    force_refresh = args.force_refresh or config.data.force_refresh

    loader = MarketDataLoader(cache_dir=config.data.cache_dir)
    prices = loader.fetch_prices(
        tickers,
        start_date,
        end_date,
        use_cache=config.data.use_cache,
        force_refresh=force_refresh,
    )
    cache_path = loader.cache_path(tickers, start_date, end_date)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} assets to {cache_path}")


if __name__ == "__main__":
    main()

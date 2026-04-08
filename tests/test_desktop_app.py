from __future__ import annotations

from dataclasses import asdict
import numpy as np
import pandas as pd
import pytest

from quantshield.data_loader import MarketDataLoader
from quantshield_app.services import CheckpointService, MarketDataService, TickerSearchService, parse_ticker_input
from quantshield_app.services.checkpoint_service import is_placeholder_ticker
from quantshield_app.services.replay_service import ReplayFrame, ReplayService
from quantshield_app.viewmodels import ReplayController

torch = pytest.importorskip("torch")

from quantshield.rl import CrossAssetAttentionActorCritic, RLTrainingConfig  # noqa: E402


def _mock_price_download(**_: object) -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=320, freq="B")
    data = pd.DataFrame(
        {
            ("Adj Close", "SPY"): np.linspace(100.0, 150.0, len(dates)),
            ("Adj Close", "QQQ"): np.linspace(120.0, 210.0, len(dates)),
            ("Adj Close", "GLD"): np.linspace(90.0, 105.0, len(dates)),
        },
        index=dates,
    )
    data.columns = pd.MultiIndex.from_tuples(data.columns)
    return data


def _sample_frames() -> list[ReplayFrame]:
    base_weights = pd.Series({"SPY": 0.5, "QQQ": 0.3, "GLD": 0.2})
    return [
        ReplayFrame(
            index=0,
            date=pd.Timestamp("2024-01-05"),
            portfolio_value=101_000.0,
            benchmark_value=100_400.0,
            portfolio_return=0.0100,
            benchmark_return=0.0040,
            excess_return=0.0060,
            turnover=0.0,
            rebalanced=True,
            weights=base_weights,
        ),
        ReplayFrame(
            index=1,
            date=pd.Timestamp("2024-01-12"),
            portfolio_value=101_500.0,
            benchmark_value=100_650.0,
            portfolio_return=0.0050,
            benchmark_return=0.0025,
            excess_return=0.0025,
            turnover=0.0,
            rebalanced=False,
            weights=base_weights,
        ),
        ReplayFrame(
            index=2,
            date=pd.Timestamp("2024-01-19"),
            portfolio_value=102_500.0,
            benchmark_value=101_000.0,
            portfolio_return=0.0099,
            benchmark_return=0.0035,
            excess_return=0.0064,
            turnover=0.08,
            rebalanced=True,
            weights=base_weights,
        ),
    ]


def test_parse_ticker_input_normalizes_and_dedupes() -> None:
    parsed = parse_ticker_input(" spy, qqq\nGLD , SPY ")
    assert parsed == ["SPY", "QQQ", "GLD"]


def test_ticker_search_service_filters_seed_universe() -> None:
    service = TickerSearchService(seed_tickers=["VOO", "IVV", "SPY", "GLD"])
    suggestions = service.search("vo", limit=5)

    assert suggestions
    assert suggestions[0].symbol == "VOO"


def test_prepare_market_data_requires_sufficient_lookback(tmp_path) -> None:
    loader = MarketDataLoader(cache_dir=tmp_path / "raw", provider=_mock_price_download)
    service = MarketDataService(loader)

    with pytest.raises(ValueError, match="Not enough pre-start history"):
        service.prepare_market_data(
            portfolio_tickers=["SPY", "QQQ", "GLD"],
            benchmark_ticker="SPY",
            start_date="2023-02-01",
            end_date="2023-06-30",
            lookback_window=80,
        )


def test_prepare_market_data_builds_replay_returns(tmp_path) -> None:
    loader = MarketDataLoader(cache_dir=tmp_path / "raw", provider=_mock_price_download)
    service = MarketDataService(loader)

    prepared = service.prepare_market_data(
        portfolio_tickers=["SPY", "QQQ", "GLD"],
        benchmark_ticker="SPY",
        start_date="2023-07-03",
        end_date="2023-12-29",
        lookback_window=63,
    )

    assert not prepared.replay_returns.empty
    assert list(prepared.replay_returns.columns) == ["SPY", "QQQ", "GLD"]
    assert prepared.replay_returns.index.min() >= pd.Timestamp("2023-07-03")


def test_prepare_market_data_rejects_placeholder_tickers(tmp_path) -> None:
    loader = MarketDataLoader(cache_dir=tmp_path / "raw", provider=_mock_price_download)
    service = MarketDataService(loader)

    with pytest.raises(ValueError, match="Synthetic checkpoint asset slots cannot be downloaded"):
        service.prepare_market_data(
            portfolio_tickers=["ASSET_01", "ASSET_02", "ASSET_03", "ASSET_04", "ASSET_05"],
            benchmark_ticker="SPY",
            start_date="2023-07-03",
            end_date="2023-12-29",
            lookback_window=63,
        )


def test_checkpoint_service_discovers_and_loads_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "outputs" / "rl_policy"
    checkpoint_dir.mkdir(parents=True)
    tickers = ["SPY", "QQQ", "GLD"]
    config = RLTrainingConfig(
        lookback_window=63,
        hidden_dim=32,
        attention_heads=4,
        attention_layers=1,
    )
    model = CrossAssetAttentionActorCritic(
        num_assets=len(tickers),
        lookback_window=config.lookback_window,
        feature_dim=3,
        hidden_dim=config.hidden_dim,
        attention_heads=config.attention_heads,
        attention_layers=config.attention_layers,
        dropout=config.dropout,
    )
    checkpoint_path = checkpoint_dir / "actor_critic_policy.pt"
    torch.save(
        {
            "tickers": tickers,
            "training_config": asdict(config),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    service = CheckpointService(search_roots=[checkpoint_dir])
    descriptors = service.discover_checkpoints()

    assert len(descriptors) == 1
    assert descriptors[0].tickers == tickers
    assert descriptors[0].hidden_dim == 32

    loaded = service.load_checkpoint(checkpoint_path, device="cpu")
    assert loaded.tickers == tickers
    assert loaded.training_config.attention_layers == 1


def test_checkpoint_descriptor_marks_placeholder_slots(tmp_path) -> None:
    checkpoint_dir = tmp_path / "outputs" / "rl_policy"
    checkpoint_dir.mkdir(parents=True)
    tickers = [f"ASSET_{index:02d}" for index in range(1, 11)]
    config = RLTrainingConfig()
    model = CrossAssetAttentionActorCritic(
        num_assets=len(tickers),
        lookback_window=config.lookback_window,
        feature_dim=3,
        hidden_dim=config.hidden_dim,
        attention_heads=config.attention_heads,
        attention_layers=config.attention_layers,
        dropout=config.dropout,
    )
    checkpoint_path = checkpoint_dir / "actor_critic_policy.pt"
    torch.save(
        {
            "tickers": tickers,
            "training_config": asdict(config),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    service = CheckpointService(search_roots=[checkpoint_dir])
    descriptor = service.discover_checkpoints()[0]

    assert descriptor.uses_placeholder_tickers is True
    assert descriptor.inference_default_tickers[0] == "VOO"
    assert "synthetic-10-slot-policy" in descriptor.display_name
    assert is_placeholder_ticker("ASSET_07") is True


def test_replay_service_validates_minimum_ticker_count() -> None:
    ordered = ReplayService._validate_portfolio_tickers(["GLD", "QQQ", "SPY", "VOO", "IVV"])
    assert ordered == ["GLD", "QQQ", "SPY", "VOO", "IVV"]

    with pytest.raises(ValueError, match="Select at least 5 tickers"):
        ReplayService._validate_portfolio_tickers(["SPY", "QQQ", "GLD", "IVV"])


def test_replay_controller_step_and_scrub_logic() -> None:
    controller = ReplayController()
    frames = _sample_frames()
    controller.set_frames(frames)

    assert controller.current_frame().index == 0
    assert controller.step_forward().index == 1
    assert controller.step_forward().index == 2
    assert controller.step_forward().index == 2
    assert controller.step_backward().index == 1
    assert controller.scrub_to(99).index == 2
    assert controller.scrub_to(-10).index == 0

from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest

from quantshield.data_loader import MarketDataLoader
from quantshield.metrics import sharpe_ratio
from quantshield.replay_durations import checkpoint_root_for_duration, duration_end_from_start, duration_start_from_end
from quantshield.utils import generate_schedule
from quantshield_app.services import CheckpointService, MarketDataService, PortfolioLibraryService, TickerInfoService, TickerSearchService, parse_ticker_input
from quantshield_app.services.checkpoint_service import DEFAULT_CHECKPOINT_ROOTS, is_placeholder_ticker
from quantshield_app.services.replay_service import PolicyReplayResult, ReplayFrame, ReplayService
from quantshield_app.services.treasury_rate_service import TreasuryRateAssumption, TreasuryRateService
from quantshield_app.viewmodels import ReplayController

torch = pytest.importorskip("torch")

from quantshield.rl import CrossAssetAttentionActorCritic, RLTrainingConfig, build_policy_state, predict_policy_weights  # noqa: E402


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


def test_duration_helpers_preserve_profile_business_day_span() -> None:
    end_date = pd.Timestamp("2024-01-31")
    start_date = duration_start_from_end(end_date, "1mo")

    assert len(pd.date_range(start_date, end_date, freq="B")) == 21
    assert duration_end_from_start(start_date, "1mo") == end_date


def test_generate_schedule_accepts_legacy_month_alias() -> None:
    index = pd.bdate_range("2024-01-02", "2024-06-28")

    legacy = generate_schedule(index, "M")
    normalized = generate_schedule(index, "ME")

    assert legacy.equals(normalized)
    assert len(legacy) > 0


def test_treasury_rate_service_matches_nearest_maturity_offline() -> None:
    service = TreasuryRateService(allow_online=False)
    assumption = service.resolve_for_window(
        business_days=760,
        as_of_date="2024-01-02",
    )

    assert assumption.maturity_label == "3-Year Treasury"
    assert assumption.fallback_used is True
    assert assumption.annual_rate == 0.0


def test_ticker_search_service_filters_seed_universe() -> None:
    service = TickerSearchService(seed_tickers=["VOO", "IVV", "SPY", "GLD"])
    suggestions = service.search("vo", limit=5)

    assert suggestions
    assert suggestions[0].symbol == "VOO"


def test_ticker_search_service_random_portfolio_uses_real_symbols(monkeypatch) -> None:
    service = TickerSearchService(seed_tickers=["VOO", "IVV", "SPY", "GLD", "QQQ", "VTI"])
    monkeypatch.setattr(
        "quantshield.sp500_random_training.fetch_sp500_constituents",
        lambda: ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "BRK-B", "JPM", "LLY", "XOM", "COST", "UNH"],
    )

    portfolio = service.random_portfolio(size=5, seed=3)

    assert len(portfolio) == 5
    assert all(not ticker.startswith("ASSET_") for ticker in portfolio)


def test_ticker_info_service_includes_price_history_and_statistics(monkeypatch) -> None:
    history_index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    history_frame = pd.DataFrame({"Close": [100.0, 102.5, 101.75]}, index=history_index)
    recommendations_frame = pd.DataFrame(
        [
            {
                "period": "0m",
                "strongBuy": 12,
                "buy": 8,
                "hold": 3,
                "sell": 1,
                "strongSell": 0,
            }
        ]
    )

    class _FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol
            self.info = {
                "longName": "Example Corp",
                "longBusinessSummary": "Example business summary.",
                "quoteType": "EQUITY",
                "currency": "USD",
                "sector": "Technology",
                "industry": "Software",
                "country": "United States",
                "beta": 1.12,
                "trailingPE": 24.5,
                "forwardPE": 21.2,
                "fiftyDayAverage": 98.25,
                "twoHundredDayAverage": 91.75,
                "fiftyTwoWeekHigh": 123.4,
                "fiftyTwoWeekLow": 74.2,
                "recommendationKey": "strong_buy",
                "numberOfAnalystOpinions": 24,
                "targetMeanPrice": 118.0,
                "targetMedianPrice": 116.5,
                "targetHighPrice": 135.0,
                "targetLowPrice": 95.0,
                "marketCap": 1_250_000_000,
            }
            self.fast_info = {
                "lastPrice": 101.75,
                "previousClose": 100.8,
                "tenDayAverageVolume": 2_500_000,
                "marketCap": 1_250_000_000,
            }
            self.recommendations_summary = recommendations_frame

        def history(self, period: str, interval: str, auto_adjust: bool) -> pd.DataFrame:
            assert period == "3mo"
            assert interval == "1d"
            assert auto_adjust is False
            return history_frame

    fake_yfinance = types.SimpleNamespace(Ticker=_FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)

    summary = TickerInfoService().fetch_summary("exm")

    assert summary.symbol == "EXM"
    assert summary.price_history[-1] == ("2024-01-04", 101.75)
    assert any(
        group == "Technicals" and metric == "50-Day Average" and value == "98.25"
        for group, metric, value in summary.statistics_rows
    )
    assert any(
        group == "Analyst Ratings" and metric == "Consensus" and value == "Strong Buy"
        for group, metric, value in summary.statistics_rows
    )
    assert any(
        group == "Analyst Ratings" and metric.startswith("Strong Buy / Buy / Hold / Sell / Strong Sell")
        for group, metric, _value in summary.statistics_rows
    )


def test_portfolio_library_service_exposes_presets(tmp_path) -> None:
    service = PortfolioLibraryService(storage_path=tmp_path / "portfolios.json")

    presets = service.list_preset_configurations()

    assert len(presets) >= 5
    assert all(configuration.source == "preset" for configuration in presets)
    assert all(len(configuration.tickers) == 10 for configuration in presets)
    assert any(configuration.name == "Technology Leaders" for configuration in presets)


def test_prepare_market_data_allows_warmup_inside_selected_window(tmp_path) -> None:
    loader = MarketDataLoader(cache_dir=tmp_path / "raw", provider=_mock_price_download)
    service = MarketDataService(loader)

    prepared = service.prepare_market_data(
        portfolio_tickers=["SPY", "QQQ", "GLD"],
        benchmark_ticker="SPY",
        start_date="2023-02-01",
        end_date="2023-06-30",
        lookback_window=80,
    )

    assert not prepared.replay_returns.empty
    assert prepared.replay_returns.index.min() >= pd.Timestamp("2023-02-01")


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


def test_replay_service_daily_schedule_updates_portfolio_value(tmp_path, monkeypatch) -> None:
    def _five_ticker_download(**_: object) -> pd.DataFrame:
        dates = pd.date_range("2023-01-02", periods=320, freq="B")
        data = pd.DataFrame(
            {
                ("Adj Close", "SPY"): np.linspace(100.0, 150.0, len(dates)),
                ("Adj Close", "QQQ"): np.linspace(120.0, 210.0, len(dates)),
                ("Adj Close", "GLD"): np.linspace(90.0, 105.0, len(dates)),
                ("Adj Close", "VTI"): np.linspace(110.0, 160.0, len(dates)),
                ("Adj Close", "VEA"): np.linspace(40.0, 55.0, len(dates)),
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data

    loader = MarketDataLoader(cache_dir=tmp_path / "raw", provider=_five_ticker_download)
    market_data = MarketDataService(loader).prepare_market_data(
        portfolio_tickers=["SPY", "QQQ", "GLD", "VTI", "VEA"],
        benchmark_ticker="SPY",
        start_date="2023-02-01",
        end_date="2023-06-30",
        lookback_window=20,
    )

    dummy_checkpoint = types.SimpleNamespace(training_config=types.SimpleNamespace(lookback_window=20))

    monkeypatch.setattr(
        "quantshield_app.services.replay_service.predict_policy_weights",
        lambda checkpoint, window, *, tickers=None: pd.Series(
            np.full(len(tickers or window.columns), 1.0 / len(tickers or window.columns)),
            index=list(tickers or window.columns),
        ),
    )

    result = ReplayService().build_replay(
        checkpoint=dummy_checkpoint,
        market_data=market_data,
        rebalance_frequency="B",
        starting_capital=100_000.0,
    )

    observed_values = [frame.portfolio_value for frame in result.frames[:5]]
    assert len(result.frames) > 5
    assert any(not np.isclose(value, observed_values[0]) for value in observed_values[1:])
    assert "equal_weight" in result.comparison_returns.columns
    assert "Equal Weight" in result.cumulative_values.columns
    assert "equal_weight_total_return" in result.metrics
    assert "active_vs_equal_weight_total_return" in result.metrics


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


def test_checkpoint_service_loads_nearest_neighbor_checkpoint(tmp_path) -> None:
    checkpoint_dir = tmp_path / "outputs" / "model_experiments" / "1y" / "oracle_memory"
    checkpoint_dir.mkdir(parents=True)
    tickers = ["ASSET_01", "ASSET_02", "ASSET_03"]
    config = RLTrainingConfig(
        lookback_window=4,
        hidden_dim=160,
        attention_heads=4,
        attention_layers=3,
    )
    returns_window = pd.DataFrame(
        {
            "ASSET_01": [0.01, 0.02, -0.01, 0.03],
            "ASSET_02": [0.00, 0.01, 0.02, -0.01],
            "ASSET_03": [-0.02, 0.00, 0.01, 0.01],
        },
        index=pd.bdate_range("2024-01-02", periods=4),
    )
    stored_state = build_policy_state(returns_window, tickers=tickers, lookback_window=config.lookback_window)
    stored_action = np.asarray([[0.7, 0.2, 0.1]], dtype=np.float32)
    checkpoint_path = checkpoint_dir / "actor_critic_policy.pt"
    torch.save(
        {
            "policy_kind": "nearest_neighbor",
            "tickers": tickers,
            "training_config": asdict(config),
            "nearest_neighbor_states": torch.as_tensor(stored_state[None, ...], dtype=torch.float32),
            "nearest_neighbor_actions": torch.as_tensor(stored_action, dtype=torch.float32),
        },
        checkpoint_path,
    )

    service = CheckpointService(search_roots=[tmp_path / "outputs"])
    descriptors = service.discover_checkpoints(duration_key="1y")

    assert len(descriptors) == 1
    assert descriptors[0].tickers == tickers
    assert descriptors[0].duration_key == "1y"

    loaded = service.load_checkpoint(checkpoint_path, device="cpu")
    weights = predict_policy_weights(loaded, returns_window, tickers=tickers)
    assert float(weights.sum()) == pytest.approx(1.0)
    assert weights["ASSET_01"] == pytest.approx(0.7)

    reduced_window = returns_window.loc[:, ["ASSET_01", "ASSET_02"]]
    reduced_weights = predict_policy_weights(loaded, reduced_window, tickers=list(reduced_window.columns))
    assert float(reduced_weights.sum()) == pytest.approx(1.0)
    assert reduced_weights["ASSET_01"] == pytest.approx(0.7 / 0.9)


def test_checkpoint_service_reads_portfolio_fit_metadata(tmp_path) -> None:
    checkpoint_dir = tmp_path / "outputs" / "portfolio_model_fits" / "1y" / "demo_fit"
    checkpoint_dir.mkdir(parents=True)
    tickers = ["SPY", "QQQ", "GLD", "VTI", "VEA"]
    config = RLTrainingConfig(
        lookback_window=63,
        hidden_dim=224,
        attention_heads=8,
        attention_layers=4,
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
    torch.save(
        {
            "tickers": tickers,
            "training_config": asdict(config),
            "state_dict": model.state_dict(),
            "duration_key": "1y",
        },
        checkpoint_dir / "actor_critic_policy.pt",
    )
    (checkpoint_dir / "portfolio_fit_summary.txt").write_text(
        "Selected candidate: portfolio_oracle_blend_224x8x4\nSelected epoch: 17\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "name": "demo_fit",
                "duration_key": "1y",
                "selected_candidate": "portfolio_oracle_blend_224x8x4",
                "selected_epoch": 17,
            }
        ]
    ).to_csv(checkpoint_dir / "fit_metadata.csv", index=False)
    pd.DataFrame(
        {
            "Split": ["validation", "all"],
            "policy_mean_excess_return": [0.0125, 0.0130],
            "t_statistic": [2.8, 3.1],
            "significant_outperformance": [True, True],
        }
    ).to_csv(checkpoint_dir / "benchmark_summary.csv", index=False)

    service = CheckpointService(search_roots=[tmp_path / "outputs"])
    descriptors = service.discover_checkpoints(duration_key="1y")

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.duration_key == "1y"
    assert descriptor.source_label == "Portfolio Fit"
    assert descriptor.model_type_label == "Fit Model"
    assert descriptor.variant_label == "Oracle Blend"
    assert descriptor.selected_epoch == 17


def test_replay_service_uses_treasury_rate_assumption(monkeypatch, tmp_path) -> None:
    def _five_ticker_download(**_: object) -> pd.DataFrame:
        dates = pd.date_range("2023-01-02", periods=220, freq="B")
        data = pd.DataFrame(
            {
                ("Adj Close", "SPY"): np.linspace(100.0, 150.0, len(dates)),
                ("Adj Close", "QQQ"): np.linspace(120.0, 210.0, len(dates)),
                ("Adj Close", "GLD"): np.linspace(90.0, 105.0, len(dates)),
                ("Adj Close", "VTI"): np.linspace(110.0, 160.0, len(dates)),
                ("Adj Close", "VEA"): np.linspace(40.0, 55.0, len(dates)),
            },
            index=dates,
        )
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        return data

    loader = MarketDataLoader(cache_dir=tmp_path / "raw", provider=_five_ticker_download)
    market_data = MarketDataService(loader).prepare_market_data(
        portfolio_tickers=["SPY", "QQQ", "GLD", "VTI", "VEA"],
        benchmark_ticker="SPY",
        start_date="2023-02-01",
        end_date="2023-06-30",
        lookback_window=20,
    )

    dummy_checkpoint = types.SimpleNamespace(training_config=types.SimpleNamespace(lookback_window=20))

    monkeypatch.setattr(
        "quantshield_app.services.replay_service.predict_policy_weights",
        lambda checkpoint, window, *, tickers=None: pd.Series(
            np.full(len(tickers or window.columns), 1.0 / len(tickers or window.columns)),
            index=list(tickers or window.columns),
        ),
    )

    stub_assumption = TreasuryRateAssumption(
        annual_rate=0.0425,
        maturity_label="6-Month Treasury",
        maturity_column_name="6 Mo",
        source="Test Treasury Source",
        as_of_date=pd.Timestamp("2023-02-01"),
        fallback_used=False,
    )

    class StubTreasuryRateService:
        def resolve_for_window(self, *, business_days: int, as_of_date: str | pd.Timestamp) -> TreasuryRateAssumption:
            assert business_days == len(market_data.replay_returns.index)
            assert pd.Timestamp(as_of_date) == pd.Timestamp("2023-02-01")
            return stub_assumption

    result = ReplayService(treasury_rate_service=StubTreasuryRateService()).build_replay(
        checkpoint=dummy_checkpoint,
        market_data=market_data,
        rebalance_frequency="B",
        starting_capital=100_000.0,
    )

    periods_per_year = 252
    expected_sharpe = sharpe_ratio(
        result.comparison_returns["portfolio"],
        risk_free_rate=stub_assumption.annual_rate,
        periods_per_year=periods_per_year,
    )

    assert result.risk_free_assumption == stub_assumption
    assert result.metrics["risk_free_rate"] == pytest.approx(0.0425)
    assert result.summary_table.loc["portfolio", "sharpe_ratio"] == pytest.approx(expected_sharpe)


def test_checkpoint_descriptor_uses_readable_display_name(tmp_path) -> None:
    checkpoint_dir = tmp_path / "outputs" / "replay_checkpoint_suites" / "1mo"
    checkpoint_dir.mkdir(parents=True)
    tickers = ["SPY", "QQQ", "GLD"]
    config = RLTrainingConfig(
        lookback_window=5,
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
    torch.save(
        {
            "tickers": tickers,
            "training_config": asdict(config),
            "state_dict": model.state_dict(),
        },
        checkpoint_dir / "actor_critic_policy.pt",
    )
    (checkpoint_dir / "random_sp500_training_summary.txt").write_text(
        "Selected candidate: balanced_192x6x4\nSelected epoch: 88\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "Split": "validation",
                "samples": 20,
                "benchmark_mean_raw_return": 0.001,
                "policy_mean_raw_return": 0.002,
                "policy_mean_excess_return": 0.001,
                "t_statistic": 2.0,
                "p_value": 0.04,
                "significant_outperformance": True,
            },
            {
                "Split": "all",
                "samples": 50,
                "benchmark_mean_raw_return": 0.001,
                "policy_mean_raw_return": 0.003,
                "policy_mean_excess_return": 0.002,
                "t_statistic": 3.5,
                "p_value": 0.001,
                "significant_outperformance": True,
            },
        ]
    ).to_csv(checkpoint_dir / "benchmark_summary.csv", index=False)

    service = CheckpointService(search_roots=[checkpoint_dir])
    descriptor = service.discover_checkpoints()[0]

    assert descriptor.display_name == "Tactical 1-Month Balanced (Validated)"
    assert str(checkpoint_dir) not in descriptor.display_name
    assert "excess 0.200%" in descriptor.display_subtitle


def test_checkpoint_service_filters_by_duration(tmp_path) -> None:
    tickers = ["SPY", "QQQ", "GLD"]
    config = RLTrainingConfig(
        lookback_window=16,
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

    roots = []
    for duration_key in ["1mo", "1y"]:
        root = tmp_path / Path(checkpoint_root_for_duration(duration_key))
        root.mkdir(parents=True)
        roots.append(root)
        torch.save(
            {
                "tickers": tickers,
                "training_config": asdict(config),
                "state_dict": model.state_dict(),
            },
            root / "actor_critic_policy.pt",
        )

    service = CheckpointService(search_roots=roots)
    descriptors = service.discover_checkpoints(duration_key="1mo")

    assert descriptors
    assert descriptors[0].duration_key == "1mo"


def test_checkpoint_service_discovers_all_default_duration_suites(tmp_path) -> None:
    tickers = ["SPY", "QQQ", "GLD"]
    config = RLTrainingConfig(
        lookback_window=16,
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

    roots = []
    for duration_key in ["1mo", "3mo", "1y"]:
        root = tmp_path / Path(checkpoint_root_for_duration(duration_key))
        root.mkdir(parents=True)
        roots.append(root)
        torch.save(
            {
                "tickers": tickers,
                "training_config": asdict(config),
                "state_dict": model.state_dict(),
            },
            root / "actor_critic_policy.pt",
        )

    service = CheckpointService(search_roots=DEFAULT_CHECKPOINT_ROOTS)
    service.search_roots = roots
    service._uses_default_roots = True
    descriptors = service.discover_checkpoints(duration_key="1mo")

    assert {descriptor.duration_key for descriptor in descriptors} == {"1mo", "3mo", "1y"}
    assert descriptors[0].duration_key == "1mo"


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
    assert "synthetic 10-slot inference" in descriptor.display_subtitle
    assert is_placeholder_ticker("ASSET_07") is True


def test_portfolio_library_service_round_trip(tmp_path) -> None:
    service = PortfolioLibraryService(storage_path=tmp_path / "portfolios.json")

    saved = service.save_portfolio("Tech Mix", ["msft", "aapl", "nvda", "amzn", "meta", "msft"])
    loaded = service.load_portfolio("Tech Mix")

    assert saved.tickers == ["MSFT", "AAPL", "NVDA", "AMZN", "META"]
    assert loaded.tickers == ["MSFT", "AAPL", "NVDA", "AMZN", "META"]
    assert [portfolio.name for portfolio in service.list_portfolios()] == ["Tech Mix"]


def test_ticker_info_service_formats_yfinance_summary(monkeypatch) -> None:
    class FakeTicker:
        info = {
            "longName": "Apple Inc.",
            "longBusinessSummary": "Consumer electronics company.",
            "quoteType": "EQUITY",
            "exchange": "NMS",
            "currency": "USD",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "website": "https://www.apple.com",
        }
        fast_info = {
            "previousClose": 187.12,
            "marketCap": 2_950_000_000_000,
        }

    fake_module = types.SimpleNamespace(Ticker=lambda _symbol: FakeTicker())
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)

    summary = TickerInfoService().fetch_summary("aapl")

    assert summary.symbol == "AAPL"
    assert summary.name == "Apple Inc."
    assert summary.yahoo_finance_url.endswith("/AAPL")
    assert any("Market Cap: 2950B" in line or "Market Cap: 2950.00B" in line for line in summary.detail_lines)


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


def test_main_window_defers_close_while_worker_thread_is_running(tmp_path, monkeypatch) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    from quantshield_app.ui.main_window import QuantShieldMainWindow

    app = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    window = QuantShieldMainWindow(checkpoint_service=CheckpointService(search_roots=[tmp_path / "checkpoints"]))

    messages: list[str] = []
    monkeypatch.setattr(window, "_show_loading_dialog", lambda: None)
    monkeypatch.setattr(window, "_append_loading_log", messages.append)

    class FakeThread:
        def isRunning(self) -> bool:
            return True

    class FakeEvent:
        ignored = False

        def ignore(self) -> None:
            self.ignored = True

    window._worker_thread = FakeThread()  # type: ignore[assignment]
    event = FakeEvent()

    window.closeEvent(event)

    assert event.ignored is True
    assert window._close_after_worker is True
    assert any("Finishing replay preparation before closing the app" in message for message in messages)
    window.deleteLater()
    app.processEvents()


def test_checkpoint_selection_dialog_tracks_visible_tab_selection() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    from quantshield_app.services.checkpoint_service import CheckpointDescriptor
    from quantshield_app.ui.checkpoint_dialog import CheckpointSelectionDialog

    app = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    one_month = CheckpointDescriptor(
        path=Path("outputs/replay_checkpoint_suites/1mo/actor_critic_policy.pt"),
        tickers=["SPY", "QQQ", "GLD"],
        lookback_window=5,
        hidden_dim=192,
        attention_heads=6,
        attention_layers=4,
        duration_key="1mo",
        candidate_name="balanced_192x6x4",
    )
    one_year = CheckpointDescriptor(
        path=Path("outputs/replay_checkpoint_suites/1y/actor_critic_policy.pt"),
        tickers=["SPY", "QQQ", "GLD"],
        lookback_window=63,
        hidden_dim=224,
        attention_heads=8,
        attention_layers=5,
        duration_key="1y",
        candidate_name="deeper_224x8x5",
    )

    dialog = CheckpointSelectionDialog(
        descriptors=[one_month, one_year],
        selected_descriptor=one_month,
        active_duration_key="1mo",
    )

    assert dialog.selected_descriptor == one_month
    dialog.tab_widget.setCurrentIndex(3)
    app.processEvents()
    assert dialog.selected_descriptor == one_year
    dialog.deleteLater()
    app.processEvents()


def test_speed_display_combo_box_uses_closed_speed_suffix() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    from quantshield_app.ui.main_window import SpeedDisplayComboBox

    app = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])
    combo = SpeedDisplayComboBox()
    combo.addItems(["0.5x", "1x", "2x", "4x", "8x"])
    combo.setCurrentText("2x")

    assert combo.itemText(2) == "2x"
    assert combo.closed_display_text() == "2x Speed"

    combo.deleteLater()
    app.processEvents()


def test_equity_curve_canvas_resets_drawdown_axis_between_results() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qtwidgets = pytest.importorskip("PySide6.QtWidgets")
    from quantshield_app.ui.charts import EquityCurveCanvas

    app = qtwidgets.QApplication.instance() or qtwidgets.QApplication([])

    comparison_returns = pd.DataFrame(
        {
            "portfolio": [0.01, -0.02, 0.015],
            "equal_weight": [0.008, -0.01, 0.012],
            "benchmark": [0.009, -0.015, 0.011],
            "excess": [0.001, -0.005, 0.004],
            "active_vs_equal_weight": [0.002, -0.01, 0.003],
        },
        index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
    )
    cumulative_values = pd.DataFrame(
        {
            "Portfolio": [101000.0, 98980.0, 100464.7],
            "Equal Weight": [100800.0, 99792.0, 100989.5],
            "Benchmark": [100900.0, 99386.5, 100479.8],
        },
        index=comparison_returns.index,
    )
    result = PolicyReplayResult(
        checkpoint=types.SimpleNamespace(path=Path("dummy.pt")),
        frames=_sample_frames(),
        prices=pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=comparison_returns.index),
        comparison_returns=comparison_returns,
        cumulative_values=cumulative_values,
        weights_history=pd.DataFrame(),
        daily_weights=pd.DataFrame(
            {
                "SPY": [0.5, 0.5, 0.5],
                "QQQ": [0.3, 0.3, 0.3],
                "GLD": [0.2, 0.2, 0.2],
            },
            index=comparison_returns.index,
        ),
        asset_returns=pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015],
                "QQQ": [0.008, -0.01, 0.012],
                "GLD": [0.004, -0.006, 0.005],
            },
            index=comparison_returns.index,
        ),
        benchmark_returns=comparison_returns["benchmark"],
        summary_table=pd.DataFrame(),
        metrics={},
        requested_tickers=["SPY", "QQQ", "GLD"],
        benchmark_ticker="SPY",
        starting_capital=100000.0,
        rebalance_frequency="B",
        rebalance_label="1D",
        rebalance_mode="manual",
        estimated_steps=3,
    )

    canvas = EquityCurveCanvas()
    canvas.set_drawdown_visible(True)
    canvas.set_result(result)
    assert len(canvas.figure.axes) == 2

    canvas.set_result(result)
    assert len(canvas.figure.axes) == 2
    app.processEvents()


def test_replay_summary_dialog_reports_duration_matched_risk_free_assumption() -> None:
    from quantshield_app.ui.portfolio_dialogs import ReplaySummaryDialog

    comparison_returns = pd.DataFrame(
        {
            "portfolio": [0.01, -0.02, 0.015],
            "equal_weight": [0.008, -0.01, 0.012],
            "benchmark": [0.009, -0.015, 0.011],
            "excess": [0.001, -0.005, 0.004],
            "active_vs_equal_weight": [0.002, -0.01, 0.003],
        },
        index=pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19"]),
    )
    cumulative_values = pd.DataFrame(
        {
            "Portfolio": [101000.0, 98980.0, 100464.7],
            "Equal Weight": [100800.0, 99792.0, 100989.5],
            "Benchmark": [100900.0, 99386.5, 100479.8],
        },
        index=comparison_returns.index,
    )
    result = PolicyReplayResult(
        checkpoint=types.SimpleNamespace(path=Path("dummy.pt")),
        frames=_sample_frames(),
        prices=pd.DataFrame({"SPY": [100.0, 101.0, 102.0]}, index=comparison_returns.index),
        comparison_returns=comparison_returns,
        cumulative_values=cumulative_values,
        weights_history=pd.DataFrame(),
        daily_weights=pd.DataFrame(
            {
                "SPY": [0.5, 0.5, 0.5],
                "QQQ": [0.3, 0.3, 0.3],
                "GLD": [0.2, 0.2, 0.2],
            },
            index=comparison_returns.index,
        ),
        asset_returns=pd.DataFrame(
            {
                "SPY": [0.01, -0.02, 0.015],
                "QQQ": [0.008, -0.01, 0.012],
                "GLD": [0.004, -0.006, 0.005],
            },
            index=comparison_returns.index,
        ),
        benchmark_returns=comparison_returns["benchmark"],
        summary_table=pd.DataFrame(
            {
                "annualized_return": [0.10, 0.08, 0.07],
                "annualized_volatility": [0.12, 0.11, 0.10],
                "sharpe_ratio": [0.45, 0.35, 0.30],
                "sortino_ratio": [0.50, 0.40, 0.33],
                "max_drawdown": [-0.09, -0.08, -0.07],
                "calmar_ratio": [1.11, 1.00, 1.00],
            },
            index=["portfolio", "equal_weight", "benchmark"],
        ),
        metrics={
            "total_return": 0.12,
            "benchmark_total_return": 0.09,
            "excess_total_return": 0.03,
            "annualized_return": 0.10,
            "annualized_volatility": 0.12,
            "sharpe_ratio": 0.45,
            "max_drawdown": -0.09,
            "equal_weight_total_return": 0.08,
            "active_vs_equal_weight_total_return": 0.04,
        },
        requested_tickers=["SPY", "QQQ", "GLD"],
        benchmark_ticker="SPY",
        starting_capital=100000.0,
        rebalance_frequency="B",
        rebalance_label="1D",
        rebalance_mode="manual",
        estimated_steps=3,
        risk_free_assumption=TreasuryRateAssumption(
            annual_rate=0.0425,
            maturity_label="1-Year Treasury",
            maturity_column_name="1 Yr",
            source="U.S. Treasury daily par yield curve",
            as_of_date=pd.Timestamp("2024-01-05"),
            fallback_used=False,
        ),
    )

    highlights = ReplaySummaryDialog._highlights_text(result)

    assert "1-Year Treasury" in highlights
    assert "4.25%" in highlights
    assert "U.S. Treasury daily par yield curve" in highlights

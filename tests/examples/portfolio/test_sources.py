"""Causal external data, configured providers, and transaction-series adapters."""

from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

import pytest

from examples.data import NamedMap, Series, load_providers
from examples.portfolio.models import AAPL, Equity
from examples.portfolio.signals import MomentumSignals
from examples.portfolio.sources import (
    CashFlow,
    CSVEquityProvider,
    CSVIndexSource,
    DummyEquityProvider,
    Fill,
    PriceHistory,
    TransactionLog,
    timestamp,
)

from ._data import AGE, DATA, TIMES


def test_publication_time_prevents_lookahead_and_age_uses_observation_time():
    observed = (timestamp("2026-01-07T09:00:00Z"), timestamp("2026-01-07T09:05:00Z"))
    available = (observed[0], timestamp("2026-01-07T09:10:00Z"))
    history = PriceHistory(AAPL, Series(observed, (100, 105)), available, (1, 1))
    assert history.mark(timestamp("2026-01-07T09:07:00Z"), max_age=AGE) == 100
    assert history.mark(available[1], max_age=AGE) == 105
    signal = MomentumSignals(NamedMap({"aapl": history.known_prices}), lookback=1).load()["aapl"]
    with pytest.raises(ValueError, match="No observation"):
        signal.asof(timestamp("2026-01-07T09:07:00Z"))
    assert signal.asof(available[1]) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="stale"):
        history.mark(available[1], max_age=timedelta(minutes=2))


def test_transaction_series_aggregate_simultaneous_executions_without_losing_ids():
    at = TIMES[1]
    log = TransactionLog(
        (Fill("one", "research", at, at, "aapl", 2, 100, 0.1), Fill("two", "research", at, at, "aapl", -3, 102, 0.2))
    )
    series = log.executions("aapl")
    assert len(log.events) == 2
    assert series.quantity.values == (-1,)
    assert series.volume.values == (5,)
    assert series.vwap.values == (101.2,)
    assert series.fees.values == pytest.approx((0.3,))
    assert log.executions("msft").quantity.values == ()
    with pytest.raises(ValueError, match="Duplicate transaction ID"):
        TransactionLog((log.events[0], log.events[0]))


def test_invalid_publication_times_volumes_and_events_are_rejected(market):
    with pytest.raises(ValueError, match="available|publication"):
        replace(market["aapl"], available_at=market["aapl"].available_at[::-1])
    with pytest.raises(ValueError, match="strictly positive"):
        replace(market["aapl"], volumes=(0,) * len(market["aapl"].volumes))
    with pytest.raises(ValueError, match="finite"):
        Fill("bad", "research", TIMES[1], TIMES[1], "aapl", 1, float("nan"))
    with pytest.raises(ValueError, match="reporting time"):
        CashFlow("bad", "research", TIMES[1], TIMES[0], 100)


def test_csv_index_source_preserves_schedule_and_validates_schema(tmp_path):
    source = tmp_path / "index.csv"
    source.write_text("at,level\n2025-01-01T12:00:00Z,100\n2025-01-02T12:00:00Z,101\n")
    index = CSVIndexSource(source).load()
    assert index.values == (100, 101)
    assert index.times[0].utcoffset() == timedelta(0)
    source.write_text("at,price\n2025-01-01T12:00:00Z,100\n")
    with pytest.raises(ValueError, match="level"):
        CSVIndexSource(source).load()


def test_requests_include_all_parameters_not_only_the_instrument_id(tmp_path):
    original = DATA / "market.csv"
    alternate = tmp_path / "alternate.csv"
    alternate.write_text(original.read_text().replace(",100,1000", ",200,1000"))
    one = CSVEquityProvider(symbol="AAPL", path=original, namespace="synthetic")
    two = CSVEquityProvider(symbol="AAPL", path=alternate, namespace="synthetic")
    assert one.instrument == two.instrument
    histories = load_providers(NamedMap({"one": one, "two": two}))
    assert histories["one"].prices.values[0] == 100
    assert histories["two"].prices.values[0] == 200
    # A new load observes changed sources instead of a persistent cache.
    alternate.write_text(original.read_text())
    assert two.load().prices.values[0] == 100


def test_provider_symbols_and_quote_currencies_remain_distinct(tmp_path):
    original = DATA / "market.csv"
    with pytest.raises(ValueError, match="No market data for other:AAPL"):
        CSVEquityProvider(symbol="AAPL", path=original, namespace="other").load()
    with pytest.raises(ValueError, match="Market currency.*must be EUR"):
        DummyEquityProvider(symbol="AAPL", currency="EUR").load()
    path = tmp_path / "providers.csv"
    content = original.read_text()
    extra = content.replace("synthetic", "other").splitlines()[1:]
    path.write_text(content + "\n".join(extra) + "\n")
    histories = load_providers(
        NamedMap(
            {
                "one": CSVEquityProvider(symbol="AAPL", path=path, namespace="synthetic"),
                "two": CSVEquityProvider(symbol="AAPL", path=path, namespace="other"),
            }
        )
    )
    assert histories["one"].prices == histories["two"].prices
    assert histories["one"].equity != histories["two"].equity


def test_loading_validates_the_feed_schema_and_requested_symbol(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("member,price\naapl,100\n")
    with pytest.raises(ValueError, match="unique columns"):
        CSVEquityProvider(symbol="AAPL", path=bad, namespace="synthetic").load()
    bad.write_text((DATA / "market.csv").read_text().replace(",AAPL,", ",MSFT,"))
    with pytest.raises(ValueError, match="No market data for synthetic:AAPL"):
        CSVEquityProvider(symbol="AAPL", path=bad, namespace="synthetic").load()


def test_provider_construction_and_symbolic_use_do_not_fetch_data(tmp_path):
    from examples.portfolio.valuation import EquityBasket, EquityHoldings, NamedPortfolio, equity

    provider = CSVEquityProvider(symbol="AAPL", path=tmp_path / "absent.csv", namespace="test")
    assert provider.instrument == Equity("AAPL", "test")

    class Holdings(EquityHoldings):
        aapl = equity.component(provider)
        msft = equity.component(provider)

    Basket = EquityBasket.define(Holdings)
    portfolio = NamedPortfolio.define(Basket)("symbolic")
    assert portfolio.basket.positions["aapl"].provider is provider
    assert portfolio.nav.free_symbols
    with pytest.raises(FileNotFoundError):
        load_providers(portfolio.providers)

"""External observations, publication times, and transaction-series adapters."""

from dataclasses import replace
from datetime import timedelta

import pytest

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.models import AAPL, MSFT
from examples.portfolio.signals import MomentumSignals
from examples.portfolio.sources import (
    CashFlow,
    CSVIndexSource,
    CSVMarketSource,
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


def test_feed_schema_identity_and_invalid_events_are_rejected(tmp_path, market):
    bad = tmp_path / "bad.csv"
    bad.write_text("member,price\naapl,100\n")
    with pytest.raises(ValueError, match="unique columns"):
        CSVMarketSource(bad, NamedMap({"aapl": AAPL})).load()
    bad.write_text((DATA / "market.csv").read_text().replace(",AAPL,", ",MSFT,"))
    with pytest.raises(ValueError, match="ticker for aapl"):
        CSVMarketSource(bad, NamedMap({"aapl": AAPL, "msft": MSFT})).load()
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

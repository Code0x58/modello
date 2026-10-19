"""Causal signal decisions and rebalance intentions."""

from dataclasses import replace

import pytest

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.account import AccountEngine
from examples.portfolio.models import AAPL, MSFT
from examples.portfolio.signals import MomentumSignals, TargetWeights, positive_signal_targets, rebalance_orders

from ._data import AGE, ARRIVAL, DECISION, TIMES


def test_future_prices_and_postings_cannot_change_decisions(engine, market, ledger, scenario):
    changed = market.map(
        lambda history: replace(
            history,
            prices=Series(
                history.prices.times,
                tuple(
                    value if at <= DECISION else value * 10
                    for at, value in zip(history.prices.times, history.prices.values, strict=True)
                ),
            ),
        )
    )
    other_engine = AccountEngine(engine.account, changed, max_price_age=AGE)
    snapshot = other_engine.evaluate(ledger, (TIMES[0], DECISION)).snapshots[-1]
    feature_prices = changed.map(lambda history: history.sample((*TIMES[:2], DECISION), max_age=AGE))
    signals = MomentumSignals(feature_prices, lookback=2).load()
    targets = positive_signal_targets(signals, DECISION)
    assert targets == scenario.targets
    assert rebalance_orders(snapshot, targets, batch="rebalance-1", arrival_at=ARRIVAL) == scenario.orders
    assert snapshot.quantities == {"aapl": 10, "msft": 10}
    assert snapshot.cash == 1000


def test_signal_targets_and_rebalance_quantities(scenario):
    assert scenario.targets.weights == {"aapl": 0.8, "msft": 0}
    apple = scenario.orders["rebalance-1:aapl"]
    microsoft = scenario.orders["rebalance-1:msft"]
    assert apple.quantity == pytest.approx(0.8 * 2520 / 104 - 10)
    assert microsoft.quantity == -10
    assert apple.model_price == 104
    assert apple.equity is AAPL and microsoft.equity is MSFT
    for invalid in ({"aapl": -0.1}, {"aapl": 1.1}, {"aapl": float("nan")}):
        with pytest.raises(ValueError):
            TargetWeights(DECISION, NamedMap(invalid))

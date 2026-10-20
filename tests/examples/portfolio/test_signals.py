"""Causal signal decisions and rebalance intentions."""

from dataclasses import replace

import pytest

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.account import AccountEngine
from examples.portfolio.models import AAPL, MSFT
from examples.portfolio.signals import MomentumSignals, TargetWeights

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
    targets = TargetWeights.from_signals(signals, DECISION)
    assert targets == scenario.targets
    assert targets.orders(snapshot, batch="rebalance-1", arrival_at=ARRIVAL).by_name == scenario.orders.by_name
    assert snapshot.quantities == {"aapl": 10, "msft": 10}
    assert snapshot.cash == 1000


def test_signal_targets_and_rebalance_quantities(scenario):
    assert scenario.targets.weights == {"aapl": 0.8, "msft": 0}
    members = scenario.decision.portfolio.basket.members
    apple = scenario.orders.for_member(members.aapl)
    microsoft = scenario.orders.for_member(members.msft)
    assert apple.quantity == pytest.approx(0.8 * 2520 / 104 - 10)
    assert microsoft.quantity == -10
    assert apple.model_price == 104
    assert apple.equity == AAPL and microsoft.equity == MSFT
    for invalid in ({"aapl": -0.1}, {"aapl": 1.1}, {"aapl": float("nan")}):
        with pytest.raises(ValueError):
            TargetWeights(DECISION, NamedMap(invalid))


def test_rebalance_model_can_solve_an_order_or_the_implied_target():
    from sympy import Rational, Symbol

    from examples.model_plan import Known
    from examples.portfolio.signals import Rebalance

    forward = Rebalance("target", nav=1000, price=100, opening_quantity=3, target_weight=Rational(1, 2))
    fitted = Rebalance("fit", nav=1000, price=100, opening_quantity=3, order_quantity=2)
    assert forward.order_quantity == 2
    assert fitted.target_weight == Rational(1, 2)
    observed = Symbol("observed_order", real=True)
    view = Rebalance("inverse", nav=1000, price=100, opening_quantity=3, order_quantity=observed)
    plan = view.compile()
    assert float(plan.evaluate(Known(view.order_quantity, 2)).target_weight) == 0.5


def test_an_unchanged_allocation_has_an_empty_order_projection(scenario):
    decision = scenario.decision
    targets = TargetWeights(decision.at, decision.weights)
    orders = decision.rebalance(targets, batch="unchanged", arrival_at=decision.at)
    assert len(orders) == 0
    assert len(orders.members) == 2
    with pytest.raises(KeyError):
        orders.for_member(decision.portfolio.basket.members.aapl)

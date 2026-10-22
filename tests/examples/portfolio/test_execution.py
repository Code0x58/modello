"""Independent execution benchmarks and implementation-shortfall costs."""

from dataclasses import replace

import pytest

from examples.data import NamedMap
from examples.portfolio.models import MSFT
from examples.portfolio.sources import (
    Fill,
    TransactionLog,
    timestamp,
)

from ._data import AGE, ARRIVAL, ASSESSMENT


@pytest.fixture
def aapl_order(scenario):
    return scenario.orders.for_member(scenario.decision.portfolio.basket.members.aapl)


@pytest.fixture
def linked_fill(aapl_order):
    at = timestamp("2026-01-07T09:06:00Z")
    return Fill("fill", "research", at, at, "aapl", 1, 105, order_id=aapl_order.id)


def test_execution_cost_decomposition_and_partial_fill_opportunity(scenario):
    members = scenario.decision.portfolio.basket.members
    apple = scenario.execution.for_member(members.aapl)
    microsoft = scenario.execution.for_member(members.msft)
    assert apple.filled_quantity == 7
    assert apple.remaining_quantity == pytest.approx(0.8 * 2520 / 104 - 17)
    assert apple.execution_vwap == pytest.approx((4 * 105 + 3 * 105.2) / 7)
    assert apple.market_vwap == pytest.approx((500 * 104.5 + 100 * 104.7 + 200 * 104.8) / 800)
    assert apple.model_slippage_cost == pytest.approx(8.6)
    assert apple.delay_cost == pytest.approx(3.5)
    assert apple.trading_cost == pytest.approx(4.1)
    assert apple.opportunity_cost == pytest.approx(apple.remaining_quantity * (104.8 - 104))
    direct = 4 * 105 + 3 * 105.2 + apple.remaining_quantity * 104.8 - abs(apple.order.quantity) * 104 + 1
    assert apple.implementation_shortfall == pytest.approx(direct)
    assert microsoft.model_slippage_cost == pytest.approx(10 * (48 - 47.7) + 1)
    assert microsoft.implementation_shortfall == pytest.approx(4)
    assert microsoft.opportunity_cost == 0
    assert apple.arrival_slippage_bps > 0 and microsoft.arrival_slippage_bps > 0


@pytest.mark.parametrize("member, price, quantity", [("aapl", 103, 2), ("msft", 49, -2)], ids=["buy", "sell"])
def test_execution_improvement_has_a_negative_cost_for_both_sides(
    scenario, market, linked_fill, member, price, quantity
):
    members = scenario.decision.portfolio.basket.members
    order = replace(scenario.orders.for_member(getattr(members, member)), quantity=quantity)
    fill = replace(linked_fill, member=member, quantity=quantity, price=price, order_id=order.id)
    result = order.assess(TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE)
    assert result.model_slippage_cost == pytest.approx(-2)
    assert result.implementation_shortfall == pytest.approx(-2)


def test_unfilled_orders_and_late_publication_are_explicit(aapl_order, linked_fill, market):
    late = replace(linked_fill, quantity=2, recorded_at=timestamp("2026-01-07T09:20:00Z"))
    report = aapl_order.assess(TransactionLog((late,)), market, at=ASSESSMENT, max_price_age=AGE)
    assert report.filled_quantity == 0
    assert report.execution_vwap is None and report.model_slippage_bps is None
    assert report.model_slippage_cost == 0
    assert report.implementation_shortfall == pytest.approx(abs(aapl_order.quantity) * (104.8 - 104))


@pytest.mark.parametrize(
    "change, message",
    [
        pytest.param({"quantity": -1}, "member, side, and arrival", id="opposite-side"),
        pytest.param({"quantity": 100}, "exceeds the model order", id="overfill"),
        pytest.param({"member": "msft"}, "member, side, and arrival", id="other-member"),
        pytest.param({"at": timestamp("2026-01-07T09:04:00Z")}, "arrival time", id="before-arrival"),
    ],
)
def test_order_fill_linkage_is_validated(aapl_order, linked_fill, market, change, message):
    with pytest.raises(ValueError, match=message):
        aapl_order.assess(
            TransactionLog((replace(linked_fill, **change),)),
            market,
            at=ASSESSMENT,
            max_price_age=AGE,
        )


def test_market_benchmarks_are_independent_of_the_accounts_fill_prices(scenario, aapl_order, linked_fill, market):
    member = scenario.decision.portfolio.basket.members.aapl
    fill = replace(linked_fill, at=ARRIVAL, recorded_at=ARRIVAL, price=999)
    result = aapl_order.assess(TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE)
    assert result.market_vwap == scenario.execution.for_member(member).market_vwap
    assert result.execution_vwap == 999
    wrong = NamedMap({"aapl": replace(market["aapl"], equity=MSFT), "msft": market["msft"]})
    with pytest.raises(ValueError, match="instrument must match"):
        aapl_order.assess(TransactionLog(()), wrong, at=ASSESSMENT, max_price_age=AGE)

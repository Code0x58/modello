"""Independent execution benchmarks and implementation-shortfall costs."""

from dataclasses import replace

import pytest

from examples.compute.named_map import NamedMap
from examples.portfolio.execution import ExecutionBenchmark
from examples.portfolio.models import MSFT
from examples.portfolio.sources import (
    Fill,
    TransactionLog,
    timestamp,
)

from ._data import AGE, ARRIVAL, ASSESSMENT


def test_execution_cost_decomposition_and_partial_fill_opportunity(scenario):
    apple = scenario.execution["rebalance-1:aapl"]
    microsoft = scenario.execution["rebalance-1:msft"]
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


def test_execution_improvement_has_a_negative_cost_for_both_sides(scenario, market):
    for name, fill_price in (("aapl", 103), ("msft", 49)):
        order = replace(scenario.orders[f"rebalance-1:{name}"], quantity=2 if name == "aapl" else -2)
        fill = Fill(
            "better",
            "research",
            timestamp("2026-01-07T09:06:00Z"),
            timestamp("2026-01-07T09:06:00Z"),
            name,
            order.quantity,
            fill_price,
            order_id=order.id,
        )
        result = ExecutionBenchmark.assess(
            NamedMap({order.id: order}), TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE
        )[order.id]
        assert result.model_slippage_cost == pytest.approx(-2)
        assert result.implementation_shortfall == pytest.approx(-2)


def test_unfilled_orders_and_late_publication_are_explicit(scenario, market):
    order = scenario.orders["rebalance-1:aapl"]
    fill = Fill(
        "late",
        "research",
        timestamp("2026-01-07T09:06:00Z"),
        timestamp("2026-01-07T09:20:00Z"),
        "aapl",
        2,
        105,
        order_id=order.id,
    )
    report = ExecutionBenchmark.assess(
        NamedMap({order.id: order}), TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE
    )[order.id]
    assert report.filled_quantity == 0
    assert report.execution_vwap is None and report.model_slippage_bps is None
    assert report.model_slippage_cost == 0
    assert report.implementation_shortfall == pytest.approx(abs(order.quantity) * (104.8 - 104))


@pytest.mark.parametrize(
    "change", [{"quantity": -1}, {"quantity": 100}, {"member": "msft"}, {"at": timestamp("2026-01-07T09:04:00Z")}]
)
def test_order_fill_linkage_is_validated(scenario, market, change):
    order = scenario.orders["rebalance-1:aapl"]
    fill = Fill(
        "bad",
        "research",
        timestamp("2026-01-07T09:06:00Z"),
        timestamp("2026-01-07T09:06:00Z"),
        "aapl",
        1,
        105,
        order_id=order.id,
    )
    with pytest.raises(ValueError):
        ExecutionBenchmark.assess(
            NamedMap({order.id: order}),
            TransactionLog((replace(fill, **change),)),
            market,
            at=ASSESSMENT,
            max_price_age=AGE,
        )


def test_market_benchmarks_are_independent_of_the_accounts_fill_prices(scenario, market):
    order = scenario.orders["rebalance-1:aapl"]
    fill = Fill("expensive", "research", ARRIVAL, ARRIVAL, "aapl", 1, 999, order_id=order.id)
    result = ExecutionBenchmark.assess(
        NamedMap({order.id: order}), TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE
    )[order.id]
    assert result.market_vwap == scenario.execution[order.id].market_vwap
    assert result.execution_vwap == 999
    wrong = NamedMap({"aapl": replace(market["aapl"], equity=MSFT), "msft": market["msft"]})
    with pytest.raises(ValueError, match="instrument must match"):
        ExecutionBenchmark.assess(
            NamedMap({order.id: order}), TransactionLog(()), wrong, at=ASSESSMENT, max_price_age=AGE
        )


def test_execution_model_can_infer_fees_from_an_observed_shortfall():
    from sympy import Rational

    from examples.portfolio.execution import ExecutionCosts

    known = dict(
        quantity=7,
        remaining=3,
        side=1,
        notional=Rational("735.6"),
        model_price=104,
        decision_price=104,
        arrival_price=Rational("104.5"),
        terminal_price=Rational("104.8"),
    )
    forward = ExecutionCosts("cost", **known, fees=1)
    fitted = ExecutionCosts("fit", **known, implementation_shortfall=11)
    assert forward.implementation_shortfall == 11
    assert fitted.fees == 1
    assert fitted.model_slippage_cost == forward.model_slippage_cost

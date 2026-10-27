"""The shared research contracts and HJB slice agree with their mathematics."""

import pytest
from sympy import Rational, diff, exp, log, simplify

from examples.market_making import (
    CARAUtility,
    ContinuationValue,
    ExponentialIntensity,
    ExponentialQuote,
    MarketMaker,
    QuadraticInventoryRisk,
    QuadraticTerminalPenalty,
)
from modello import InstanceDummy
from modello.contrib.research import ResearchCase


def test_default_component_contracts_are_proved_and_scoped_in_the_parent():
    report = MarketMaker.definition_report()
    for path in (
        "utility.gamma/positive",
        "utility.value/increasing",
        "utility.value/concave",
        "risk.cost/convex",
        "terminal.cost/nonnegative",
        "bid.intensity.rate/curvature_bound",
        "ask.intensity.rate/decreasing",
    ):
        assert report.explain(path).verdict == "strictly true"
    assert all(check.verdict == "strictly true" for check in report.checks)


def test_parameter_conditions_refresh_after_subclassing_and_observations():
    class UnrestrictedUtility(CARAUtility):
        gamma = InstanceDummy("gamma", real=True)

    report = UnrestrictedUtility.definition_report()
    condition = report.explain("gamma/positive").condition
    assert report.explain("gamma/positive").verdict == "conditionally true"
    assert report.explain("value/increasing").verdict == "conditionally true"
    assert len(condition.free_symbols) == 1
    parameter = next(iter(condition.free_symbols))
    assert condition == (parameter > 0)
    assert report.explain("value/smooth").verdict == report.explain("value/concave").verdict == "strictly true"
    case = ResearchCase.for_model(UnrestrictedUtility)
    for gamma, expected in ((2, "strictly true"), (0, "strictly false"), (-1, "strictly false")):
        assert case.observe(gamma=gamma).check().explain("value/increasing").verdict == expected
    assert all(check.verdict == "strictly true" for check in CARAUtility.definition_report().checks)


def test_incompatible_components_refresh_inherited_requirements_automatically():
    class ConvexUtility(CARAUtility):
        value = exp(CARAUtility.gamma * CARAUtility.wealth)

    class IncreasingIntensity(ExponentialIntensity):
        rate = ExponentialIntensity.scale * exp(ExponentialIntensity.decay * ExponentialIntensity.distance)

    assert ConvexUtility.definition_report().explain("value/concave").verdict == "strictly false"
    report = IncreasingIntensity.definition_report()
    assert report.explain("rate/positive").verdict == report.explain("rate/smooth").verdict == "strictly true"
    assert report.explain("rate/decreasing").verdict == report.explain("rate/right_tail").verdict == "strictly false"


def test_quote_is_the_cara_stationary_maximum_and_matches_the_closed_hamiltonian():
    quote = ExponentialQuote("symbolic")
    distance = quote.distance
    first = simplify(diff(quote.gain, distance).subs(distance, quote.optimal_distance))
    second = simplify(diff(quote.gain, distance, 2).subs(distance, quote.optimal_distance))
    assert first == 0
    assert second.is_negative is True
    expected = (
        quote.intensity.scale
        * quote.step
        / (quote.intensity.decay + quote.gamma * quote.step)
        * exp(-quote.intensity.decay * quote.optimal_distance)
    )
    assert simplify(quote.value - expected) == 0
    # The maximizer is unique globally: the derivative has the sign of
    # exp((k+gamma*Delta)*(delta*-delta)) - exp(k*(delta*-delta)).


@pytest.mark.parametrize(
    ("inventory", "expected_bid", "expected_ask", "expected_slope"),
    [(0, Rational(1, 2), Rational(1, 2), -1), (1, 0, Rational(1, 2), 0), (-1, Rational(1, 2), 0, 0)],
)
def test_reduced_hjb_signs_and_inventory_boundary_gates(inventory, expected_bid, expected_ask, expected_slope):
    quote = ExponentialQuote(
        "quote", intensity=ExponentialIntensity("liquidity", scale=2, decay=1), gamma=1, step=1, cost=0
    )
    maker = MarketMaker(
        "slice",
        gamma=1,
        step=1,
        capacity=1,
        inventory=inventory,
        residual=0,
        risk=QuadraticInventoryRisk("risk", sigma=1),
        terminal=QuadraticTerminalPenalty("penalty", eta=2),
        continuation=ContinuationValue("theta", current=0, after_bid=0, after_ask=0),
        bid=quote,
        ask=quote,
    )
    assert (maker.bid_term, maker.ask_term) == (expected_bid, expected_ask)
    assert maker.continuation.time_derivative == expected_slope
    assert maker.residual == 0
    assert maker.terminal_value == -2 * inventory**2
    assert quote.optimal_distance == log(2)


def test_quotes_use_opposite_inventory_neighbors_and_the_trade_size():
    continuation = ContinuationValue("theta", step=2, current=3, after_bid=1, after_ask=5)
    assert (continuation.bid_cost, continuation.ask_cost) == (1, -1)
    liquidity = ExponentialIntensity("liquidity", scale=2, decay=1)
    bid = ExponentialQuote("bid", intensity=liquidity, gamma=1, step=2, cost=continuation.bid_cost)
    ask = ExponentialQuote("ask", intensity=liquidity, gamma=1, step=2, cost=continuation.ask_cost)
    assert simplify(bid.optimal_distance - ask.optimal_distance) == 2

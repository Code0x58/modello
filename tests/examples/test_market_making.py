"""The exploratory contracts and HJB slice agree with their mathematics."""

import pytest
from sympy import Function, Rational, S, Symbol, diff, exp, log, simplify

from examples.market_making import (
    CARAUtility,
    ContinuationValue,
    ExponentialIntensity,
    ExponentialQuote,
    MarketMaker,
    QuadraticInventoryRisk,
    QuadraticTerminalPenalty,
    Requirements,
    checked,
    convex,
    intensity_requirements,
    positive_slope,
    smooth,
    uniform_curvature_bound,
    utility_requirements,
)
from modello import InstanceDummy


def test_default_component_contracts_are_proved_and_scoped_in_the_parent():
    reports = Requirements.for_model(MarketMaker)
    assert set(reports) == {"utility", "risk", "terminal", "bid.intensity", "ask.intensity"}
    assert all(check.verdict == "strictly true" for report in reports.values() for check in report.checks)


def test_parameter_conditions_are_distinct_from_failures_and_rechecked_after_binding():
    @checked(utility_requirements)
    class UnrestrictedUtility(CARAUtility):
        gamma = InstanceDummy("gamma", real=True)

    checks = Requirements.for_model(UnrestrictedUtility)[""].checks
    condition = checks[0].condition
    assert checks[0].verdict == checks[2].verdict == "conditionally true"
    assert len(condition.free_symbols) == 1
    parameter = next(iter(condition.free_symbols))
    assert condition == (parameter > 0)
    assert checks[1].verdict == checks[3].verdict == "strictly true"
    assert utility_requirements(UnrestrictedUtility("positive", gamma=2))[2].verdict == "strictly true"
    assert utility_requirements(UnrestrictedUtility("constant", gamma=0))[2].verdict == "strictly false"
    assert utility_requirements(UnrestrictedUtility("negative", gamma=-1))[2].verdict == "strictly false"
    assert all(check.verdict == "strictly true" for check in Requirements.for_model(CARAUtility)[""].checks)


def test_incompatible_components_can_be_defined_and_report_a_failed_requirement():
    @checked(utility_requirements)
    class ConvexUtility(CARAUtility):
        value = exp(CARAUtility.gamma * CARAUtility.wealth)

    @checked(intensity_requirements)
    class IncreasingIntensity(ExponentialIntensity):
        rate = ExponentialIntensity.scale * exp(ExponentialIntensity.decay * ExponentialIntensity.distance)

    assert Requirements.for_model(ConvexUtility)[""].checks[-1].verdict == "strictly false"
    checks = Requirements.for_model(IncreasingIntensity)[""].checks
    assert checks[0].verdict == checks[1].verdict == "strictly true"
    assert checks[2].verdict == checks[3].verdict == "strictly false"


def test_global_certificates_reject_a_pole_and_a_local_only_increase():
    x = Symbol("x", real=True)
    assert smooth(1 / x, x, 2) is S.false
    assert positive_slope(x**2, x) is S.false
    assert convex(x**2, x) is S.true
    # This deliberately stronger derivative certificate is not a complete
    # characterization of strict monotonicity: x³ still increases strictly.
    assert positive_slope(x**3, x) is S.false
    assert smooth(abs(x), x, 2) is None
    assert smooth(Function("f")(x), x, 1) is None
    # A pointwise bound strictly below 2 need not have supremum below 2.
    assert uniform_curvature_bound(1 / log(1 + exp(x)), x) is None


def test_parameter_poles_remain_unresolved_in_smoothness_checks():
    x, a = Symbol("x", real=True), Symbol("a", real=True)
    assert smooth(x / a, x, 2) is None
    assert smooth(exp(x / a), x, 2) is None
    nonzero = Symbol("nonzero", real=True, nonzero=True)
    assert smooth(exp(x / nonzero), x, 2) is S.true


def test_uniform_curvature_preserves_zero_derivative_exclusions():
    x, k = Symbol("x", real=True), Symbol("k", real=True)
    condition = uniform_curvature_bound(exp(-k * x), x)
    assert condition.free_symbols == {k}
    assert condition.subs(k, 0) is S.false
    assert condition.subs(k, 1) is S.true
    assert uniform_curvature_bound(S.One, x) is S.false
    assert uniform_curvature_bound(x**2, x) is S.false


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
    assert all(check.verdict == "strictly true" for check in intensity_requirements(quote.intensity))


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

"""The BSM study distinguishes exact contracts, boundaries, and observations."""

import math
from dataclasses import replace

import pytest
from sympy import Eq, Interval, Piecewise, Rational, S, exp, log, simplify, sqrt

from examples.research_models.black_scholes import (
    AffineSchedule,
    ConstantSchedule,
    EuropeanOptions,
    OptionStudy,
    PayoffBoundary,
    _payoff_integral,
    boundary_checks,
    dividend_free_method,
    normal_cdf,
    normal_density,
    study_case,
    symbolic_method,
    valuation_method,
)
from modello.contrib.research import ResearchCase, ResearchError, smooth


@pytest.fixture(scope="module")
def cases() -> dict[str, ResearchCase[OptionStudy]]:
    return {
        "constant": study_case(),
        "affine": study_case(schedule=True, now=Rational(1, 4)),
        "no_dividends": study_case(dividends=False),
    }


@pytest.fixture(scope="module")
def symbolic_results(cases):
    return {name: cases[name].run(symbolic_method(cases[name])) for name in ("constant", "affine")}


@pytest.mark.parametrize("name", ("constant", "affine"))
def test_exact_identities_and_checked_schedule_chain_rule(cases, symbolic_results, name):
    case = cases[name]
    result = symbolic_results[name]
    assert len(result.data.residuals) == 18
    assert set(result.data.residuals.values()) == {S.Zero}
    assert result.data.gamma_positive
    assert all(evidence.verdict == "strictly true" for evidence in result.report.checks)
    assert result.report.explain("chain_rate_total").verdict == "strictly true"
    assert result.report.explain("chain_yield_total").verdict == "strictly true"
    assert result.report.explain("chain_variance_total").verdict == "strictly true"
    for total in ("rate_total", "yield_total", "variance_total"):
        assert result.report.explain(f"terminal_{total}").verdict == "strictly true"
        regularity = result.report.explain(f"schedule_{total}_c1")
        assert regularity.verdict == "strictly true"
        assert regularity.resolved is not None
        assert regularity.resolved.domain == Interval(0, 1)
    # A scalar observation does not reduce the quantified spot curve to a point.
    spot = case.symbols.pricing.spot
    assert spot in case.expression("pricing.call", wrt=spot).free_symbols
    assert not case.expression("pricing.call").free_symbols


def test_cdf_composition_is_an_explicit_trust_boundary(cases, symbolic_results):
    case = cases["constant"]
    assert EuropeanOptions.strike.is_positive is None
    assert EuropeanOptions.carry.variance_total.is_positive is None
    original = replace(case, theorems=()).check()
    assert original.explain("pricing.call/call_c2").verdict == "unresolved"
    checked = symbolic_results["constant"].report.explain("pricing.call/call_c2")
    assert checked.kind == "theorem"
    assert checked.verdict == "strictly true"
    assert checked.premises
    assert all(premise.verdict == "strictly true" for premise in checked.premises)
    assert any(premise.requirement.name == "call_form" for premise in checked.premises)


def test_symbolic_interest_is_valid_but_numerical_valuation_needs_observations(cases):
    case = ResearchCase.for_model(OptionStudy, "unobserved_interest")
    case = case.observe(
        now=0,
        spot=100,
        strike=100,
        rate_total=case.symbols.schedule.rate_intercept,
        yield_total=0,
        variance_total=Rational(1, 25),
        schedule=ConstantSchedule("symbolic_rate", yield_intercept=0, variance_intercept=Rational(1, 25)),
    )
    case = replace(case, theorems=cases["constant"].theorems)
    with pytest.raises(ResearchError, match="finite real scalar observation") as failure:
        case.run(valuation_method(case))
    assert failure.value.case is case
    assert all(item.verdict == "strictly true" for item in failure.value.report.checks)


def test_yield_invalidates_dividend_free_parity_without_changing_earlier_branch(cases):
    no_dividends = cases["no_dividends"]
    dividends = cases["constant"]
    original_result = no_dividends.run(dividend_free_method())
    assert original_result.report.explain("dividend_free_parity").verdict == "strictly true"
    with pytest.raises(ResearchError) as captured:
        dividends.run(dividend_free_method())
    refutation = captured.value.report.explain("dividend_free_parity")
    assert refutation.verdict == "strictly false"
    assert refutation.witness
    assert dividends.expression("pricing.carry.yield_total") == Rational(1, 100)
    assert no_dividends.expression("pricing.carry.yield_total") == 0
    assert no_dividends.expression("pricing.call_minus_put", wrt="pricing.spot") == original_result.data


def test_replacement_refreshes_the_schedule_and_blocks_a_stale_pricing_snapshot(cases):
    original = cases["constant"]
    changed = original.replace("schedule", AffineSchedule).observe(
        schedule=AffineSchedule(
            "changing_schedule",
            rate_slope=Rational(1, 100),
            yield_slope=Rational(1, 200),
            variance_slope=Rational(1, 50),
        )
    )
    time = ConstantSchedule.time
    assert original.expression("schedule.rate_total", wrt=time) == (1 - time) / 25
    assert simplify(changed.expression("schedule.rate_total", wrt=time) - ((1 - time) / 25 + (1 - time**2) / 200)) == 0
    # Pricing totals are explicit observations and must be refreshed deliberately.
    assert changed.expression("pricing.carry.rate_total") == Rational(1, 25)
    with pytest.raises(ResearchError) as captured:
        changed.run(symbolic_method(changed))
    assert captured.value.report.explain("snapshot_rate_total").verdict != "strictly true"
    assert original.expression("schedule.rate_slope") == 0


@pytest.mark.parametrize(
    "settings, rejected",
    (
        ({"now": 1}, "before_maturity"),
        ({"variance": 0}, "pricing.call/positive_variance"),
        ({"variance": -Rational(1, 25)}, "pricing.call/positive_variance"),
        ({"spot": 0}, "positive_spot"),
        ({"strike": 0}, "pricing.call/positive"),
    ),
)
def test_interior_method_does_not_silently_extend_to_singular_or_invalid_inputs(settings, rejected):
    case = study_case(**settings)
    with pytest.raises(ResearchError) as captured:
        case.run(valuation_method(case))
    assert captured.value.case is case
    assert captured.value.report.explain(rejected).verdict != "strictly true"


def test_zero_variance_payoff_has_a_checked_kink():
    slopes = boundary_checks()
    assert slopes == {"call_left_slope": 0, "call_right_slope": 1, "put_left_slope": -1, "put_right_slope": 0}
    assert slopes["call_left_slope"] != slopes["call_right_slope"]
    assert slopes["put_left_slope"] != slopes["put_right_slope"]
    case = ResearchCase.for_model(PayoffBoundary)
    x = PayoffBoundary.moneyness
    assert simplify(case.expression("call", wrt=x) - case.expression("put", wrt=x) - x) == 0
    kink = case.require(smooth("kink", wrt="moneyness", consumer="boundary", reason="Test the C2 boundary premise."))
    assert kink.check().explain("smooth").verdict == "strictly false"


@pytest.mark.parametrize("settings", ({"spot": 80}, {"spot": 100}, {"spot": 120}, {"schedule": True}))
def test_discounted_lognormal_quadrature_is_an_independent_numerical_reference(settings):
    pytest.importorskip("scipy.integrate")
    case = study_case(**settings)
    result = case.run(valuation_method(case))
    data = result.data
    assert data.kind == "numerical observation"
    assert data.call_error < 1e-9
    assert data.put_error < 1e-9
    assert 0 < data.call_delta < float(exp(-case.expression("pricing.carry.yield_total")))
    assert data.put_delta < 0
    assert data.gamma > 0
    assert data.call - data.put == pytest.approx(
        float(case.expression("pricing.discounted_spot") - case.expression("pricing.discounted_strike")), abs=1e-12
    )


def test_changed_greek_does_not_keep_its_old_contract(cases):
    class IncorrectDelta(EuropeanOptions):
        call_delta = exp(-EuropeanOptions.carry.yield_total) * normal_cdf(EuropeanOptions.d1) + 1

    case = cases["constant"].replace("pricing", IncorrectDelta)
    with pytest.raises(ResearchError) as captured:
        case.run(valuation_method(case))
    assert captured.value.report.consistent is True
    assert captured.value.report.explain("pricing.call_delta/call_delta_form").verdict == "strictly false"


@pytest.fixture(scope="module")
def adversarial_cases(cases):
    shift = Rational(1, 10)

    class ShiftedTotals(ConstantSchedule):
        rate_total = ConstantSchedule.rate_intercept * (1 - ConstantSchedule.time) + shift
        yield_total = ConstantSchedule.yield_intercept * (1 - ConstantSchedule.time) + shift
        variance_total = ConstantSchedule.variance_intercept * (1 - ConstantSchedule.time) + shift

    class PatchedTerminal(ConstantSchedule):
        rate_total = Piecewise(
            (ConstantSchedule.rate_intercept * (1 - ConstantSchedule.time) + shift, ConstantSchedule.time < 1),
            (0, True),
        )

    def schedule_case(model, *, all_totals=False):
        case = (
            ResearchCase.for_model(OptionStudy)
            .replace("schedule", model)
            .observe(
                now=0,
                spot=100,
                strike=100,
                rate_total=Rational(1, 25) + shift,
                yield_total=Rational(1, 100) + (shift if all_totals else 0),
                variance_total=Rational(1, 25) + (shift if all_totals else 0),
                schedule=model(
                    "schedule",
                    rate_intercept=Rational(1, 25),
                    yield_intercept=Rational(1, 100),
                    variance_intercept=Rational(1, 25),
                ),
            )
        )
        for theorem in cases["constant"].theorems:
            case = case.with_theorem(theorem)
        return case

    class CompensatedHelpers(EuropeanOptions):
        # Both prices and Greeks deliberately compensate for the altered
        # helpers. Their full-form contracts still hold, so the source checks
        # must reject the misleading helpers themselves.
        d1 = (
            log(EuropeanOptions.spot / EuropeanOptions.strike)
            + EuropeanOptions.carry.rate_total
            - EuropeanOptions.carry.yield_total
            + EuropeanOptions.carry.variance_total / 2
        ) / sqrt(EuropeanOptions.carry.variance_total) + 1
        d2 = d1 - sqrt(EuropeanOptions.carry.variance_total)
        discounted_spot = EuropeanOptions.spot * exp(-EuropeanOptions.carry.yield_total) + 2
        discounted_strike = EuropeanOptions.strike * exp(-EuropeanOptions.carry.rate_total) + 1
        call = (discounted_spot - 2) * normal_cdf(d1 - 1) - (discounted_strike - 1) * normal_cdf(d2 - 1)
        put = (discounted_strike - 1) * normal_cdf(1 - d2) - (discounted_spot - 2) * normal_cdf(1 - d1)
        call_delta = exp(-EuropeanOptions.carry.yield_total) * normal_cdf(d1 - 1)
        put_delta = exp(-EuropeanOptions.carry.yield_total) * (normal_cdf(d1 - 1) - 1)
        gamma = (
            exp(-EuropeanOptions.carry.yield_total)
            * normal_density(d1 - 1)
            / (EuropeanOptions.spot * sqrt(EuropeanOptions.carry.variance_total))
        )
        call_minus_put = call - put + 1

    return {
        "shifted": schedule_case(ShiftedTotals, all_totals=True),
        "patched": schedule_case(PatchedTerminal),
        "helpers": cases["constant"].replace("pricing", CompensatedHelpers),
    }


@pytest.mark.parametrize("factory", (symbolic_method, valuation_method))
def test_tail_integral_terminal_conditions_fix_constants_even_with_matching_snapshots(adversarial_cases, factory):
    case = adversarial_cases["shifted"]
    with pytest.raises(ResearchError) as captured:
        case.run(factory(case))
    report = captured.value.report
    assert report.consistent is True
    for total in ("rate_total", "yield_total", "variance_total"):
        assert report.explain(f"snapshot_{total}").verdict == "strictly true"
        assert report.explain(f"chain_{total}").verdict == "strictly true"
        assert report.explain(f"schedule_{total}_c1").verdict == "strictly true"
        assert report.explain(f"terminal_{total}").verdict == "strictly false"


@pytest.mark.parametrize("factory", (symbolic_method, valuation_method))
def test_terminal_patch_does_not_replace_closed_interval_regularities(adversarial_cases, factory):
    case = adversarial_cases["patched"]
    with pytest.raises(ResearchError) as captured:
        case.run(factory(case))
    report = captured.value.report
    assert report.consistent is True
    assert report.explain("snapshot_rate_total").verdict == "strictly true"
    assert report.explain("terminal_rate_total").verdict == "strictly true"
    assert report.explain("schedule_rate_total_c1").verdict != "strictly true"


@pytest.mark.parametrize("factory", (symbolic_method, valuation_method))
def test_compensated_prices_do_not_certify_incorrect_helpers(adversarial_cases, factory):
    case = adversarial_cases["helpers"]
    with pytest.raises(ResearchError) as captured:
        case.run(factory(case))
    report = captured.value.report
    assert report.consistent is True
    for field in ("call", "put", "call_delta", "put_delta", "gamma"):
        assert report.explain(f"pricing.{field}/{field}_form").verdict == "strictly true"
    for field in ("d1", "d2", "discounted_spot", "discounted_strike", "call_minus_put"):
        assert report.explain(f"pricing.{field}/{field}_source").verdict == "strictly false"


@pytest.mark.parametrize(("spot", "strike"), ((100, 1), (1, 100)))
def test_payoff_quadrature_retains_mass_at_remote_exercise_boundaries(spot, strike):
    case = study_case(spot=spot, strike=strike, variance=Rational(1, 10000))
    result = case.run(valuation_method)
    assert all(item.verdict == "strictly true" for item in result.report.checks)
    data = result.data
    assert data.quadrature_call == pytest.approx(data.call, rel=1e-10, abs=1e-9)
    assert data.quadrature_put == pytest.approx(data.put, rel=1e-10, abs=1e-9)
    assert min(data.quadrature_call, data.quadrature_put, *data.quadrature_error_estimates) >= 0
    assert max(data.quadrature_error_estimates) < 1e-8


@pytest.mark.parametrize(
    ("spot", "strike", "variance"),
    ((100, 100, 0.04), (100, 1, 1e-100), (1, 100, 1e-100), (100, 100, 10000), (1e-200, 1e200, 0.04)),
)
def test_gaussian_payoff_kernel_matches_independent_normal_tail_values(spot, strike, variance):
    rate, dividend, deviation = 0.04, 0.01, math.sqrt(variance)
    d1 = (math.log(spot) - math.log(strike) + rate - dividend + variance / 2) / deviation
    d2 = (math.log(spot) - math.log(strike) + rate - dividend - variance / 2) / deviation
    discounted_spot, discounted_strike = spot * math.exp(-dividend), strike * math.exp(-rate)

    def cdf(z):
        return math.erfc(-z / math.sqrt(2)) / 2

    expected = (
        discounted_spot * cdf(d1) - discounted_strike * cdf(d2),
        discounted_strike * cdf(-d2) - discounted_spot * cdf(-d1),
    )
    actual = (
        _payoff_integral(math.log(spot) - dividend, -d1, deviation),
        _payoff_integral(math.log(strike) - rate, d2, deviation),
    )
    for (value, error), reference in zip(actual, expected, strict=True):
        assert value == pytest.approx(reference, rel=1e-10, abs=1e-10)
        assert math.isfinite(error) and 0 <= error <= max(1, reference) * 1e-8


@pytest.mark.parametrize("deviation", (1e-10, 1e-50))
def test_gaussian_payoff_kernel_resolves_tiny_positive_payoffs(deviation):
    value, error = _payoff_integral(0, -deviation / 2, deviation)
    expected = math.erf(deviation / (2 * math.sqrt(2)))
    assert value == pytest.approx(expected, rel=1e-8, abs=0)
    assert math.isfinite(error) and error >= 0


@pytest.mark.parametrize(("lower", "expected"), ((-math.inf, 3.0), (math.inf, 0.0)))
def test_gaussian_payoff_kernel_handles_infinite_exercise_boundaries(lower, expected):
    value, error = _payoff_integral(math.log(3), lower, 1)
    assert value == pytest.approx(expected, abs=1e-10)
    assert math.isfinite(error) and error >= 0


def test_nonfinite_payoff_quadrature_retains_the_case_and_report(cases, monkeypatch):
    import scipy.integrate

    monkeypatch.setattr(scipy.integrate, "quad", lambda *args, **kwargs: (math.nan, 0.0))
    case = cases["constant"]
    with pytest.raises(ResearchError, match="Payoff quadrature produced nonfinite") as captured:
        case.run(valuation_method)
    assert captured.value.case is case
    assert all(item.verdict == "strictly true" for item in captured.value.report.checks)


@pytest.fixture(scope="module")
def assumption_branches(cases):
    template = cases["constant"]
    base = ResearchCase.for_model(OptionStudy, "unobserved_time").observe(
        spot=100,
        strike=100,
        rate_total=Rational(1, 25),
        yield_total=Rational(1, 100),
        variance_total=Rational(1, 25),
        schedule=ConstantSchedule(
            "schedule",
            rate_intercept=Rational(1, 25),
            yield_intercept=Rational(1, 100),
            variance_intercept=Rational(1, 25),
        ),
    )
    for theorem in template.theorems:
        base = base.with_theorem(theorem)
    first = base.assume(Eq(base.symbols.now, 0))
    changed = base.assume(Eq(base.symbols.now, Rational(1, 2)))
    return first, changed


@pytest.mark.parametrize("factory", (symbolic_method, valuation_method))
def test_case_bound_method_rejects_changed_assumptions_with_the_same_problem(assumption_branches, factory):
    first, changed = assumption_branches
    assert first.problem is changed.problem
    # A factory's already evaluated snapshot expectations must not cross from
    # the first assumption branch to the second, even though the problem is shared.
    method = factory(first)
    assert method.bound_case is first
    with pytest.raises(ResearchError, match="different research case"):
        changed.run(method)
    with pytest.raises(ResearchError) as captured:
        changed.run(factory(changed))
    assert captured.value.report.explain("snapshot_rate_total").verdict != "strictly true"

"""Free-boundary certificates, trust boundaries, and independent barrier search."""

import subprocess
import sys

import pytest
from sympy import S

from examples.research_models.perpetual_put import (
    TRUSTED_STOPPING_ASSUMPTIONS,
    ConstantBSMMarket,
    PerpetualPut,
    algebra_method,
    c2_consumer_method,
    numerical_method,
    obstacle_theorem,
    perpetual_put_case,
    symbolic_case,
    trust_stopping_theorem,
    verification_method,
)
from modello.contrib.research import ResearchCase, ResearchError


@pytest.fixture(scope="module")
def case():
    return perpetual_put_case()


@pytest.fixture(scope="module")
def algebra(case):
    return case.run(algebra_method(case))


def test_exact_boundary_and_actual_one_sided_derivatives(algebra):
    data = algebra.data
    assert (data.beta, data.boundary, data.value_at_initial_spot) == (-2, S(200) / 3, S(400) / 27)
    assert data.one_sided == {
        "value_left": S(100) / 3,
        "value_right": S(100) / 3,
        "delta_left": -1,
        "delta_right": -1,
        "gamma_left": 0,
        "gamma_right": S(9) / 200,
    }
    assert data.second_derivative_jump == S(9) / 200
    assert data.boundary_report.explain("nonzero_second_derivative_jump").condition is S.true
    for name in (
        "boundary_value_matching",
        "boundary_smooth_fit",
        "continuation_pricing_generator",
        "exercise_generator_nonpositive",
        "exercise_nonnegative",
        "exercise_below_strike",
        "continuation_positive",
        "continuation_below_strike",
        "decay_at_infinity",
        "transversality_envelope",
    ):
        assert algebra.report.explain(name).condition is S.true


def test_obstacle_inequality_has_checked_calculus_premises(algebra):
    evidence = algebra.report.explain("continuation_obstacle")
    assert evidence.kind == "theorem"
    assert evidence.condition is S.true
    assert all(premise.condition is S.true for premise in evidence.premises)
    assert algebra.report.explain("continuation_obstacle/tangent_value").condition is S.true
    assert algebra.report.explain("continuation_obstacle/tangent_slope").condition is S.true
    assert "stochastic" not in evidence.steps[0].lower()


def test_constant_join_proof_checks_c1_and_refutes_c2_without_a_stochastic_theorem(case, algebra):
    c1 = algebra.data.regularity.explain("global_c1")
    c2 = algebra.data.regularity.explain("global_c2")
    assert c1.verdict == "strictly true" and c1.kind == "symbolic"
    assert c2.verdict == "strictly false" and c2.kind == "counterexample"
    assert c2.witness == ((case.expression("moneyness", wrt="moneyness"), S.One),)
    with pytest.raises(ResearchError) as failure:
        case.run(c2_consumer_method())
    assert failure.value.report.explain("requires_global_c2").condition is S.false
    assert algebra.data.one_sided["delta_left"] == algebra.data.one_sided["delta_right"]
    assert algebra.data.one_sided["gamma_left"] != algebra.data.one_sided["gamma_right"]


def test_stochastic_optimality_requires_explicit_trust_and_exports_the_assumptions(case):
    with pytest.raises(ResearchError) as failure:
        case.run(verification_method(case))
    assert failure.value.report.explain("optimal_stopping_under_GBM").verdict == "unresolved"
    trusted = trust_stopping_theorem(case)
    result = trusted.run(verification_method(trusted))
    conclusion = result.report.explain("optimal_stopping_under_GBM")
    assert conclusion.kind == "theorem" and conclusion.condition is S.true
    assert all(premise.condition is S.true for premise in conclusion.premises)
    assert result.data.trusted_assumptions == TRUSTED_STOPPING_ASSUMPTIONS
    text = " ".join(result.data.trusted_assumptions)
    for term in ("risk-neutral", "Brownian", "stopping times", "infinite stopping time", "Ito", "transversality"):
        assert term in text
    assert result.data.certificate.second_derivative_jump > 0


@pytest.mark.parametrize("initial_spot,expected", [(50, 50), (S(200) / 3, S(100) / 3)])
def test_immediate_exercise_and_boundary_values(initial_spot, expected):
    current = perpetual_put_case(initial_spot=initial_spot)
    certificate = current.run(algebra_method(current)).data
    assert certificate.value_at_initial_spot == expected
    numeric = current.run(numerical_method(current)).data
    assert numeric.barrier == float(initial_spot)
    assert numeric.discounted_payoff == pytest.approx(float(expected), abs=1e-12)
    assert numeric.value_error < 1e-12


def test_independent_barrier_optimization_matches_the_analytic_boundary(case):
    numeric = case.run(numerical_method(case)).data
    assert numeric.barrier == pytest.approx(200 / 3, abs=1e-5)
    assert numeric.discounted_payoff == pytest.approx(400 / 27, abs=1e-12)
    assert numeric.barrier_error < 1e-5 and numeric.value_error < 1e-12
    assert numeric.evaluations > 0
    assert numeric.kind == "numerical observation over constant stopping barriers"


@pytest.mark.parametrize("rate,boundary", [(S.One / 50, 50), (S.One / 20, S(500) / 7)])
def test_positive_parameter_changes_refresh_the_boundary_and_obstacle_proof(rate, boundary):
    current = perpetual_put_case(rate=rate)
    result = current.run(algebra_method(current))
    assert result.data.boundary == boundary
    assert result.data.second_derivative_jump > 0
    assert result.report.explain("continuation_obstacle").condition is S.true


def test_zero_rate_cannot_reuse_the_positive_parameter_shortcut():
    current = perpetual_put_case(rate=0)
    with pytest.raises(ResearchError) as failure:
        current.run(algebra_method(current))
    assert failure.value.case is current
    assert failure.value.report.explain("positive_rate").condition is S.false
    assert failure.value.report.explain("positive_boundary").condition is S.false
    assert failure.value.report.explain("transversality_envelope").condition is S.false


def test_dividends_cannot_reuse_the_zero_dividend_exponent():
    current = perpetual_put_case(dividend=S.One / 100)
    with pytest.raises(ResearchError) as failure:
        current.run(algebra_method(current))
    assert failure.value.report.explain("zero_dividends").condition is S.false
    assert failure.value.report.explain("continuation_pricing_generator").condition is S.false


def test_replacing_a_consumed_exponent_rechecks_its_exact_form(case):
    class WrongExponent(ConstantBSMMarket):
        beta = -S.One

    changed = case.replace("market", WrongExponent)
    with pytest.raises(ResearchError) as failure:
        changed.run(algebra_method(changed))
    assert failure.value.report.explain("form_beta").condition is S.false
    assert case.expression("beta") == -2


def test_a_redefined_curvature_cannot_supply_a_stale_obstacle_certificate():
    class WrongCurvature(PerpetualPut):
        continuation_curvature = S.Zero

    current = (
        ResearchCase.for_model(WrongCurvature)
        .with_theorem(obstacle_theorem())
        .observe(
            rate=S.One / 25,
            volatility=S.One / 5,
            strike=100,
            dividend=0,
            initial_spot=100,
        )
    )
    with pytest.raises(ResearchError) as failure:
        current.run(algebra_method(current))
    assert failure.value.report.explain("form_continuation_curvature").condition is S.false
    assert failure.value.report.explain("continuation_obstacle").condition is not S.true


def test_curve_restrictions_and_derived_observations_do_not_silently_change_the_problem(case):
    x = case.expression("moneyness", wrt="moneyness")
    restricted = case.assume(x > 2)
    with pytest.raises(ResearchError, match="both regions"):
        restricted.run(algebra_method(restricted))
    observed = case.observe(spot=100)
    with pytest.raises(ResearchError, match="derived curve values"):
        observed.run(algebra_method(observed))


def test_a_method_cannot_reuse_case_bound_native_requirements(case):
    branch = case.assume(True)
    with pytest.raises(ResearchError, match="rebuild"):
        branch.run(algebra_method(case))


def test_numerical_parameter_and_setting_errors_retain_case_context(case):
    unobserved = symbolic_case().observe(rate=S.One / 25, volatility=S.One / 5, strike=100, dividend=0)
    with pytest.raises(ResearchError, match="parameter observations") as failure:
        unobserved.run(numerical_method(unobserved))
    assert failure.value.case is unobserved
    with pytest.raises(ResearchError, match="xatol") as failure:
        case.run(numerical_method(case), xatol=float("nan"))
    assert failure.value.case is case


def test_import_loads_no_optional_numerical_or_plotting_packages():
    code = (
        "import sys; import examples.research_models.perpetual_put; "
        "assert not {'numpy','scipy','matplotlib'} & sys.modules.keys()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

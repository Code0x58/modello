"""CIR scalar identities, separate boundary premises, and numerical references."""

import importlib
import subprocess
import sys
from types import new_class

import pytest
from sympy import Rational, S, exp, simplify

from examples.financial_models.cir_bond import (
    CIRBond,
    CIRParameters,
    cir_context,
    numerical_method,
    price,
    pricing_method,
    strict_positive_boundary_method,
    symbolic_context,
    verification_method,
    with_feller,
)
from modello.contrib.verification import VerificationError


@pytest.fixture(scope="module")
def ordinary_context():
    return cir_context()


@pytest.fixture(scope="module")
def ordinary_price(ordinary_context):
    return price(ordinary_context)


@pytest.fixture(scope="module")
def high_volatility_context():
    return cir_context(sigma=Rational(2, 5))


@pytest.fixture(scope="module")
def terminal_context():
    return cir_context(initial_rate=0, horizon=0)


@pytest.fixture(scope="module")
def changed_assumptions(ordinary_context):
    return ordinary_context.assume(S.true)


def test_price_has_exact_scalar_and_explicit_trusted_stochastic_evidence(ordinary_price):
    report = ordinary_price.report
    for name in (
        "riccati_b",
        "riccati_log_a",
        "pricing_generator",
        "terminal_price",
        "rate_c2",
        "rate_decreasing",
        "rate_convex",
        "positive_price",
    ):
        assert report.explain(name).verdict == "strictly true"
    rule = report.explain("affine_valuation")
    assert rule.kind == "rule"
    assert rule.verdict == "strictly true"
    paths = report.explain("affine_valuation/nonnegative_paths")
    assert paths.kind == "rule" and paths.verdict == "strictly true"
    assert "CIR_affine_discounted_expectation" in rule.steps[0]
    assert "feller_condition" not in {item.obligation.name for item in rule.premises}
    assert 0 < float(ordinary_price.data.price) < 1


def test_verification_reduces_the_actual_riccati_and_generator_identities(ordinary_context):
    result = ordinary_context.run(verification_method(ordinary_context))
    assert result.data.riccati_b == result.data.riccati_log_a == result.data.generator == 0
    assert result.data.terminal_price == 1
    assert result.report.explain("coefficient_denominator").verdict == "strictly true"


def test_feller_rule_is_separate_and_its_failure_does_not_refute_valuation(high_volatility_context):
    context = high_volatility_context
    valuation = price(context)
    assert context.expression("feller_margin") < 0
    assert valuation.report.explain("affine_valuation").verdict == "strictly true"
    assert valuation.report.explain("affine_valuation/nonnegative_paths").verdict == "strictly true"
    assert 0 < float(valuation.data.price) < 1
    with pytest.raises(VerificationError) as caught:
        context.run(strict_positive_boundary_method(context))
    assert caught.value.context is context
    report = caught.value.report
    assert report.explain("unattainable_zero").verdict == "unresolved"
    assert report.explain("unattainable_zero/feller_condition").verdict == "strictly false"


def test_feller_can_be_added_and_removed_without_changing_its_parent_context():
    earlier = cir_context(theta=None)
    bounded = with_feller(earlier)
    condition = bounded.assumptions[-1]
    relaxed = bounded.without_assumptions(condition, reason="Compare pricing without the optional boundary condition.")
    result = bounded.run(strict_positive_boundary_method(bounded))
    assert result.data.kind == "trusted stochastic boundary rule"
    assert result.report.explain("unattainable_zero").verdict == "strictly true"
    assert bounded.assumptions == (condition,)
    assert earlier.assumptions == relaxed.assumptions == ()
    assert earlier.changes == bounded.changes == ()
    assert earlier.problem is bounded.problem is relaxed.problem
    assert relaxed is not earlier
    assert "Removed assumption" in relaxed.changes[-1]
    assert "Compare pricing without the optional boundary condition." in relaxed.changes[-1]
    with pytest.raises(VerificationError) as caught:
        relaxed.run(strict_positive_boundary_method(relaxed))
    assert caught.value.report.explain("unattainable_zero").verdict == "conditionally true"
    assert caught.value.report.explain("unattainable_zero/feller_condition").condition.has(earlier.symbols.theta)
    assert any("Compare pricing without" in issue for issue in caught.value.report.issues)
    bounded_price, relaxed_price = price(bounded), price(relaxed)
    assert bounded_price.data == relaxed_price.data
    assert earlier.expression("value") == bounded.expression("value") == relaxed.expression("value")
    for valuation in (bounded_price, relaxed_price):
        assert valuation.report.explain("affine_valuation").verdict == "strictly true"
        assert valuation.report.explain("affine_valuation/nonnegative_paths").verdict == "strictly true"
    assert bounded.assumptions == (condition,) and not earlier.assumptions


def test_the_feller_equality_context_also_has_an_unattainable_boundary():
    context = cir_context(theta=Rational(1, 30))
    assert context.expression("feller_margin") == 0
    result = context.run(strict_positive_boundary_method(context))
    assert result.report.explain("unattainable_zero/feller_condition").verdict == "strictly true"


def test_zero_initial_rate_is_a_valid_pricing_input():
    context = cir_context(initial_rate=0)
    result = price(context)
    assert simplify(result.data.price - exp(result.data.log_a)) == 0
    assert result.report.explain("rate_c2").verdict == "strictly true"
    assert result.report.explain("affine_valuation/nonnegative_paths").verdict == "strictly true"
    assert not any(item.obligation.consumer == "diffusion" for item in result.report.evidence)
    with pytest.raises(VerificationError) as caught:
        context.run(strict_positive_boundary_method(context))
    assert caught.value.report.explain("unattainable_zero/strictly_positive_initial_rate").verdict == "strictly false"


def test_terminal_bond_is_one_and_does_not_assert_strict_decrease(terminal_context):
    result = price(terminal_context)
    assert result.data.price == 1
    assert result.data.b == result.data.log_a == 0
    assert result.report.explain("rate_nonincreasing").verdict == "strictly true"
    with pytest.raises(KeyError):
        result.report.explain("rate_decreasing")
    numerical = terminal_context.run(numerical_method)
    assert numerical.data.price == 1 and numerical.data.b == numerical.data.log_a == 0
    assert numerical.data.diagnostics["ode_nfev"] == 0
    with pytest.raises(VerificationError, match="strictly positive"):
        terminal_context.run(numerical_method, rtol=0)


def test_a_changed_price_cannot_reuse_the_source_shortcut():
    class WrongTerminal(CIRBond):
        price = 2 * CIRBond.price

    context = symbolic_context(WrongTerminal).observe(
        parameters=CIRParameters("risk_neutral", kappa=Rational(3, 5), theta=Rational(1, 20), sigma=Rational(1, 5)),
        initial_rate=Rational(1, 25),
        horizon=5,
    )
    with pytest.raises(VerificationError) as caught:
        price(context)
    assert caught.value.report.explain("affine_price_form").verdict == "strictly false"
    assert caught.value.report.explain("terminal_price").verdict == "strictly false"


@pytest.mark.parametrize("field", ["variance_rate", "b_at_horizon", "value"])
def test_overridden_generator_or_report_helpers_block_analytic_and_numerical_shortcuts(field):
    changed = new_class(
        "ChangedCIR", (CIRBond,), exec_body=lambda namespace: namespace.__setitem__(field, 2 * getattr(CIRBond, field))
    )
    context = symbolic_context(changed).observe(
        parameters=CIRParameters("risk_neutral", kappa=Rational(3, 5), theta=Rational(1, 20), sigma=Rational(1, 5)),
        initial_rate=Rational(1, 25),
        horizon=5,
    )
    claim = {"variance_rate": "cir_variance_form", "b_at_horizon": "horizon_b_form", "value": "initial_value_form"}[
        field
    ]
    for factory in (pricing_method, numerical_method):
        with pytest.raises(VerificationError) as caught:
            context.run(factory(context))
        assert caught.value.report.explain(claim).verdict == "strictly false"


@pytest.mark.parametrize(
    "factory", [pricing_method, verification_method, numerical_method, strict_positive_boundary_method]
)
def test_context_aware_methods_reject_reuse_after_an_assumption_change(ordinary_context, changed_assumptions, factory):
    method = factory(ordinary_context)
    assert method.bound_context is ordinary_context
    changed = changed_assumptions
    assert changed.problem is ordinary_context.problem
    with pytest.raises(VerificationError, match="bound to a different verification context") as caught:
        changed.run(method)
    assert caught.value.context is changed


@pytest.mark.parametrize("context_name", ["ordinary_context", "high_volatility_context"])
def test_independent_riccati_integration_and_quadrature_agree_with_the_formula(request, context_name):
    context = request.getfixturevalue(context_name)
    result = context.run(numerical_method(context), rtol=1e-11, atol=1e-13)
    assert result.data.diagnostics["kind"] == "numerical observation"
    for name in ("b_difference", "log_a_difference", "price_difference", "quadrature_log_a_difference"):
        assert result.data.diagnostics[name] < 1e-9
    assert result.data.diagnostics["ode_nfev"] > 0


@pytest.mark.parametrize("value", [S.Infinity, S.NaN, float("nan"), float("inf"), 1 + S.ImaginaryUnit])
def test_nonfinite_or_nonreal_observations_are_rejected(value):
    with pytest.raises(ValueError, match="finite real"):
        cir_context(initial_rate=value)


def test_import_does_not_load_optional_numerical_dependencies():
    program = (
        "import sys; import examples.financial_models.cir_bond; "
        "assert 'scipy' not in sys.modules; assert 'numpy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)
    assert importlib.util.find_spec("examples.financial_models.cir_bond") is not None

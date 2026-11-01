"""Research claims retain scope, evidence, and honest unresolved outcomes."""

from dataclasses import replace

import pytest
from sympy import Abs, Derivative, Eq, Function, I, Interval, Rational, S, Subs, exp

from modello import Coordinate, InstanceDummy, component
from modello.contrib.verification import (
    InferenceRule,
    Method,
    Obligation,
    VerificationContext,
    VerificationError,
    VerificationModel,
    continuously_differentiable,
    convex,
    derivative_ratio_bound,
    differentiable,
    equal_to,
    nonzero,
    positive,
    predicate,
    specifies,
    strictly_increasing,
    supremum_below,
)
from modello.errors import ModelDefinitionError, ModelValidationError


class Curve(VerificationModel):
    x = Coordinate("x", real=True)
    a = InstanceDummy("a", real=True)
    value = specifies(
        a * x**3, strictly_increasing("value", wrt="x", consumer="optimizer", reason="Strict order matters.")
    )


def verdict(case, claim):
    return case.with_obligations(replace(claim, consumer="test", reason="A test claim.")).verify().explain(claim.name)


def test_definition_reports_refresh_without_running_the_constructor():
    class Cubic(Curve):
        a = S.One

        def __init__(self, *args, **kwargs):
            raise AssertionError("A declaration report must not construct models")

    assert Cubic.definition_report().checks[0].verdict == "strictly true"
    assert Curve.definition_report().checks[0].verdict == "unresolved"


def test_coordinate_is_protected_from_observations_and_branches_are_independent():
    class Risk(VerificationModel):
        q = InstanceDummy("q", real=True)
        gamma = InstanceDummy("gamma", real=True)
        cost = gamma * q**2

    case = VerificationContext.for_model(Risk)
    claim = convex("cost", wrt="q")
    assert verdict(case, claim).verdict == "conditionally true"
    good = case.observe(q=3, gamma=2)
    bad = case.observe(q=3, gamma=-2)
    assert verdict(good, claim).verdict == "strictly true"
    assert verdict(bad, claim).verdict == "strictly false"
    assert good.expression("cost", wrt="q") == 2 * good.symbols.q**2
    assert verdict(case, claim).verdict == "conditionally true"


def test_assumptions_discharge_conditions_and_contradictions_do_not_prove_everything():
    case = VerificationContext.for_model(Curve)
    claim = positive("a")
    assert verdict(case, claim).verdict == "conditionally true"
    assert verdict(case.assume(case.symbols.a > 0), claim).verdict == "strictly true"
    inconsistent = case.assume(case.symbols.a > 0, case.symbols.a < 0)
    report = inconsistent.with_obligations(replace(claim, consumer="test", reason="Positive parameter.")).verify()
    assert report.consistent is False
    assert all(item.verdict == "unresolved" for item in report.checks)


def test_monotonicity_is_separate_from_a_positive_derivative_certificate():
    case = VerificationContext.for_model(Curve).observe(a=1)
    assert verdict(case, strictly_increasing("value", wrt="x")).verdict == "strictly true"
    assert verdict(case, positive(Derivative(case.symbols.value, Curve.x), wrt=Curve.x)).verdict == "strictly false"
    assert verdict(case, strictly_increasing("value", wrt="x", domain=Interval(0, 1))).verdict == "strictly true"


def test_poles_nonsmoothness_and_uninterpreted_functions():
    class Functions(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        pole = 1 / x
        parameter_pole = exp(x / a)
        kink = Abs(x)
        formal = Function("f")(x)

    case = VerificationContext.for_model(Functions)
    assert verdict(case, continuously_differentiable("pole", wrt="x")).verdict == "strictly false"
    assert (
        verdict(case, continuously_differentiable("pole", wrt="x", domain=Interval.open(0, S.Infinity))).verdict
        == "strictly true"
    )
    parameter = verdict(case, continuously_differentiable("parameter_pole", wrt="x"))
    assert parameter.verdict == "conditionally true"
    assert parameter.condition.subs(case.symbols.a, 0) is S.false
    assert verdict(case, continuously_differentiable("kink", wrt="x")).verdict == "strictly false"
    assert verdict(case, convex("kink", wrt="x")).verdict == "strictly true"
    assert verdict(case, continuously_differentiable("formal", wrt="x")).verdict == "unresolved"
    assert verdict(case, differentiable("kink", wrt="x", at=1)).verdict == "strictly true"


def test_uniform_bounds_require_checked_coordinate_independent_hints():
    class Intensity(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", positive=True)
        k = InstanceDummy("k", positive=True)
        rate = a / (1 + exp(k * x))

    case = VerificationContext.for_model(Intensity)
    claim = derivative_ratio_bound("rate", wrt="x")
    assert verdict(case, claim).verdict == "unresolved"
    hinted = case.with_bound_hint(claim.name, upper=1)
    assert verdict(hinted, claim).verdict == "strictly true"
    assert verdict(case.with_bound_hint(claim.name, upper=-1), claim).verdict == "unresolved"
    assert verdict(case, supremum_below(1 - exp(-Intensity.x), 2, wrt=Intensity.x)).verdict == "unresolved"


def test_curvature_cancellation_retains_zero_slope_exclusions():
    class Exponential(VerificationModel):
        x = Coordinate("x", real=True)
        k = InstanceDummy("k", real=True)
        rate = exp(-k * x)

    case = VerificationContext.for_model(Exponential)
    evidence = verdict(case, derivative_ratio_bound("rate", wrt="x"))
    assert evidence.verdict == "conditionally true"
    assert evidence.condition.subs(case.symbols.k, 0) is S.false
    assert verdict(case.observe(k=0), derivative_ratio_bound("rate", wrt="x")).verdict == "strictly false"


def test_automatic_requirements_preserve_derivatives_divisions_and_evaluation_points():
    class Operations(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        f = Abs(x)
        slope = Subs(Derivative(f, x), x, 0)
        ratio = 1 / a

    report = Operations.definition_report()
    slope = next(item for item in report.checks if item.obligation.consumer == "slope")
    denominator = next(item for item in report.checks if item.obligation.consumer == "ratio")
    assert slope.verdict == "strictly false"
    assert denominator.verdict == "conditionally true"
    assert slope.obligation.property == "differentiable"
    assert slope.obligation.at == 0


def test_replacement_rebinds_contracts_and_observations_without_stale_results():
    class Increasing(Curve):
        a = S.One

    class Decreasing(Curve):
        value = -Curve.x

    class Parent(VerificationModel):
        curve = component(Increasing)

    source = VerificationContext.for_model(Parent)
    assert source.verify().checks[0].verdict == "strictly true"
    changed = source.replace("curve", Decreasing)
    assert changed.verify().checks[0].verdict == "strictly false"
    assert source.verify().checks[0].verdict == "strictly true"


def test_exact_form_and_methods_preserve_their_premises():
    case = VerificationContext.for_model(Curve).observe(a=1)
    identity = equal_to("value", lambda get: get("x") ** 3, wrt="x")
    method = Method(
        "evaluate",
        lambda current: current.expression("value"),
        (replace(identity, consumer="method", reason="This formula is cubic."),),
        uses=("value",),
    )
    result = case.run(method)
    assert result.data == Curve.x**3
    assert result.report.explain(identity.name).verdict == "strictly true"
    with pytest.raises(VerificationError) as error:
        VerificationContext.for_model(Curve).observe(a=2).run(method)
    assert error.value.context.problem.model is Curve
    assert error.value.report.explain(identity.name).verdict == "strictly false"


def test_rules_check_premises_and_circular_rules_stay_unresolved():
    case = VerificationContext.for_model(Curve)
    conclusion = Obligation("conclusion", "a", "rule")
    premise = positive("a", consumer="rule", reason="The parameter must be positive.")
    rule = InferenceRule("test-rule", conclusion, (premise,), "A test trusted rule")
    case = case.with_rule(rule)
    evidence = verdict(case, conclusion)
    assert evidence.kind == "rule"
    assert evidence.verdict == "conditionally true"
    assert evidence.premises[0].obligation == premise
    report = case.with_obligations(replace(conclusion, consumer="test", reason="Inspect recursive evidence.")).verify()
    assert report.explain("conclusion/positive").obligation == premise
    cyclic = replace(rule, premises=(replace(conclusion, consumer="cycle", reason="Circular premise."),))
    assert verdict(replace(case, rules=(cyclic,)), conclusion).verdict == "unresolved"
    assert verdict(case.observe(a=-1), conclusion).verdict == "unresolved"


def test_sampling_never_promotes_a_global_claim_and_foreign_symbols_are_rejected():
    case = VerificationContext.for_model(Curve).observe(a=1)
    sampled = case.sample("value", wrt="x", points=[0, Rational(1, 2), 1])
    assert sampled.kind == "numerical observation"
    assert tuple(float(value) for value in sampled.values) == (0, 0.125, 1)
    with pytest.raises(ModelValidationError):
        case.assume(VerificationContext.for_model(Curve).symbols.a > 0)
    with pytest.raises(ModelDefinitionError):
        case.with_obligations(
            continuously_differentiable("missing", wrt="x", consumer="test", reason="Invalid path.")
        ).verify()


def test_the_proof_budget_is_an_explicit_unresolved_outcome():
    case = VerificationContext.for_model(Curve).with_obligations(
        predicate(Eq(1, 1), name="extra", consumer="test", reason="Budget regression.")
    )
    assert case.verify(max_steps=1).explain("extra").verdict == "unresolved"


def test_reports_escape_names_and_preserve_proof_dependencies():
    case = VerificationContext.for_model(Curve).observe(a=1)
    report = case.with_obligations(
        positive("a", name="<script>", consumer="<consumer>", reason="<reason>", source="javascript:alert(1)")
    ).verify()
    data = report.to_dict()
    assert data["environment"]["sympy"]
    assert data["checks"][0]["dependencies"]
    assert len(data["evidence_sha256"]) == 64
    assert "<script>" not in report.to_html(title="<script>")
    assert "&lt;script&gt;" in report.to_html()
    assert 'href="javascript:' not in report.to_html()
    assert '"schema_version": 1' in report.to_json()


def test_parameter_exponents_do_not_receive_unjustified_regularities():
    class Powers(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = x**a

    assert (
        verdict(VerificationContext.for_model(Powers), continuously_differentiable("value", wrt="x")).verdict
        == "unresolved"
    )


def test_domains_and_observation_conflicts_are_not_silently_assumed():
    class Restricted(VerificationModel):
        x = Coordinate("x", positive=True)
        a = InstanceDummy("a", positive=True)
        value = x

    case = VerificationContext.for_model(Restricted)
    assert verdict(case, positive("value", wrt="x")).verdict == "unresolved"
    assert verdict(case, positive("value", wrt="x", domain=Interval.open(0, S.Infinity))).verdict == "strictly true"
    assert case.observe(a=-1).verify().consistent is False
    assert case.observe(a=1).observe(a=2).verify().consistent is False


def test_complex_values_cannot_acquire_ordered_properties_from_zero_curvature():
    class ComplexCurve(VerificationModel):
        x = Coordinate("x", real=True)
        value = I * x

    case = VerificationContext.for_model(ComplexCurve)
    assert verdict(case, convex("value", wrt="x")).verdict == "unresolved"
    assert verdict(case, positive(I)).verdict == "strictly false"
    assert verdict(case, nonzero(I)).verdict == "strictly true"


def test_methods_check_consumed_annotations_without_blocking_unrelated_claims():
    class TwoCurves(VerificationModel):
        x = Coordinate("x", real=True)
        invalid = specifies(-x, strictly_increasing("invalid", wrt="x", consumer="order", reason="Order matters."))
        valid = x**3

    case = VerificationContext.for_model(TwoCurves)
    with pytest.raises(VerificationError):
        case.run(Method("read_invalid", lambda current: current.expression("invalid"), uses=("invalid",)))
    assert (
        case.run(Method("read_valid", lambda current: current.expression("valid"), uses=("valid",))).data
        == TwoCurves.x**3
    )

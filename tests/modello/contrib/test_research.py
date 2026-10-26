"""Research claims retain scope, evidence, and honest unresolved outcomes."""

from dataclasses import replace

import pytest
from sympy import Abs, Derivative, Eq, Function, I, Interval, Rational, S, Subs, exp

from modello import Coordinate, InstanceDummy, annotate, component
from modello.contrib.research import (
    Claim,
    Method,
    Requirement,
    Requires,
    ResearchCase,
    ResearchError,
    ResearchModel,
    Theorem,
    convex,
    curvature_bound,
    differentiable,
    exact_form,
    increasing,
    nonzero,
    positive,
    predicate,
    smooth,
    uniform_bound,
    upper_bound,
)
from modello.errors import ModelDefinitionError, ModelValidationError


class Curve(ResearchModel):
    x = Coordinate("x", real=True)
    a = InstanceDummy("a", real=True)
    value = annotate(
        a * x**3, Requires(Requirement(increasing("value", wrt="x"), "optimizer", "Strict order matters."))
    )


def verdict(case, claim):
    return case.require(Requirement(claim, "test", "A test claim.")).check().explain(claim.name)


def test_definition_reports_refresh_without_running_the_constructor():
    class Cubic(Curve):
        a = S.One

        def __init__(self, *args, **kwargs):
            raise AssertionError("A declaration report must not construct models")

    assert Cubic.definition_report().checks[0].verdict == "strictly true"
    assert Curve.definition_report().checks[0].verdict == "unresolved"


def test_coordinate_is_protected_from_observations_and_branches_are_independent():
    class Risk(ResearchModel):
        q = InstanceDummy("q", real=True)
        gamma = InstanceDummy("gamma", real=True)
        cost = gamma * q**2

    case = ResearchCase.for_model(Risk)
    claim = convex("cost", wrt="q")
    assert verdict(case, claim).verdict == "conditionally true"
    good = case.observe(q=3, gamma=2)
    bad = case.observe(q=3, gamma=-2)
    assert verdict(good, claim).verdict == "strictly true"
    assert verdict(bad, claim).verdict == "strictly false"
    assert good.expression("cost", wrt="q") == 2 * good.symbols.q**2
    assert verdict(case, claim).verdict == "conditionally true"


def test_assumptions_discharge_conditions_and_contradictions_do_not_prove_everything():
    case = ResearchCase.for_model(Curve)
    claim = positive("a")
    assert verdict(case, claim).verdict == "conditionally true"
    assert verdict(case.assume(case.symbols.a > 0), claim).verdict == "strictly true"
    inconsistent = case.assume(case.symbols.a > 0, case.symbols.a < 0)
    report = inconsistent.require(Requirement(claim, "test", "Positive parameter.")).check()
    assert report.consistent is False
    assert all(item.verdict == "unresolved" for item in report.checks)


def test_monotonicity_is_separate_from_a_positive_derivative_certificate():
    case = ResearchCase.for_model(Curve).observe(a=1)
    assert verdict(case, increasing("value", wrt="x")).verdict == "strictly true"
    assert verdict(case, positive(Derivative(case.symbols.value, Curve.x), wrt=Curve.x)).verdict == "strictly false"
    assert verdict(case, increasing("value", wrt="x", domain=Interval(0, 1))).verdict == "strictly true"


def test_poles_nonsmoothness_and_uninterpreted_functions():
    class Functions(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        pole = 1 / x
        parameter_pole = exp(x / a)
        kink = Abs(x)
        formal = Function("f")(x)

    case = ResearchCase.for_model(Functions)
    assert verdict(case, smooth("pole", wrt="x")).verdict == "strictly false"
    assert verdict(case, smooth("pole", wrt="x", domain=Interval.open(0, S.Infinity))).verdict == "strictly true"
    parameter = verdict(case, smooth("parameter_pole", wrt="x"))
    assert parameter.verdict == "conditionally true"
    assert parameter.condition.subs(case.symbols.a, 0) is S.false
    assert verdict(case, smooth("kink", wrt="x")).verdict == "strictly false"
    assert verdict(case, convex("kink", wrt="x")).verdict == "strictly true"
    assert verdict(case, smooth("formal", wrt="x")).verdict == "unresolved"
    assert verdict(case, differentiable("kink", wrt="x", at=1)).verdict == "strictly true"


def test_uniform_bounds_require_checked_coordinate_independent_hints():
    class Intensity(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", positive=True)
        k = InstanceDummy("k", positive=True)
        rate = a / (1 + exp(k * x))

    case = ResearchCase.for_model(Intensity)
    claim = curvature_bound("rate", wrt="x")
    assert verdict(case, claim).verdict == "unresolved"
    hinted = case.with_hint(claim.name, upper_bound(1))
    assert verdict(hinted, claim).verdict == "strictly true"
    assert verdict(case.with_hint(claim.name, upper_bound(-1)), claim).verdict == "unresolved"
    assert verdict(case, uniform_bound(1 - exp(-Intensity.x), 2, wrt=Intensity.x)).verdict == "unresolved"


def test_curvature_cancellation_retains_zero_slope_exclusions():
    class Exponential(ResearchModel):
        x = Coordinate("x", real=True)
        k = InstanceDummy("k", real=True)
        rate = exp(-k * x)

    case = ResearchCase.for_model(Exponential)
    evidence = verdict(case, curvature_bound("rate", wrt="x"))
    assert evidence.verdict == "conditionally true"
    assert evidence.condition.subs(case.symbols.k, 0) is S.false
    assert verdict(case.observe(k=0), curvature_bound("rate", wrt="x")).verdict == "strictly false"


def test_automatic_requirements_preserve_derivatives_divisions_and_evaluation_points():
    class Operations(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        f = Abs(x)
        slope = Subs(Derivative(f, x), x, 0)
        ratio = 1 / a

    report = Operations.definition_report()
    slope = next(item for item in report.checks if item.requirement.consumer == "slope")
    denominator = next(item for item in report.checks if item.requirement.consumer == "ratio")
    assert slope.verdict == "strictly false"
    assert denominator.verdict == "conditionally true"
    assert slope.requirement.claim.property == "differentiable"
    assert slope.requirement.claim.at == 0


def test_replacement_rebinds_contracts_and_observations_without_stale_results():
    class Increasing(Curve):
        a = S.One

    class Decreasing(Curve):
        value = -Curve.x

    class Parent(ResearchModel):
        curve = component(Increasing)

    source = ResearchCase.for_model(Parent)
    assert source.check().checks[0].verdict == "strictly true"
    changed = source.replace("curve", Decreasing)
    assert changed.check().checks[0].verdict == "strictly false"
    assert source.check().checks[0].verdict == "strictly true"


def test_exact_form_and_methods_preserve_their_premises():
    case = ResearchCase.for_model(Curve).observe(a=1)
    identity = exact_form("value", lambda get: get("x") ** 3, wrt="x")
    method = Method(
        "evaluate",
        lambda current: current.expression("value"),
        (Requirement(identity, "method", "This formula is cubic."),),
        uses=("value",),
    )
    result = case.run(method)
    assert result.data == Curve.x**3
    assert result.report.explain(identity.name).verdict == "strictly true"
    with pytest.raises(ResearchError) as error:
        ResearchCase.for_model(Curve).observe(a=2).run(method)
    assert error.value.case.problem.model is Curve
    assert error.value.report.explain(identity.name).verdict == "strictly false"


def test_theorems_check_premises_and_circular_rules_stay_unresolved():
    case = ResearchCase.for_model(Curve)
    conclusion = Claim("conclusion", "a", "theorem")
    premise = Requirement(positive("a"), "theorem", "The parameter must be positive.")
    theorem = Theorem("test-rule", conclusion, (premise,), "A test trusted rule")
    case = case.with_theorem(theorem)
    evidence = verdict(case, conclusion)
    assert evidence.kind == "theorem"
    assert evidence.verdict == "conditionally true"
    assert evidence.premises[0].requirement == premise
    report = case.require(Requirement(conclusion, "test", "Inspect recursive evidence.")).check()
    assert report.explain("conclusion/positive").requirement == premise
    cyclic = replace(theorem, premises=(Requirement(conclusion, "cycle", "Circular premise."),))
    assert verdict(replace(case, theorems=(cyclic,)), conclusion).verdict == "unresolved"
    assert verdict(case.observe(a=-1), conclusion).verdict == "unresolved"


def test_sampling_never_promotes_a_global_claim_and_foreign_symbols_are_rejected():
    case = ResearchCase.for_model(Curve).observe(a=1)
    sampled = case.sample("value", wrt="x", points=[0, Rational(1, 2), 1])
    assert sampled.kind == "numerical observation"
    assert tuple(float(value) for value in sampled.values) == (0, 0.125, 1)
    with pytest.raises(ModelValidationError):
        case.assume(ResearchCase.for_model(Curve).symbols.a > 0)
    with pytest.raises(ModelDefinitionError):
        case.require(Requirement(smooth("missing", wrt="x"), "test", "Invalid path.")).check()


def test_the_proof_budget_is_an_explicit_unresolved_outcome():
    case = ResearchCase.for_model(Curve).require(
        Requirement(predicate(Eq(1, 1), name="extra"), "test", "Budget regression.")
    )
    assert case.check(max_steps=1).explain("extra").verdict == "unresolved"


def test_reports_escape_names_and_preserve_proof_dependencies():
    case = ResearchCase.for_model(Curve).observe(a=1)
    report = case.require(
        Requirement(positive("a", name="<script>"), "<consumer>", "<reason>", "javascript:alert(1)")
    ).check()
    data = report.to_dict()
    assert data["environment"]["sympy"]
    assert data["checks"][0]["dependencies"]
    assert len(data["evidence_sha256"]) == 64
    assert "<script>" not in report.to_html(title="<script>")
    assert "&lt;script&gt;" in report.to_html()
    assert 'href="javascript:' not in report.to_html()
    assert '"schema_version": 1' in report.to_json()


def test_parameter_exponents_do_not_receive_unjustified_regularities():
    class Powers(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = x**a

    assert verdict(ResearchCase.for_model(Powers), smooth("value", wrt="x")).verdict == "unresolved"


def test_domains_and_observation_conflicts_are_not_silently_assumed():
    class Restricted(ResearchModel):
        x = Coordinate("x", positive=True)
        a = InstanceDummy("a", positive=True)
        value = x

    case = ResearchCase.for_model(Restricted)
    assert verdict(case, positive("value", wrt="x")).verdict == "unresolved"
    assert verdict(case, positive("value", wrt="x", domain=Interval.open(0, S.Infinity))).verdict == "strictly true"
    assert case.observe(a=-1).check().consistent is False
    assert case.observe(a=1).observe(a=2).check().consistent is False


def test_complex_values_cannot_acquire_ordered_properties_from_zero_curvature():
    class ComplexCurve(ResearchModel):
        x = Coordinate("x", real=True)
        value = I * x

    case = ResearchCase.for_model(ComplexCurve)
    assert verdict(case, convex("value", wrt="x")).verdict == "unresolved"
    assert verdict(case, positive(I)).verdict == "strictly false"
    assert verdict(case, nonzero(I)).verdict == "strictly true"


def test_methods_check_consumed_annotations_without_blocking_unrelated_claims():
    class TwoCurves(ResearchModel):
        x = Coordinate("x", real=True)
        invalid = annotate(-x, Requires(Requirement(increasing("invalid", wrt="x"), "order", "Order matters.")))
        valid = x**3

    case = ResearchCase.for_model(TwoCurves)
    with pytest.raises(ResearchError):
        case.run(Method("read_invalid", lambda current: current.expression("invalid"), uses=("invalid",)))
    assert (
        case.run(Method("read_valid", lambda current: current.expression("valid"), uses=("valid",))).data
        == TwoCurves.x**3
    )

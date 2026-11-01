"""One requirement value serves declarations, cases, methods, and rules."""

from dataclasses import replace

import pytest
from sympy import Derivative, Function, Interval, S, Subs

import modello.contrib.verification as verification
from modello import Coordinate, InstanceDummy, component
from modello.contrib.verification import (
    InferenceRule,
    Method,
    Obligation,
    VerificationContext,
    VerificationError,
    VerificationModel,
    positive,
    specifies,
)
from modello.errors import ModelDefinitionError, ModelInputError


def test_one_requirement_value_works_at_every_consumer():
    condition = positive("a", name="admissible", reason="This computation needs a positive parameter.")
    assert isinstance(condition, Obligation)

    class Parameter(VerificationModel):
        a = specifies(InstanceDummy("a", real=True), condition)

    method = Method("read", lambda current: current.expression("a"), preconditions=(condition,), uses=("a",))
    base = VerificationContext.for_model(Parameter)
    result = base.observe(a=2).with_obligations(condition).run(method)
    assert result.data == 2
    assert len(result.report.checks) == 2  # field declaration and shared case/method condition
    assert all(item.obligation == condition for item in result.report.checks)
    with pytest.raises(VerificationError) as failure:
        base.observe(a=-1).run(method)
    assert failure.value.report.explain("a/admissible").verdict == "strictly false"


def test_rule_matching_ignores_motivation_without_ignoring_conflicting_ids():
    class Parameter(VerificationModel):
        a = InstanceDummy("a", real=True)

    conclusion = Obligation("admissible", "a", "rule")
    requested = replace(conclusion, consumer="calculation", reason="A consumer needs this conclusion.")
    case = VerificationContext.for_model(Parameter).observe(a=2)
    case = case.with_rule(InferenceRule("rule", conclusion, (positive("a"),), "Explicit test rule"))
    evidence = case.with_obligations(requested).verify().explain("admissible")
    assert evidence.kind == "rule" and evidence.verdict == "strictly true"
    assert evidence.obligation is requested
    with pytest.raises(ModelDefinitionError, match="unique"):
        case.with_obligations(requested, replace(requested, reason="A conflicting declaration.")).verify()


def test_model_requirements_refresh_definition_evidence_and_follow_component_scope():
    class Parameter(VerificationModel):
        a = InstanceDummy("a", real=True)

    earlier = VerificationContext.for_model(Parameter)
    constrained = specifies(Parameter, positive("a", name="admissible"))
    assert constrained is not Parameter and issubclass(constrained, Parameter)
    assert Parameter.definition_report().checks == earlier.verify().checks == ()
    assert constrained.definition_report().explain("admissible").verdict == "conditionally true"

    class Parent(VerificationModel):
        a = InstanceDummy("outer", real=True)
        child = component(constrained)

    case = VerificationContext.for_model(Parent).observe(a=3, child=constrained("child", a=-1))
    assert case.verify().explain("child/admissible").verdict == "strictly false"
    assert case.run(Method("outer", lambda current: current.expression("a"), uses=("a",))).data == 3
    with pytest.raises(VerificationError):
        case.run(Method("child", lambda current: current.expression("child.a"), uses=("child.a",)))


def test_hint_values_are_scalar_proposals():
    case = VerificationContext.for_model(VerificationModel)
    hinted = case.with_bound_hint("bound", upper=2)
    assert dict(hinted.bound_hints)["bound"] == S(2)
    with pytest.raises(ModelInputError, match="scalar expression"):
        case.with_bound_hint("bound", upper=object())
    with pytest.raises(ModelInputError, match="scalar expression"):
        case.with_bound_hint("bound", upper=S.true)


def test_specifies_rejects_non_obligation_inputs():
    with pytest.raises(ModelDefinitionError, match="specifies accepts Obligation records"):
        specifies(S.One, S.One)  # type: ignore[arg-type]
    with pytest.raises(ModelDefinitionError, match="name"):
        Obligation(positive("a"), "consumer", "positive")  # type: ignore[arg-type]


def test_model_attachment_preserves_inherited_requirements_and_earlier_classes():
    class Parameter(VerificationModel):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    first = specifies(Parameter, positive("a", name="positive_a"))
    second = specifies(first, positive("b", name="positive_b"))
    assert Parameter.definition_report().checks == ()
    assert {item.id for item in first.definition_report().checks} == {"positive_a"}
    assert {item.id for item in second.definition_report().checks} == {"positive_a", "positive_b"}


@pytest.mark.parametrize("intrinsic", (False, True))
def test_strict_polynomial_order_keeps_zero_coefficients_out_of_the_certificate(intrinsic):
    class Cubic(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", nonnegative=True) if intrinsic else InstanceDummy("a", real=True)
        value = specifies(a * x**3, verification.strictly_increasing("value", wrt="x"))

    case = VerificationContext.for_model(Cubic)
    if not intrinsic:
        case = case.assume(case.symbols.a >= 0)
    assert case.verify().explain("value/strictly_increasing").verdict == "conditionally true"
    if intrinsic:
        assert Cubic.definition_report().explain("value/strictly_increasing").verdict == "conditionally true"
    method = Method("read", lambda current: 1, uses=("value",))
    with pytest.raises(VerificationError):
        case.run(method)
    assert case.assume(case.symbols.a > 0).run(method).data == 1
    assert case.observe(a=0).verify().explain("value/strictly_increasing").verdict == "strictly false"


def test_requirement_names_cannot_bypass_explicit_or_declared_method_gates():
    class Parameter(VerificationModel):
        a = specifies(InstanceDummy("a", real=True), positive("a", name="auto:custom"))
        value = 2 * a

    case = VerificationContext.for_model(Parameter).observe(a=-1)
    explicit = positive("a", name="user/auto:1")
    with pytest.raises(VerificationError) as failure:
        case.run(Method("explicit", lambda current: "EXECUTED", preconditions=(explicit,)))
    assert failure.value.report.explain(explicit.name).verdict == "strictly false"
    with pytest.raises(VerificationError) as failure:
        case.run(Method("declared", lambda current: "EXECUTED", uses=("value",)))
    assert failure.value.report.explain("a/auto:custom").verdict == "strictly false"


def test_rule_premise_ids_reuse_identical_checks_and_reject_conflicts():
    class Parameters(VerificationModel):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    conclusion = Obligation("combined", "a", "rule")
    with pytest.raises(ModelDefinitionError, match="premise names must be unique"):
        InferenceRule("conflict", conclusion, (positive("a"), positive("b")), "Explicit test rule")
    premise = positive("a")
    rule = InferenceRule("reused", conclusion, (premise, premise), "Explicit test rule")
    assert rule.premises == (premise,)
    report = (
        VerificationContext.for_model(Parameters).observe(a=2).with_rule(rule).with_obligations(conclusion).verify()
    )
    assert report.explain("combined/positive").verdict == "strictly true"


@pytest.mark.parametrize("nested", (False, True))
def test_rule_premise_ids_cannot_collide_with_other_evidence(nested):
    class Parameters(VerificationModel):
        a = InstanceDummy("a", positive=True)
        b = InstanceDummy("b", positive=True)

    conclusion = Obligation("combined", "a", "rule")
    child = Obligation("child", "a", "rule")
    premises = (child, positive("b", name="child/positive")) if nested else (positive("a"),)
    case = VerificationContext.for_model(Parameters).with_rule(InferenceRule("rule", conclusion, premises, "Test rule"))
    case = case.with_obligations(conclusion)
    if nested:
        case = case.with_rule(InferenceRule("child rule", child, (positive("a"),), "Test rule"))
    else:
        case = case.with_obligations(positive("b", name="combined/positive"))
    with pytest.raises(ModelDefinitionError, match="Evidence IDs must be unique"):
        case.verify()


@pytest.mark.parametrize(
    "property",
    (
        "positive",
        "nonnegative",
        "nonzero",
        "strictly_increasing",
        "strictly_decreasing",
        "convex",
        "concave",
        "continuously_differentiable",
        "differentiable",
        "limit",
        "supremum_below",
        "derivative_ratio_bound",
        "equal_to",
    ),
)
def test_empty_requirement_domains_neither_certify_nor_refute_scalar_claims(property):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = S.NegativeOne

    claim = Obligation("empty", "value", property, wrt="x", domain=S.EmptySet, bound=S(2), expected=S.One)
    evidence = VerificationContext.for_model(Curve).verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "unresolved"
    assert not evidence.witness
    assert "domain is empty" in " ".join(evidence.steps)


@pytest.mark.parametrize("trusted", (False, True))
@pytest.mark.parametrize("domain_source", ("literal", "observed", "native", "unsupported_native"))
def test_empty_domains_block_methods_after_binding_and_before_rule_application(domain_source, trusted):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        start = InstanceDummy("start", real=True)
        stop = InstanceDummy("stop", real=True)
        value = S.One

    case = VerificationContext.for_model(Curve)
    if domain_source == "observed":
        domain = Interval(case.symbols.start, case.symbols.stop)
        case = case.observe(start=2, stop=1)
    elif domain_source == "native":
        domain = Interval(Subs(Derivative(Curve.x**2, Curve.x, evaluate=False), Curve.x, 1), 1)
    elif domain_source == "unsupported_native":

        class Missing(Function):
            def _eval_derivative(self, variable):
                raise NotImplementedError("No derivative implementation")

        domain = Interval(Subs(Derivative(Missing(Curve.x), Curve.x, evaluate=False), Curve.x, 0), 1)
    else:
        domain = Interval(1, 0)
    claim = positive("value", wrt="x", domain=domain)
    if trusted:
        case = case.with_rule(InferenceRule("trusted rule", claim, (), "Test rule"))
    calls = []
    with pytest.raises(VerificationError) as failure:
        case.run(Method("read", lambda current: calls.append(current), preconditions=(claim,)))
    evidence = failure.value.report.explain(claim.name)
    assert not calls
    assert evidence.verdict == "unresolved"
    assert ("nonempty" if domain_source == "unsupported_native" else "domain is empty") in " ".join(evidence.steps)


def test_empty_declared_domains_remain_open_in_definition_reports():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = specifies(S.NegativeOne, positive("value", wrt="x", domain=S.EmptySet))

    assert Curve.definition_report().explain("value/positive").verdict == "unresolved"


@pytest.mark.parametrize("value", (-1, 1))
def test_symbolic_domains_do_not_assume_an_unstated_endpoint_order(value):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        start = InstanceDummy("start", real=True)
        stop = InstanceDummy("stop", real=True)
        value = InstanceDummy("value", real=True)

    case = VerificationContext.for_model(Curve).observe(value=value)
    claim = positive("value", wrt="x", domain=Interval(case.symbols.start, case.symbols.stop))
    evidence = case.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "unresolved"
    assert "nonempty" in " ".join(evidence.steps)
    assert case.observe(start=2, stop=1).verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    assert case.observe(start=0, stop=1).verify(obligations=(claim,)).explain(claim.name).verdict == (
        "strictly true" if value > 0 else "strictly false"
    )

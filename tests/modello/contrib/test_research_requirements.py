"""One requirement value serves declarations, cases, methods, and rules."""

from dataclasses import replace

import pytest
from sympy import S

import modello.contrib.research as research
from modello import Coordinate, InstanceDummy, component
from modello.contrib.research import (
    Method,
    Requirement,
    ResearchCase,
    ResearchError,
    ResearchModel,
    Theorem,
    positive,
    requires,
)
from modello.errors import ModelDefinitionError, ModelInputError


def test_one_requirement_value_works_at_every_consumer():
    condition = positive("a", name="admissible", reason="This computation needs a positive parameter.")
    assert isinstance(condition, Requirement)

    class Parameter(ResearchModel):
        a = requires(InstanceDummy("a", real=True), condition)

    method = Method("read", lambda current: current.expression("a"), requirements=(condition,), uses=("a",))
    base = ResearchCase.for_model(Parameter)
    result = base.observe(a=2).require(condition).run(method)
    assert result.data == 2
    assert len(result.report.checks) == 2  # field declaration and shared case/method condition
    assert all(item.requirement == condition for item in result.report.checks)
    with pytest.raises(ResearchError) as failure:
        base.observe(a=-1).run(method)
    assert failure.value.report.explain("a/admissible").verdict == "strictly false"


def test_theorem_matching_ignores_motivation_without_ignoring_conflicting_ids():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    conclusion = Requirement("admissible", "a", "theorem")
    requested = replace(conclusion, consumer="calculation", reason="A consumer needs this conclusion.")
    case = ResearchCase.for_model(Parameter).observe(a=2)
    case = case.with_theorem(Theorem("rule", conclusion, (positive("a"),), "Explicit test rule"))
    evidence = case.require(requested).check().explain("admissible")
    assert evidence.kind == "theorem" and evidence.verdict == "strictly true"
    assert evidence.requirement is requested
    with pytest.raises(ModelDefinitionError, match="unique"):
        case.require(requested, replace(requested, reason="A conflicting declaration.")).check()


def test_model_requirements_refresh_definition_evidence_and_follow_component_scope():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)

    earlier = ResearchCase.for_model(Parameter)
    constrained = requires(Parameter, positive("a", name="admissible"))
    assert constrained is not Parameter and issubclass(constrained, Parameter)
    assert Parameter.definition_report().checks == earlier.check().checks == ()
    assert constrained.definition_report().explain("admissible").verdict == "conditionally true"

    class Parent(ResearchModel):
        a = InstanceDummy("outer", real=True)
        child = component(constrained)

    case = ResearchCase.for_model(Parent).observe(a=3, child=constrained("child", a=-1))
    assert case.check().explain("child/admissible").verdict == "strictly false"
    assert case.run(Method("outer", lambda current: current.expression("a"), uses=("a",))).data == 3
    with pytest.raises(ResearchError):
        case.run(Method("child", lambda current: current.expression("child.a"), uses=("child.a",)))


def test_hint_values_are_scalar_proposals_and_removed_wrappers_are_not_exported():
    case = ResearchCase.for_model(ResearchModel)
    hinted = case.with_hint("bound", upper=2)
    assert dict(hinted.hints)["bound"] == S(2)
    with pytest.raises(ModelInputError, match="scalar expression"):
        case.with_hint("bound", upper=object())
    with pytest.raises(ModelInputError, match="scalar expression"):
        case.with_hint("bound", upper=S.true)
    for name in ("Claim", "Requires", "ProofHint", "upper_bound"):
        assert not hasattr(research, name)


def test_requires_rejects_nonrequirements_and_ambiguous_legacy_construction():
    with pytest.raises(ModelDefinitionError, match="property requirements"):
        requires(S.One, S.One)  # type: ignore[arg-type]
    with pytest.raises(ModelDefinitionError, match="name"):
        Requirement(positive("a"), "consumer", "positive")  # type: ignore[arg-type]


def test_model_attachment_preserves_inherited_requirements_and_earlier_classes():
    class Parameter(ResearchModel):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    first = requires(Parameter, positive("a", name="positive_a"))
    second = requires(first, positive("b", name="positive_b"))
    assert Parameter.definition_report().checks == ()
    assert {item.id for item in first.definition_report().checks} == {"positive_a"}
    assert {item.id for item in second.definition_report().checks} == {"positive_a", "positive_b"}


@pytest.mark.parametrize("intrinsic", (False, True))
def test_strict_polynomial_order_keeps_zero_coefficients_out_of_the_certificate(intrinsic):
    class Cubic(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", nonnegative=True) if intrinsic else InstanceDummy("a", real=True)
        value = requires(a * x**3, research.increasing("value", wrt="x"))

    case = ResearchCase.for_model(Cubic)
    if not intrinsic:
        case = case.assume(case.symbols.a >= 0)
    assert case.check().explain("value/increasing").verdict == "conditionally true"
    if intrinsic:
        assert Cubic.definition_report().explain("value/increasing").verdict == "conditionally true"
    method = Method("read", lambda current: 1, uses=("value",))
    with pytest.raises(ResearchError):
        case.run(method)
    assert case.assume(case.symbols.a > 0).run(method).data == 1
    assert case.observe(a=0).check().explain("value/increasing").verdict == "strictly false"


def test_requirement_names_cannot_bypass_explicit_or_declared_method_gates():
    class Parameter(ResearchModel):
        a = requires(InstanceDummy("a", real=True), positive("a", name="auto:custom"))
        value = 2 * a

    case = ResearchCase.for_model(Parameter).observe(a=-1)
    explicit = positive("a", name="user/auto:1")
    with pytest.raises(ResearchError) as failure:
        case.run(Method("explicit", lambda current: "EXECUTED", requirements=(explicit,)))
    assert failure.value.report.explain(explicit.name).verdict == "strictly false"
    with pytest.raises(ResearchError) as failure:
        case.run(Method("declared", lambda current: "EXECUTED", uses=("value",)))
    assert failure.value.report.explain("a/auto:custom").verdict == "strictly false"


def test_theorem_premise_ids_reuse_identical_checks_and_reject_conflicts():
    class Parameters(ResearchModel):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    conclusion = Requirement("combined", "a", "theorem")
    with pytest.raises(ModelDefinitionError, match="premise names must be unique"):
        Theorem("conflict", conclusion, (positive("a"), positive("b")), "Explicit test rule")
    premise = positive("a")
    rule = Theorem("reused", conclusion, (premise, premise), "Explicit test rule")
    assert rule.premises == (premise,)
    report = ResearchCase.for_model(Parameters).observe(a=2).with_theorem(rule).require(conclusion).check()
    assert report.explain("combined/positive").verdict == "strictly true"


@pytest.mark.parametrize("nested", (False, True))
def test_theorem_premise_ids_cannot_collide_with_other_evidence(nested):
    class Parameters(ResearchModel):
        a = InstanceDummy("a", positive=True)
        b = InstanceDummy("b", positive=True)

    conclusion = Requirement("combined", "a", "theorem")
    child = Requirement("child", "a", "theorem")
    premises = (child, positive("b", name="child/positive")) if nested else (positive("a"),)
    case = ResearchCase.for_model(Parameters).with_theorem(Theorem("rule", conclusion, premises, "Test rule"))
    case = case.require(conclusion)
    if nested:
        case = case.with_theorem(Theorem("child rule", child, (positive("a"),), "Test rule"))
    else:
        case = case.require(positive("b", name="combined/positive"))
    with pytest.raises(ModelDefinitionError, match="Evidence IDs must be unique"):
        case.check()

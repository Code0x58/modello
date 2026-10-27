"""Boundary cases uncovered by independent review of the research workflow."""

from dataclasses import replace

import pytest
from sympy import Abs, Derivative, Eq, Interval, Ne, Piecewise, S, Subs, exp, frac, sin

from modello import Coordinate, InstanceDummy, component
from modello.contrib.research import (
    Method,
    ResearchCase,
    ResearchError,
    ResearchModel,
    curvature_bound,
    decreasing,
    differentiable,
    exact_form,
    increasing,
    nonzero,
    positive,
    requires,
    uniform_bound,
    vanishes,
)


def check(case, claim):
    return case.require(replace(claim, consumer="test", reason="Review regression.")).check().explain(claim.name)


def test_discontinuity_does_not_refute_definedness_order_or_identity():
    class Discontinuous(ResearchModel):
        x = Coordinate("x", real=True)
        value = frac(x) + 1

    case = ResearchCase.for_model(Discontinuous)
    for claim in (
        positive("value", wrt="x", domain=Interval(0, 1)),
        nonzero("value", wrt="x", domain=Interval(0, 1)),
        exact_form("value", lambda get: get("value"), wrt="x", domain=Interval(0, 1)),
    ):
        assert check(case, claim).verdict == "strictly true"


def test_nonpolynomial_certificate_failure_is_an_inapplicable_attempt():
    class Functions(ResearchModel):
        x = Coordinate("x", real=True)
        value = requires(exp(x), decreasing("value", wrt="x", consumer="test", reason="A wrong claim."))
        wave = sin(x)

    assert Functions.definition_report().checks[0].verdict == "strictly false"
    assert check(ResearchCase.for_model(Functions), increasing("wave", wrt="x")).verdict in {
        "unresolved",
        "strictly false",
    }


def test_native_declaration_metadata_is_bound_in_its_component_scope():
    class Native(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = requires(a * x, increasing(a * x, wrt=x, consumer="test", reason="Native expression."))

    class Parent(ResearchModel):
        curve = component(Native)

    assert Native.definition_report().checks[0].verdict == "conditionally true"
    case = ResearchCase.for_model(Parent).observe(curve=Native("curve", a=2))
    assert case.check().checks[0].verdict == "strictly true"


def test_replacement_rebinds_native_coordinates_and_parameters():
    class Old(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", positive=True)
        value = a * x

    class New(Old):
        x = Coordinate("y", real=True)
        value = Old.a * x

    class Parent(ResearchModel):
        curve = component(Old)

    case = ResearchCase.for_model(Parent).observe(curve=Old("curve", a=2))
    claim = increasing(case.symbols.curve.value, wrt=Old.x)
    original = case.require(replace(claim, consumer="test", reason="Native curve coordinate."))
    changed = original.replace("curve", New)
    assert original.check().checks[0].verdict == changed.check().checks[0].verdict == "strictly true"
    assert changed.check().checks[0].coordinate == New.x


def test_all_parameter_payloads_are_specialized_and_audited():
    class Parameterized(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = x**2
        kink = Abs(x)

    case = ResearchCase.for_model(Parameterized)
    claim = uniform_bound("value", case.symbols.a, wrt="x", domain=Interval(0, 1))
    evidence = check(case.observe(a=5).with_hint(claim.name, upper=1), claim)
    assert evidence.verdict == "strictly true"
    assert evidence.resolved.bound == 5
    claim = uniform_bound("value", case.symbols.a + 1, wrt="x", domain=Interval(0, case.symbols.a))
    assert check(case.observe(a=1).with_hint(claim.name, upper=1), claim).verdict == "strictly true"
    local = differentiable("kink", wrt="x", at=case.symbols.a)
    assert check(case.observe(a=1), local).verdict == "strictly true"


def test_tail_limits_do_not_require_interior_regularity():
    class Pole(ResearchModel):
        x = Coordinate("x", real=True)
        value = 1 / x

    assert check(ResearchCase.for_model(Pole), vanishes("value", wrt="x")).verdict == "strictly true"


def test_substitution_guards_are_local_for_direct_and_named_curves():
    class Evaluated(ResearchModel):
        x = Coordinate("x", real=True)
        pole = 1 / x
        direct = Subs(1 / x, x, 1)
        named = Subs(pole, x, 1)
        slope = Subs(Derivative(pole, x), x, 1)

    case = ResearchCase.for_model(Evaluated)
    for path, expected in (("direct", 1), ("named", 1), ("slope", -1)):
        method = Method(path, lambda current, field=path: current.expression(field).doit(), uses=(path,))
        assert case.run(method).data == expected
    assert check(case, nonzero("slope")).verdict == "strictly true"
    assert check(case, exact_form("slope", -1)).verdict == "strictly true"


def test_local_differentiability_retains_the_original_neighborhood_domain():
    class Isolated(ResearchModel):
        x = Coordinate("x", real=True)
        value = Piecewise((x**2, Eq(x, 0)), (S.NaN, True))

    assert check(ResearchCase.for_model(Isolated), differentiable("value", wrt="x", at=0)).verdict != "strictly true"


def test_definedness_retains_parameter_dependent_coordinate_guards():
    class Partial(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = Piecewise((x**2 + 1, x > a), (S.NaN, True))

    case = ResearchCase.for_model(Partial)
    for claim in (positive("value", wrt="x"), exact_form("value", lambda get: get("value"), wrt="x")):
        assert check(case, claim).verdict != "strictly true"


def test_audits_and_fingerprints_include_the_compared_proposition():
    class Quadratic(ResearchModel):
        x = Coordinate("x", real=True)
        value = x**2

    case = ResearchCase.for_model(Quadratic)
    first = case.require(exact_form("value", S(2), wrt="x", consumer="test", reason="Compare a form.")).check()
    second = case.require(exact_form("value", S(3), wrt="x", consumer="test", reason="Compare a form.")).check()
    assert first.to_dict()["checks"][0]["expected"] == "2"
    assert second.to_dict()["checks"][0]["expected"] == "3"
    assert first.to_dict()["evidence_sha256"] != second.to_dict()["evidence_sha256"]
    assert "Expected form:" in first.to_html()


def test_differentiability_remains_open_when_only_continuity_of_derivatives_fails():
    class Explicit(ResearchModel):
        x = Coordinate("x", real=True)
        value = Piecewise((x**4 * sin(1 / x), Ne(x, 0)), (0, True))

    assert (
        check(ResearchCase.for_model(Explicit), differentiable("value", wrt="x", order=2)).verdict != "strictly false"
    )


def test_curvature_ratio_does_not_require_continuity_of_the_second_derivative():
    class Oscillating(ResearchModel):
        x = Coordinate("x", real=True)
        value = x + Piecewise((x**4 * sin(1 / x), Ne(x, 0)), (0, True))

    claim = curvature_bound("value", wrt="x", domain=Interval(-S.One / 10, S.One / 10))
    assert check(ResearchCase.for_model(Oscillating), claim).verdict != "strictly false"


def test_automatic_obligations_retain_named_divisions_before_cancellation():
    class Cancelled(ResearchModel):
        x = InstanceDummy("x", real=True)
        numerator = x + 1
        denominator = x + 1
        ratio = numerator / denominator
        pole = 1 / x
        inverse = 1 / pole
        product = pole * x

    case = ResearchCase.for_model(Cancelled)
    assert any(item.id.startswith("ratio/auto:") for item in case.check().checks)
    for path in ("ratio", "inverse", "product"):
        method = Method(path, lambda current, field=path: current.expression(field), uses=(path,))
        with pytest.raises(ResearchError):
            case.run(method)
        assert case.observe(x=1).run(method).data in (1, S.One)

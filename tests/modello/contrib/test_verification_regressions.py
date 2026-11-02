"""Regression scenarios uncovered by independent review of the verification workflow."""

from dataclasses import replace

import pytest
from sympy import (
    Abs,
    Add,
    Derivative,
    Eq,
    FiniteSet,
    Function,
    Interval,
    Ne,
    Piecewise,
    Pow,
    S,
    Subs,
    Symbol,
    acosh,
    arg,
    asin,
    assuming,
    cos,
    cosh,
    cot,
    exp,
    frac,
    gamma,
    lambdify,
    log,
    pi,
    sin,
    sqrt,
)

from modello import Coordinate, InstanceDummy, Modello, component
from modello.contrib.verification import (
    Method,
    VerificationContext,
    VerificationError,
    VerificationModel,
    continuously_differentiable,
    convex,
    derivative_ratio_bound,
    differentiable,
    equal_to,
    nonnegative,
    nonzero,
    positive,
    specifies,
    strictly_decreasing,
    strictly_increasing,
    supremum_below,
    tends_to_zero,
)
from modello.errors import ModelDefinitionError, ModelValidationError


def verify_obligation(context, claim):
    return (
        context.with_obligations(replace(claim, consumer="test", reason="Review regression."))
        .verify()
        .explain(claim.name)
    )


@pytest.mark.parametrize("function", (log, cot, gamma, arg, lambda x: 1 / x))
def test_zero_coefficient_retains_source_poles(function):
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        source = function(x)
        value = a * source

    context = VerificationContext.for_model(Curves).observe(a=0)
    for claim in (continuously_differentiable("value", wrt="x"), equal_to("value", S.Zero, wrt="x")):
        assert verify_obligation(context, claim).verdict == "strictly false"
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(0,))
    assert context.sample("value", wrt="x", points=(1,)).values == (0,)
    expression = context.expression("value", wrt="x")
    assert expression.subs(Curves.x, 1).doit().evalf() == 0
    assert expression.subs(Curves.x, 0).doit().evalf() is S.NaN


@pytest.mark.parametrize("function", (log, asin, acosh))
@pytest.mark.parametrize("property", ("identity", "continuously_differentiable", "nonnegative", "local_derivative"))
def test_real_continuity_exclusions_do_not_refute_finite_complex_intermediates(function, property):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        coefficient = InstanceDummy("coefficient", real=True)
        value = coefficient * function(x)

    context = VerificationContext.for_model(Curve).observe(coefficient=0)
    domain = Interval(-3, -2)
    if property == "identity":
        claim = equal_to("value", S.Zero, wrt="x", domain=domain)
    elif property == "local_derivative":
        claim = differentiable("value", wrt="x", domain=domain, at=-S(5) / 2)
    else:
        claim = (continuously_differentiable if property == "continuously_differentiable" else nonnegative)(
            "value", wrt="x", domain=domain
        )
    assert verify_obligation(context, claim).verdict == "unresolved"
    points = (-1,) if function is asin else (-3, -2)
    assert context.sample("value", wrt="x", points=points).values == (0,) * len(points)


def test_disjoint_real_continuity_certificates_do_not_refute_finite_sources():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        coefficient = InstanceDummy("coefficient", real=True)
        value = coefficient * log(x) * log(-x)

    context = VerificationContext.for_model(Curve).observe(coefficient=0)
    claim = equal_to("value", S.Zero, wrt="x", domain=Interval(1, 2))
    assert verify_obligation(context, claim).verdict == "unresolved"
    assert context.sample("value", wrt="x", points=(1, 2)).values == (0, 0)


@pytest.mark.parametrize("function", (cos, cosh))
def test_real_continuity_certificates_do_not_refute_smooth_composed_functions(function):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = function(sqrt(x))

    context = VerificationContext.for_model(Curve)
    assert (
        verify_obligation(context, continuously_differentiable("value", wrt="x", domain=Interval(-3, -2))).verdict
        == "unresolved"
    )
    assert (
        verify_obligation(context, continuously_differentiable("value", wrt="x", domain=Interval(1, 2))).verdict
        == "strictly true"
    )
    values = context.sample("value", wrt="x", points=(-3, -2)).values
    assert all(value.is_finite is True and value.is_real is True for value in values)


@pytest.mark.parametrize(
    "condition", (lambda a: a > sqrt(2), lambda a: a > pi, lambda a: Eq(a, pi), lambda a: sin(a) > 0)
)
def test_unsupported_inequality_reasoning_remains_inspectable(condition):
    class Parameter(VerificationModel):
        a = InstanceDummy("a", real=True)

    source = VerificationContext.for_model(Parameter)
    assumption = condition(source.symbols.a)
    context = source.assume(assumption)
    claim = positive("a")
    report = context.verify(obligations=(claim,))
    assert assumption in report.assumptions
    assert report.consistent is not False
    assert report.explain(claim.name).condition == (context.symbols.a > 0)
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("positive_parameter", lambda current: calls.append(current), (claim,)))
    assert not calls


def test_source_guards_preserve_local_substitutions_and_identity_domains():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * log(x)
        local = Subs(value, x, 1)
        undefined = Subs(value, x, 0)
        slope = Derivative(value, x)

    context = VerificationContext.for_model(Curves).observe(a=0)
    domain = Interval.open(0, S.Infinity)
    assert (
        verify_obligation(context, continuously_differentiable("value", wrt="x", domain=domain)).verdict
        == "strictly true"
    )
    assert verify_obligation(context, equal_to("value", S.Zero, wrt="x", domain=domain)).verdict == "strictly true"
    assert verify_obligation(context, continuously_differentiable("local", wrt="x")).verdict == "strictly true"
    assert verify_obligation(context, continuously_differentiable("undefined", wrt="x")).verdict == "strictly false"
    assert verify_obligation(context, continuously_differentiable("slope", wrt="x")).verdict == "strictly false"
    assert verify_obligation(context, differentiable("value", wrt="x", at=1)).verdict == "strictly true"
    assert verify_obligation(context, differentiable("value", wrt="x", at=0)).verdict == "strictly false"
    expected = context.symbols.a * log(Curves.x)
    assert verify_obligation(context, equal_to(S.Zero, expected, wrt="x")).verdict == "strictly false"
    assert verify_obligation(context, equal_to(S.Zero, expected, wrt="x", domain=domain)).verdict == "strictly true"


def test_unsupported_source_definedness_is_not_erased_by_zero_coefficients():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * Function("f")(x)

    context = VerificationContext.for_model(Curves).observe(a=0)
    assert verify_obligation(context, continuously_differentiable("value", wrt="x")).verdict == "unresolved"


@pytest.mark.parametrize("backend", ("math", "numpy"))
def test_source_guards_support_native_numerical_compilation(backend):
    if backend == "numpy":
        pytest.importorskip("numpy")

    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * log(x)

    expression = VerificationContext.for_model(Curves).observe(a=0).expression("value", wrt="x")
    assert lambdify(Curves.x, expression, modules=backend)(1) == 0


def test_compilation_cannot_use_backend_extensions_to_fill_source_holes():
    pytest.importorskip("numpy")

    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * arg(x)

    expression = VerificationContext.for_model(Curves).observe(a=0).expression("value", wrt="x")
    assert expression.subs(Curves.x, 0) is S.NaN
    with pytest.raises(NotImplementedError, match="source domain"):
        lambdify(Curves.x, expression, modules="numpy")


def test_fixed_unevaluated_functions_do_not_supply_false_finite_certificates():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)

    context = VerificationContext.for_model(Curves).observe(a=0)
    source = arg(0, evaluate=False)
    assert context.expression(source, wrt="x") is S.NaN
    assert verify_obligation(context, continuously_differentiable(source, wrt="x")).verdict == "strictly false"
    with pytest.raises(ModelValidationError):
        context.sample(source, wrt="x", points=(1,))


@pytest.mark.parametrize("nested", (False, True))
def test_zero_observation_retains_native_derivative_premises(nested):
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)

    context = VerificationContext.for_model(Curves).observe(a=0)
    for point, expected in ((0, "strictly false"), (1, "strictly true")):
        value = context.symbols.a * Subs(Derivative(Abs(Curves.x), Curves.x), Curves.x, point)
        if nested:
            value = log(1 + value)
        obligation = equal_to(value, S.Zero, wrt="x")
        assert verify_obligation(context, obligation).verdict == expected
        assert verify_obligation(context, continuously_differentiable(value, wrt="x")).verdict == expected
        method = Method("native", lambda current: 0, (obligation,))
        if point == 0:
            with pytest.raises(VerificationError):
                context.run(method)
            with pytest.raises(ModelValidationError):
                context.sample(value, wrt="x", points=(1,))
        else:
            assert context.run(method).data == 0
            assert context.sample(value, wrt="x", points=(1,)).values == (0,)

    zero_inside = Derivative(context.symbols.a * Abs(Curves.x), Curves.x)
    assert verify_obligation(context, equal_to(zero_inside, S.Zero, wrt="x")).verdict == "strictly true"


def test_observed_derivative_coordinates_retain_their_evaluation_scope():
    class State(VerificationModel):
        q = InstanceDummy("q", real=True)

    original = VerificationContext.for_model(State)
    native = Derivative(Abs(original.symbols.q), original.symbols.q)
    assert verify_obligation(original.observe(q=1), equal_to(native, S.One)).verdict == "strictly true"
    assert verify_obligation(original.observe(q=0), equal_to(native, S.Zero)).verdict == "strictly false"


def test_symbolic_observations_preserve_their_source_domains():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a * b

    context = VerificationContext.for_model(Curves).observe(a=log(Curves.x), b=0)
    assert verify_obligation(context, continuously_differentiable("value", wrt="x")).verdict == "strictly false"
    assert (
        verify_obligation(
            context, continuously_differentiable("value", wrt="x", domain=Interval.open(0, S.Infinity))
        ).verdict
        == "strictly true"
    )
    assert context.sample("value", wrt="x", points=(1,)).values == (0,)
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(0,))


def test_supported_source_domain_is_shared_by_scalar_property_checks():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * log(x)
        shifted = 1 + value

    context = VerificationContext.for_model(Curves).observe(a=0)
    domain = Interval.open(0, S.Infinity)
    for claim in (
        nonnegative("value", wrt="x", domain=domain),
        positive("shifted", wrt="x", domain=domain),
        nonzero("shifted", wrt="x", domain=domain),
        convex("value", wrt="x", domain=domain),
        supremum_below("value", 1, wrt="x", domain=domain),
        tends_to_zero("value", wrt="x"),
    ):
        assert verify_obligation(context, claim).verdict == "strictly true"


def test_source_guards_do_not_borrow_coordinate_assumptions_for_a_larger_domain():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * log(x)

    context = VerificationContext.for_model(Curves).observe(a=0)
    with assuming(Curves.x > 0):
        assert verify_obligation(context, continuously_differentiable("value", wrt="x")).verdict == "strictly false"
    assert (
        verify_obligation(context.assume(Curves.x > 0), continuously_differentiable("value", wrt="x")).verdict
        == "strictly false"
    )
    assumed = VerificationContext.for_model(Curves)
    assert (
        verify_obligation(
            assumed.assume(Eq(assumed.symbols.a, 0)), continuously_differentiable("value", wrt="x")
        ).verdict
        != "strictly true"
    )


@pytest.mark.parametrize("point", (-1, 0, S.Infinity, S.ImaginaryUnit, S.true, (1, 2)))
def test_sampling_rejects_invalid_points_and_intrinsic_domain_violations(point):
    class PositiveAxis(VerificationModel):
        x = Coordinate("x", positive=True)
        value = sqrt(x**2)

    context = VerificationContext.for_model(PositiveAxis)
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(point,))
    assert tuple(float(value) for value in context.sample("value", wrt="x", points=(1, 2)).values) == (1, 2)


@pytest.mark.parametrize(
    "source",
    (
        lambda x, point: Subs(Derivative(Abs(x), x, evaluate=False), x, point),
        lambda x, point: log(point, evaluate=False),
        lambda x, point: arg(point, evaluate=False),
    ),
    ids=("native-derivative", "logarithm", "argument"),
)
def test_sampling_points_retain_native_premises_and_scalar_source_domains(source):
    class Curve(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    context = VerificationContext.for_model(Curve)
    invalid = Pow(source(Curve.x, S.Zero), S.Zero, evaluate=False)
    with pytest.raises(ModelValidationError, match="Sampling points"):
        context.sample("value", wrt="x", points=(invalid,))

    valid = Pow(source(Curve.x, S.One), S.Zero, evaluate=False)
    sample = context.sample("value", wrt="x", points=(valid,))
    assert sample.points == (valid,)
    assert sample.values == (1.0,)


def test_sampling_rejects_unbound_parameters_and_symbolic_points():
    class Curves(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * x**3

    context = VerificationContext.for_model(Curves)
    with pytest.raises(ModelValidationError, match="bound parameters"):
        context.sample("value", wrt="x", points=(1,))
    with pytest.raises(ModelValidationError, match="bound finite real"):
        context.sample("value", wrt="x", points=(context.symbols.a,))
    with pytest.raises(ModelValidationError, match="scalar expression"):
        context.sample(S.true, wrt="x", points=(1,))
    assert context.sample("value", wrt="x", points=(0,)).values == (0,)


def test_discontinuity_does_not_refute_definedness_order_or_identity():
    class Discontinuous(VerificationModel):
        x = Coordinate("x", real=True)
        value = frac(x) + 1

    context = VerificationContext.for_model(Discontinuous)
    for claim in (
        positive("value", wrt="x", domain=Interval(0, 1)),
        nonzero("value", wrt="x", domain=Interval(0, 1)),
        equal_to("value", lambda get: get("value"), wrt="x", domain=Interval(0, 1)),
    ):
        assert verify_obligation(context, claim).verdict == "strictly true"


def test_nonpolynomial_certificate_failure_is_an_inapplicable_attempt():
    class Functions(VerificationModel):
        x = Coordinate("x", real=True)
        value = specifies(exp(x), strictly_decreasing("value", wrt="x", consumer="test", reason="A wrong claim."))
        wave = sin(x)

    assert Functions.definition_report().evidence[0].verdict == "strictly false"
    assert verify_obligation(
        VerificationContext.for_model(Functions), strictly_increasing("wave", wrt="x")
    ).verdict in {
        "unresolved",
        "strictly false",
    }


def test_native_declaration_metadata_is_bound_in_its_component_scope():
    class Native(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = specifies(a * x, strictly_increasing(a * x, wrt=x, consumer="test", reason="Native expression."))

    class Parent(VerificationModel):
        curve = component(Native)

    assert Native.definition_report().evidence[0].verdict == "conditionally true"
    context = VerificationContext.for_model(Parent).observe(curve=Native("curve", a=2))
    assert context.verify().evidence[0].verdict == "strictly true"


def test_replacement_rebinds_native_coordinates_and_parameters():
    class Old(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", positive=True)
        value = a * x

    class New(Old):
        x = Coordinate("y", real=True)
        value = Old.a * x

    class Parent(VerificationModel):
        curve = component(Old)

    context = VerificationContext.for_model(Parent).observe(curve=Old("curve", a=2))
    claim = strictly_increasing(context.symbols.curve.value, wrt=Old.x)
    original = context.with_obligations(replace(claim, consumer="test", reason="Native curve coordinate."))
    changed = original.replace("curve", New)
    assert original.verify().evidence[0].verdict == changed.verify().evidence[0].verdict == "strictly true"
    assert changed.verify().evidence[0].coordinate == New.x


def test_all_parameter_payloads_are_specialized_and_audited():
    class Parameterized(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = x**2
        kink = Abs(x)

    context = VerificationContext.for_model(Parameterized)
    claim = supremum_below("value", context.symbols.a, wrt="x", domain=Interval(0, 1))
    evidence = verify_obligation(context.observe(a=5).with_bound_hint(claim.name, upper=1), claim)
    assert evidence.verdict == "strictly true"
    assert evidence.resolved.bound == 5
    claim = supremum_below("value", context.symbols.a + 1, wrt="x", domain=Interval(0, context.symbols.a))
    assert (
        verify_obligation(context.observe(a=1).with_bound_hint(claim.name, upper=1), claim).verdict == "strictly true"
    )
    local = differentiable("kink", wrt="x", at=context.symbols.a)
    assert verify_obligation(context.observe(a=1), local).verdict == "strictly true"


def test_tail_limits_do_not_require_interior_regularity():
    class Pole(VerificationModel):
        x = Coordinate("x", real=True)
        value = 1 / x

    assert (
        verify_obligation(VerificationContext.for_model(Pole), tends_to_zero("value", wrt="x")).verdict
        == "strictly true"
    )


@pytest.mark.parametrize("guarded", (False, True))
@pytest.mark.parametrize("domain", (Interval(1, 2), FiniteSet(1), Interval(S.NegativeInfinity, -1)))
def test_right_tail_claims_require_an_unbounded_above_domain(guarded, domain):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = 1 / x if guarded else exp(-x)

    claim = replace(tends_to_zero("value", wrt="x"), domain=domain)
    context = VerificationContext.for_model(Curve)
    evidence = verify_obligation(context, claim)
    assert evidence.verdict == "unresolved"
    assert not evidence.witness
    assert "unbounded right-hand" in " ".join(evidence.steps)
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("tail", lambda current: calls.append(current), (claim,)))
    assert not calls
    constrained = specifies(Curve, claim)
    assert constrained.definition_report().explain(claim.name).verdict == "unresolved"


@pytest.mark.parametrize("guarded", (False, True))
@pytest.mark.parametrize("domain", (S.Reals, Interval.open(0, S.Infinity), Interval(1, S.Infinity)))
def test_right_tail_claims_keep_supported_unbounded_domains(guarded, domain):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = 1 / x if guarded else exp(-x)

    claim = replace(tends_to_zero("value", wrt="x"), domain=domain)
    context = VerificationContext.for_model(Curve)
    assert verify_obligation(context, claim).verdict == "strictly true"
    assert context.run(Method("tail", lambda current: "established", (claim,))).data == "established"


def test_substitution_guards_are_local_for_direct_and_named_curves():
    class Evaluated(VerificationModel):
        x = Coordinate("x", real=True)
        pole = 1 / x
        direct = Subs(1 / x, x, 1)
        named = Subs(pole, x, 1)
        slope = Subs(Derivative(pole, x), x, 1)

    context = VerificationContext.for_model(Evaluated)
    for path, expected in (("direct", 1), ("named", 1), ("slope", -1)):
        method = Method(path, lambda current, field=path: current.expression(field).doit(), uses=(path,))
        assert context.run(method).data == expected
    assert verify_obligation(context, nonzero("slope")).verdict == "strictly true"
    assert verify_obligation(context, equal_to("slope", -1)).verdict == "strictly true"


def test_local_differentiability_retains_the_original_neighborhood_domain():
    class Isolated(VerificationModel):
        x = Coordinate("x", real=True)
        value = Piecewise((x**2, Eq(x, 0)), (S.NaN, True))

    assert (
        verify_obligation(VerificationContext.for_model(Isolated), differentiable("value", wrt="x", at=0)).verdict
        != "strictly true"
    )


def test_definedness_retains_parameter_dependent_coordinate_guards():
    class Partial(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = Piecewise((x**2 + 1, x > a), (S.NaN, True))

    context = VerificationContext.for_model(Partial)
    for claim in (positive("value", wrt="x"), equal_to("value", lambda get: get("value"), wrt="x")):
        assert verify_obligation(context, claim).verdict != "strictly true"


def test_audits_and_fingerprints_include_the_compared_proposition():
    class Quadratic(VerificationModel):
        x = Coordinate("x", real=True)
        value = x**2

    context = VerificationContext.for_model(Quadratic)
    first = context.with_obligations(
        equal_to("value", S(2), wrt="x", consumer="test", reason="Compare a form.")
    ).verify()
    second = context.with_obligations(
        equal_to("value", S(3), wrt="x", consumer="test", reason="Compare a form.")
    ).verify()
    assert first.to_dict()["evidence"][0]["expected"] == "2"
    assert second.to_dict()["evidence"][0]["expected"] == "3"
    assert first.to_dict()["evidence_sha256"] != second.to_dict()["evidence_sha256"]
    assert "Expected form:" in first.to_html()


def test_exports_preserve_independent_same_named_coordinates_and_shared_aliases():
    class Left(VerificationModel):
        x = Coordinate("x", real=True)
        value = x**2

    class Right(VerificationModel):
        x = Coordinate("x", real=True)
        value = x**2

    class Pair(VerificationModel):
        a = component(Left)
        alias = component(Left)
        b = component(Right)

    context = VerificationContext.for_model(Pair).assume(Left.x > Right.x)
    expression = Left.x**2 - Right.x**2
    shared = equal_to("a.value", context.symbols.alias.value, wrt=Left.x, name="shared")
    first = context.verify(obligations=(equal_to(expression, 0, wrt=Left.x), shared))
    second = context.verify(obligations=(equal_to(2 * expression, 0, wrt=Left.x), shared))
    assert first.explain("equal_to").expression.free_symbols == {Left.x, Right.x}
    assert first.explain("equal_to").verdict == "unresolved"
    assert first.explain("shared").verdict == "strictly true"
    assert dict(first.labels)[Left.x] == "a.x"
    assert dict(first.labels)[Right.x] == "b.x"
    audit = first.to_dict()
    for field in ("declared_subject", "subject", "expression"):
        assert audit["evidence"][0][field] in {"a.x**2 - b.x**2", "-b.x**2 + a.x**2"}
    assert audit["evidence"][0]["coordinate"] == "a.x"
    assert audit["evidence"][1]["expression"] == audit["evidence"][1]["expected"] == "a.x**2"
    assert audit["assumptions"] == ["a.x > b.x"]
    assert audit["evidence_sha256"] != second.to_dict()["evidence_sha256"]
    html = first.to_html()
    assert "<mi>a.x</mi>" in html and "<mi>b.x</mi>" in html
    assert audit["evidence"][0]["expression"] in html


def test_coordinate_labels_cannot_alias_a_field_path():
    class Curve(VerificationModel):
        argument = Coordinate("t", real=True)
        t = InstanceDummy("t", real=True)
        value = argument - t

    context = VerificationContext.for_model(Curve)
    report = context.verify(obligations=(equal_to("value", 0, wrt="argument"),))
    assert dict(report.labels)[Curve.argument] == "argument"
    assert report.to_dict()["evidence"][0]["expression"] in {"argument - t", "-t + argument"}


def test_export_printing_does_not_evaluate_the_declared_expression():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)

    declared = Add(Curve.x, -Curve.x, evaluate=False)
    report = VerificationContext.for_model(Curve).verify(obligations=(equal_to(declared, 0, wrt=Curve.x),))
    audit = report.to_dict()["evidence"][0]
    assert audit["declared_subject"] in {"-x + x", "x - x"}
    assert audit["expression"] == audit["declared_subject"]
    assert report.explain("equal_to").verdict == "strictly true"


def test_differentiability_remains_open_when_only_continuity_of_derivatives_fails():
    class Explicit(VerificationModel):
        x = Coordinate("x", real=True)
        value = Piecewise((x**4 * sin(1 / x), Ne(x, 0)), (0, True))

    assert (
        verify_obligation(VerificationContext.for_model(Explicit), differentiable("value", wrt="x", order=2)).verdict
        != "strictly false"
    )


def test_curvature_ratio_does_not_require_continuity_of_the_second_derivative():
    class Oscillating(VerificationModel):
        x = Coordinate("x", real=True)
        value = x + Piecewise((x**4 * sin(1 / x), Ne(x, 0)), (0, True))

    claim = derivative_ratio_bound("value", wrt="x", domain=Interval(-S.One / 10, S.One / 10))
    assert verify_obligation(VerificationContext.for_model(Oscillating), claim).verdict != "strictly false"


def test_automatic_obligations_retain_named_divisions_before_cancellation():
    class Cancelled(VerificationModel):
        x = InstanceDummy("x", real=True)
        numerator = x + 1
        denominator = x + 1
        ratio = numerator / denominator
        pole = 1 / x
        inverse = 1 / pole
        product = pole * x

    context = VerificationContext.for_model(Cancelled)
    assert any(item.id.startswith("ratio/auto:") for item in context.verify().evidence)
    for path in ("ratio", "inverse", "product"):
        method = Method(path, lambda current, field=path: current.expression(field), uses=(path,))
        with pytest.raises(VerificationError):
            context.run(method)
        assert context.observe(x=1).run(method).data in (1, S.One)


@pytest.mark.parametrize("field", ("pole", "slope"))
@pytest.mark.parametrize("nested, alias", ((False, False), (False, True), (True, False), (True, True)))
def test_aliases_consume_the_same_automatic_obligations(field, nested, alias):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        pole = 1 / x
        slope = Derivative(Abs(x), x, evaluate=False)
        pole_alias = pole
        slope_alias = slope

    class Parent(VerificationModel):
        curve = component(Curve)
        pole_alias = curve.pole
        slope_alias = curve.slope

    context = VerificationContext.for_model(Parent if nested else Curve)
    path = field + "_alias" if alias else ("curve." if nested else "") + field
    calls = []
    with pytest.raises(VerificationError) as failure:
        context.run(Method("evaluate", lambda current: calls.append(current), uses=(path,)))
    assert not calls
    origin = ("curve." if nested else "") + field + "/auto:1"
    assert failure.value.report.explain(origin).verdict == "strictly false"


@pytest.mark.parametrize("point", (0, 1))
def test_alias_consumption_preserves_local_substitution_scopes(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        pole = 1 / x
        value = Subs(pole, x, point)
        slope = Subs(Derivative(pole, x), x, point)

    class Parent(VerificationModel):
        curve = component(Curve)
        value = curve.value
        slope = curve.slope

    context = VerificationContext.for_model(Parent)
    if point == 0:
        assert context.verify().consistent is False
        for path in ("value", "slope"):
            with pytest.raises(VerificationError):
                context.run(Method("evaluate", lambda current: "must not execute", uses=(path,)))
        return
    for path, expected in (("value", 1), ("slope", -1)):
        result = context.run(Method("evaluate", lambda current: current.expression(path).doit(), uses=(path,)))
        assert result.data == expected
        assert result.report.explain("curve.pole/auto:1").verdict == "strictly false"


@pytest.mark.parametrize("operation", ("verify_obligation", "expression", "sample"))
@pytest.mark.parametrize("coordinate", ("foreign", "derived", "native_derived", "nonsymbol"))
def test_verification_operations_reject_unknown_or_dependent_coordinates(operation, coordinate):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        doubled = 2 * x
        kink = Abs(x)

    context = VerificationContext.for_model(Curve)
    variable = {
        "foreign": Symbol("x", real=True),
        "derived": "doubled",
        "native_derived": context.symbols.doubled,
        "nonsymbol": S.One,
    }[coordinate]
    error = ModelValidationError if coordinate == "foreign" else ModelDefinitionError
    with pytest.raises(error):
        if operation == "verify_obligation":
            context.verify(obligations=(differentiable("kink", wrt=variable),))
        elif operation == "expression":
            context.expression("kink", wrt=variable)
        else:
            context.sample("kink", wrt=variable, points=(0, 1))
    assert verify_obligation(context, differentiable("kink", wrt="x")).verdict == "strictly false"


@pytest.mark.parametrize("factory", (supremum_below, derivative_ratio_bound))
@pytest.mark.parametrize("alias", (False, True))
@pytest.mark.parametrize("ambient", (False, True))
def test_uniform_targets_must_be_independent_after_binding(factory, alias, ambient):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = exp(-x)
        target = x + 2

    context = VerificationContext.for_model(Curve)
    target = context.symbols.target if alias else Curve.x + 2
    obligation = (
        supremum_below(S.Zero, target, wrt=Curve.x)
        if factory is supremum_below
        else derivative_ratio_bound("value", wrt="x", bound=target)
    )
    if not ambient:
        context = context.assume(Curve.x > 0)
    with assuming(Curve.x > 0 if ambient else S.true):
        with pytest.raises(ModelDefinitionError, match="uniform target bound"):
            context.run(Method("bounded calculation", lambda current: "must not execute", (obligation,)))


def test_uniform_targets_can_depend_on_independent_parameters():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        bound = InstanceDummy("b", real=True)

    context = VerificationContext.for_model(Curve)
    bounded = context.assume(context.symbols.bound > 0)
    assert verify_obligation(bounded, supremum_below(S.Zero, bounded.symbols.bound, wrt="x")).verdict == "strictly true"


@pytest.mark.parametrize("order", (1, 2))
def test_parameter_dependent_continuity_failures_do_not_refute_every_specialization(order):
    class Curve(VerificationModel):
        x = Coordinate("x", nonnegative=True)
        a = InstanceDummy("a", real=True)
        value = a * sqrt(x)

    context = VerificationContext.for_model(Curve)
    obligation = continuously_differentiable("value", wrt="x", domain=Interval(0, 1), order=order)
    assert verify_obligation(context, obligation).verdict == "unresolved"
    assert verify_obligation(context.observe(a=0), obligation).verdict == "strictly true"
    assert verify_obligation(context.observe(a=1), obligation).verdict == "strictly false"
    assert verify_obligation(context.assume(context.symbols.a > 0), obligation).verdict == "strictly false"


def test_a_removable_derivative_singularity_does_not_refute_smoothness():
    class Curve(VerificationModel):
        x = Coordinate("x", nonnegative=True)
        value = cos(sqrt(x))

    # The derivative has limit -1/2 at zero despite its unevaluated 0/0 form.
    evidence = verify_obligation(
        VerificationContext.for_model(Curve), continuously_differentiable("value", wrt="x", domain=Interval(0, 1))
    )
    assert evidence.verdict == "unresolved"
    assert not evidence.witness

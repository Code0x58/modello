"""Joint equations and native scopes retain their mathematical obligations."""

import pytest
from sympy import (
    Abs,
    Add,
    Derivative,
    Eq,
    FiniteSet,
    Function,
    Integral,
    Interval,
    Limit,
    Mul,
    Piecewise,
    Product,
    S,
    Subs,
    Sum,
    Symbol,
    exp,
    log,
    sin,
)

from modello import Coordinate, InstanceDummy, Modello
from modello.contrib.verification import (
    InferenceRule,
    Method,
    Obligation,
    VerificationContext,
    VerificationError,
    VerificationModel,
    derivative_ratio_bound,
    differentiable,
    equal_to,
    nonnegative,
    positive,
    supremum_below,
)
from modello.errors import ModelValidationError


def test_pure_defining_equations_bind_parameters_and_verify_feasibility():
    class Implicit(VerificationModel):
        a = InstanceDummy("a", real=True)
        b = 2 * a
        a = b + 1

    context = VerificationContext.for_model(Implicit)
    assert context.expression("a") == -1
    claim = positive("a")
    evidence = context.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "strictly false"
    assert evidence.source_operands
    impossible = context.assume(context.symbols.a > 0)
    assert impossible.verify(obligations=(claim,)).consistent is False
    calls = []
    with pytest.raises(VerificationError):
        impossible.run(Method("calculate", lambda current: calls.append(current), (claim,), uses=("a",)))
    assert not calls


def test_a_contradictory_defining_cycle_blocks_even_an_unguarded_method():
    class Impossible(VerificationModel):
        a = InstanceDummy("a", real=True)
        b = a + 1
        a = b + 1

    context = VerificationContext.for_model(Impossible)
    assert context.verify().consistent is False
    with pytest.raises(VerificationError):
        context.run(Method("calculate", lambda current: "must not execute"))


@pytest.mark.parametrize(
    "assumptions, value",
    (
        ({"positive": True}, -S.One),
        ({"real": True}, S.ImaginaryUnit),
        ({}, S.Infinity),
        ({}, S.ComplexInfinity),
        ({}, S.NaN),
    ),
)
def test_intrinsic_domains_apply_to_definitions_without_observations(assumptions, value):
    class Impossible(VerificationModel):
        a = InstanceDummy("a", **assumptions)
        a = value

    context = VerificationContext.for_model(Impossible)
    assert context.verify().consistent is False
    with pytest.raises(VerificationError):
        context.run(Method("read", lambda current: current.expression("a"), uses=("a",)))


@pytest.mark.parametrize(
    ("operator", "expected"),
    (
        (lambda x: Subs(log(x), x, -1), S.ImaginaryUnit * S.Pi),
        (lambda x: Subs(log(x), x, 0), None),
        (lambda x: Subs(log(x), x, 1), S.Zero),
        (lambda x: Subs(1 / x, x, -1), -S.One),
        (lambda x: Subs(1 / x, x, 0), None),
        (lambda x: Subs(1 / x, x, 1), S.One),
        (lambda x: Sum(log(x), (x, 0, 0)), None),
        (lambda x: Product(log(x), (x, 0, 0)), None),
        (lambda x: Integral(log(S.Zero), (x, 0, 1)), None),
        (lambda x: Integral(x, (x, 0, 1)), S.Half),
        (lambda x: Integral(log(x), (x, 0, 0)), S.Zero),
        (lambda x: Limit(sin(x) / x, x, 0, dir="+"), S.One),
        (lambda x: Limit(1 / x, x, 0, dir="+"), None),
    ),
    ids=(
        "complex-log",
        "undefined-log",
        "zero-log",
        "negative-reciprocal",
        "undefined-reciprocal",
        "positive-reciprocal",
        "undefined-sum",
        "undefined-product",
        "nonfinite-integral",
        "finite-integral",
        "empty-integral",
        "finite-limit",
        "nonfinite-limit",
    ),
)
def test_closed_native_values_obey_model_domains_before_method_execution(operator, expected):
    class Scoped(VerificationModel):
        x = Coordinate("x", real=True)
        value = operator(x)

    context = VerificationContext.for_model(Scoped)
    assert Scoped.definition_report().consistent is (expected is not None)
    assert context.verify().consistent is (expected is not None)
    assert context.family.is_empty is (expected is None)
    calls = []
    method = Method(
        "read",
        lambda current: calls.append(current) or current.family.expression(current.symbols.value),
        uses=("value",),
    )
    if expected is None:
        with pytest.raises(VerificationError):
            context.run(method)
        assert not calls
    else:
        assert context.run(method).data == expected
        assert calls == [context]


@pytest.mark.parametrize("base", (Modello, VerificationModel))
def test_unsupported_cyclic_calculus_remains_inspectable_and_blocks_execution(base):
    class Cyclic(base):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = a * x
        a = b + x
        slope = Derivative(b, x, evaluate=False)

    context = VerificationContext.for_model(Cyclic)
    report = context.verify()
    assert report.consistent is None
    assert report.evidence and all(item.verdict == "unresolved" for item in report.evidence)
    assert "unresolved" in report.to_html()
    assert report.to_dict()["evidence"]
    with pytest.raises(VerificationError):
        context.run(Method("read", lambda current: "must not execute", uses=("slope",)))


@pytest.mark.parametrize(
    "operator, expected",
    (
        (lambda x: Integral(1 / x, (x, 1, 2)), log(2)),
        (lambda x: Integral(Derivative(log(x), x), (x, 1, 2)), log(2)),
        (lambda x: Sum(1 / x, (x, 1, 2)), S(3) / 2),
        (lambda x: Product(1 / x, (x, 1, 2)), S.Half),
        (lambda x: Sum(1 / x, (x, 3, 0)), -S(3) / 2),
        (lambda x: Sum(1 / x, (x, 0, -1)), S.Zero),
        (lambda x: Integral(1 / x, (x, 0, 0)), S.Zero),
    ),
)
def test_automatic_operations_use_native_binder_domains(operator, expected):
    class Scoped(VerificationModel):
        x = InstanceDummy("x", real=True)
        value = operator(x)

    context = VerificationContext.for_model(Scoped)
    method = Method("calculate", lambda current: current.expression("value").doit(), uses=("value",))
    assert context.run(method).data == expected
    assert context.assume(context.symbols.x > 0).run(method).data == expected


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
@pytest.mark.parametrize("source", ("definition", "native", "observation"))
@pytest.mark.parametrize("point", (0, 1))
def test_empty_reductions_verify_native_endpoints_before_certifying_identities(operator, identity, source, point):
    class Scoped(Modello):
        x = Coordinate("x", real=True)
        n = Coordinate("n", integer=True)
        endpoint = Subs(Derivative(Abs(x), x), x, point)
        reduction = operator(
            1, (x if operator is Integral else n, endpoint, endpoint if operator is Integral else endpoint - 1)
        )
        value = reduction if source == "definition" else InstanceDummy("value", real=True)

    context = VerificationContext.for_model(Scoped)
    subject = context.expression("reduction") if source == "native" else "value"
    if source == "observation":
        context = context.observe(value=context.expression("reduction"))
    claim = equal_to(subject, identity)
    evidence = context.verify(obligations=(claim,)).explain(claim.name)
    calls = []
    method = Method("identity", lambda current: calls.append(current) or identity, (claim,))
    if point == 0:
        assert evidence.verdict == "strictly false"
        with pytest.raises(VerificationError):
            context.run(method)
        assert not calls
        with pytest.raises(ModelValidationError):
            context.sample(subject, wrt="x", points=(1,))
    else:
        assert evidence.verdict == "strictly true"
        assert context.run(method).data == identity
        assert calls == [context]
        assert tuple(map(float, context.sample(subject, wrt="x", points=(1,)).values)) == (float(identity),)


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
def test_valid_empty_reductions_do_not_verify_an_inactive_undefined_derivative(operator, identity):
    class Scoped(Modello):
        x = Coordinate("x", real=True)
        n = Coordinate("n", integer=True)
        invalid = Subs(Derivative(Abs(x), x), x, 0)
        value = operator(invalid, (x if operator is Integral else n, 1, 1 if operator is Integral else 0))

    context = VerificationContext.for_model(Scoped)
    claims = (equal_to("value", identity), nonnegative("value"))
    result = context.run(Method("empty", lambda current: current.expression("value").doit(), claims))
    assert result.data == identity
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
def test_valid_empty_reductions_do_not_require_a_definition_for_their_body(operator, identity):
    class Scoped(Modello):
        x = Coordinate("x", real=True) if operator is Integral else Coordinate("x", integer=True)
        value = operator(Function("f")(x), (x, 1, 1 if operator is Integral else 0))

    context = VerificationContext.for_model(Scoped)
    claims = (equal_to("value", identity), nonnegative("value"))
    result = context.run(Method("empty", lambda current: current.expression("value").doit(), claims))
    assert result.data == identity
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
def test_valid_empty_reductions_do_not_inspect_inactive_cyclic_algebra(operator, identity):
    class Scoped(Modello):
        x = Coordinate("x", real=True)
        y = InstanceDummy("y", real=True)
        y = y / 2
        value = operator(y, (x, 1, 1 if operator is Integral else 0))

    context = VerificationContext.for_model(Scoped)
    claims = (equal_to("value", identity), nonnegative("value"))
    result = context.run(Method("empty", lambda current: current.expression("value").doit(), claims))
    assert result.data == identity
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)


def test_native_local_symbols_are_owned_by_their_model_expression():
    index = Symbol("local_index", integer=True)

    class Scoped(VerificationModel):
        value = Sum(1 / index, (index, 1, 2))

    context = VerificationContext.for_model(Scoped)
    result = context.run(Method("calculate", lambda current: current.expression("value").doit(), uses=("value",)))
    assert result.data == S(3) / 2


@pytest.mark.parametrize("coordinate", (Coordinate, InstanceDummy))
@pytest.mark.parametrize("operator", (Sum, Product))
def test_free_variable_assumptions_cannot_certify_a_locally_singular_reduction(coordinate, operator):
    class Scoped(VerificationModel):
        x = coordinate("x", real=True)
        value = operator(1 / x, (x, 0, 1))

    context = VerificationContext.for_model(Scoped)
    local = context.expression("x", wrt="x")
    context = context.assume(local > 0)
    obligations = tuple(item for item in context.verify().evidence if item.obligation.consumer == "value")
    assert obligations and all(item.verdict != "strictly true" for item in obligations)
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("calculate", lambda current: calls.append(current), uses=("value",)))
    assert not calls
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(1,))


def test_a_local_substitution_keeps_its_own_scope_inside_a_reduction():
    class Scoped(VerificationModel):
        x = Coordinate("x", real=True)
        value = Sum(Subs(Derivative(log(x), x), x, 1), (x, 0, 1))

    context = VerificationContext.for_model(Scoped)
    result = context.run(
        Method("calculate", lambda current: current.sample("value", wrt="x", points=(1,)), uses=("value",))
    )
    assert tuple(map(float, result.data.values)) == (2.0,)


def test_nested_reduction_endpoints_keep_their_outer_variable_scope():
    class Nested(VerificationModel):
        x = InstanceDummy("x", integer=True)
        y = InstanceDummy("y", integer=True)
        value = Sum(1, (y, 0, 1 / x), (x, 0, 1))

    context = VerificationContext.for_model(Nested)
    context = context.assume(context.symbols.x > 0)
    assert context.verify().explain("value/auto:1").verdict == "unresolved"
    with pytest.raises(VerificationError):
        context.run(Method("calculate", lambda current: "must not execute", uses=("value",)))


def test_coupled_derivative_scopes_cannot_borrow_free_observations_or_sample_points():
    class Coupled(VerificationModel):
        x = InstanceDummy("x", real=True)
        y = Coordinate("y", real=True)
        value = Sum(Subs(Derivative(x * Abs(y), y), y, 0), (x, 0, 1))

    context = VerificationContext.for_model(Coupled)
    observed = context.observe(x=0)
    assert observed.verify().explain("value/auto:1").verdict == "unresolved"
    with pytest.raises(VerificationError):
        observed.run(Method("calculate", lambda current: "must not execute", uses=("value",)))
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(0,))


@pytest.mark.parametrize("domain", (Interval(1, 2), FiniteSet(1, 2)))
def test_ordinary_derivative_certificates_need_neighborhoods_at_all_requested_points(domain):
    class OneSided(VerificationModel):
        x = Coordinate("x", real=True)
        curve = Piecewise((x, x >= 1), (S.NaN, True))
        value = Sum(Derivative(curve, x), (x, 1, 2))

    context = VerificationContext.for_model(OneSided)
    claim = differentiable("curve", wrt="x", domain=domain)
    report = context.verify(obligations=(claim,))
    assert report.explain(claim.name).verdict == "unresolved"
    assert report.explain("value/auto:1").verdict == "unresolved"
    interior = differentiable("curve", wrt="x", domain=Interval.open(1, 2), name="interior")
    assert context.verify(obligations=(interior,)).explain(interior.name).verdict == "strictly true"
    with pytest.raises(VerificationError):
        context.run(Method("calculate", lambda current: "must not execute", uses=("value",)))


@pytest.mark.parametrize("domain, verdict", ((Interval.open(1, 2), "strictly true"), (Interval(1, 2), "unresolved")))
def test_guarded_curvature_uses_the_defined_body_after_verifying_ordinary_neighborhoods(domain, verdict):
    class OneSided(VerificationModel):
        x = Coordinate("x", real=True)
        curve = Piecewise((exp(x), x >= 1), (S.NaN, True))

    context = VerificationContext.for_model(OneSided)
    claim = derivative_ratio_bound("curve", wrt="x", domain=domain)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == verdict


def test_limit_premises_use_a_punctured_scope_without_refuting_other_neighborhoods():
    class Scoped(VerificationModel):
        x = Coordinate("x", real=True)
        near = Limit(sin(x) / x, x, 0, dir="+")
        far = Limit(sin(x) / (x - 1), x, 0, dir="+")

    report = VerificationContext.for_model(Scoped).verify()
    assert report.explain("near/auto:1").verdict == "strictly true"
    assert report.explain("far/auto:1").verdict == "unresolved"


@pytest.mark.parametrize("nested", (False, True))
def test_consumed_canonical_and_rule_premise_hints_are_not_reported_unused(nested):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        rate = 1 / (1 + exp(x))

    context = VerificationContext.for_model(Curve)
    bound = derivative_ratio_bound("rate", wrt="x")
    if nested:
        conclusion = Obligation("target", "rate", "rule")
        context = context.with_rule(InferenceRule("test rule", conclusion, (bound,), "Explicit test rule"))
        context = context.with_obligations(conclusion)
    else:
        context = context.with_obligations(bound)
    report = context.with_bound_hint("rate/derivative_ratio_bound", upper=1).verify()
    assert report.explain("target" if nested else bound.name).verdict == "strictly true"
    assert not any("Unused proof hints" in issue for issue in report.issues)


@pytest.mark.parametrize("kind", ("uniform", "curvature"))
def test_a_loose_conditional_hint_cannot_refute_a_true_bound(kind):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        bound = InstanceDummy("B", real=True)
        value = sin(x) if kind == "uniform" else 1 / (1 + exp(x))

    context = VerificationContext.for_model(Curve)
    obligation = supremum_below if kind == "uniform" else derivative_ratio_bound
    claim = obligation("value", bound=context.symbols.bound, wrt="x")
    hinted = context.with_bound_hint(claim.name, upper=3)
    assert hinted.verify(obligations=(claim,)).explain(claim.name).verdict == "conditionally true"
    valid = hinted.assume(context.symbols.bound > 1, context.symbols.bound < 3)
    assert valid.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    with pytest.raises(VerificationError):
        valid.run(Method("certificate", lambda current: "must not execute", (claim,)))
    tight = valid.with_bound_hint(claim.name, upper=1)
    assert tight.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("order", (1, 2))
def test_a_bound_hint_cannot_hide_a_refuted_curvature_prerequisite(order):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        value = (
            Piecewise((x + x**3, x <= 0), (2 * (x + x**3), True))
            if order == 1
            else Piecewise((x, x < 0), (x + x**2, True))
        )

    context = VerificationContext.for_model(Curve)
    claim = derivative_ratio_bound("value", wrt="x", bound=10)
    p = context.symbols.p
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, p)
    guarded_hint = Add(5, Mul(0, native, evaluate=False), evaluate=False)
    # The ratio is bounded on each side, but the subject's first or second
    # derivatives do not match at zero.
    for upper, point in (
        (None, None),
        (5, None),
        (-1, None),
        (S.ImaginaryUnit, None),
        (guarded_hint, 0),
        (guarded_hint, 1),
    ):
        current = context if point is None else context.assume(Eq(p, point))
        if upper is not None:
            current = current.with_bound_hint(claim.name, upper=upper)
        evidence = current.verify(obligations=(claim,)).explain(claim.name)
        assert evidence.verdict == "strictly false"
        assert "do not match" in " ".join(evidence.steps)
    assert evidence.witness == ((Curve.x, S.Zero),)
    calls = []
    with pytest.raises(VerificationError) as failure:
        context.with_bound_hint(claim.name, upper=5).run(
            Method("curvature", lambda current: calls.append(current), (claim,))
        )
    assert failure.value.report.explain(claim.name).verdict == "strictly false"
    assert not calls


@pytest.mark.parametrize("kind", ("uniform", "curvature"))
@pytest.mark.parametrize("scope", ("native", "compound", "observation", "binding"))
def test_failed_optional_hint_premises_cannot_hide_an_independent_refutation(kind, scope):
    class Curve(Modello):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        delta = a - b
        value = S.One
        if scope == "binding":
            y = InstanceDummy("y", real=True)
            y = y / 2

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x, evaluate=False), Curve.x, 0)
    if scope == "observation":
        context = context.observe(a=context.symbols.b, b=context.symbols.a + native)
        hint = 1 + context.symbols.delta
    elif scope == "compound":
        hint = Subs(Derivative(Abs(Curve.x), Curve.x, evaluate=False), Curve.x**2, 0)
    elif scope == "binding":
        hint = Derivative(context.symbols.y, Curve.x, evaluate=False)
    else:
        hint = native
    claim = supremum_below("value", 0, wrt="x") if kind == "uniform" else derivative_ratio_bound("value", wrt="x")
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly false"
    hinted = context.with_bound_hint(claim.name, upper=hint)
    evidence = hinted.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "strictly false"
    assert evidence.hint is not None
    if scope == "observation":
        assert evidence.source_operands
    calls = []
    with pytest.raises(VerificationError) as failure:
        hinted.run(Method("refuted", lambda current: calls.append(current), (claim,), uses=("value",)))
    assert failure.value.report.explain(claim.name).verdict == "strictly false"
    assert not calls
    if scope == "binding":
        unrefuted = supremum_below(sin(Curve.x), 2, wrt="x", name="unrefuted")
        uncertain = context.with_bound_hint(unrefuted.name, upper=hint)
        evidence = uncertain.verify(obligations=(unrefuted,)).explain(unrefuted.name)
        assert evidence.verdict == "unresolved"
        assert evidence.hint is not None
        assert "cyclic" in " ".join(evidence.steps)
        repaired = uncertain.with_bound_hint(unrefuted.name, upper=1)
        assert repaired.run(Method("repaired", lambda current: 1, (unrefuted,))).data == 1
        foreign = context.with_bound_hint(claim.name, upper=Symbol("foreign", real=True))
        with pytest.raises(ModelValidationError, match="context's symbols"):
            foreign.verify(obligations=(claim,))


@pytest.mark.parametrize("kind", ("uniform_native", "curvature_native", "uniform_margin"))
def test_conditional_hint_conditions_cannot_refute_an_admissible_subject(kind):
    class Curve(Modello):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        if kind == "uniform_margin":
            value = Piecewise((-1 - x**2, p**2 <= 0), (S.NaN, True), evaluate=False)
        elif kind == "curvature_native":
            value = exp(x / p)
        else:
            value = Mul(0, 1 / p, evaluate=False)

    context = VerificationContext.for_model(Curve)
    p = context.symbols.p
    if kind == "uniform_margin":
        claim = supremum_below("value", p**2, wrt="x")
        hint, valid_hint, point = S.Zero, -S.One, 0
    else:
        claim = (
            derivative_ratio_bound("value", wrt="x")
            if kind == "curvature_native"
            else supremum_below("value", 1, wrt="x")
        )
        hint = Derivative(Piecewise((Curve.x, Curve.x < 0), ((p + 1) * Curve.x, True)), Curve.x, evaluate=False)
        valid_hint, point = S.One if kind == "curvature_native" else S.Zero, 1
    baseline = context.with_bound_hint(claim.name, upper=valid_hint)
    assert baseline.verify(obligations=(claim,)).explain(claim.name).verdict == "conditionally true"
    hinted = context.with_bound_hint(claim.name, upper=hint)
    evidence = hinted.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "unresolved"
    assert evidence.hint is not None
    calls = []
    with pytest.raises(VerificationError):
        hinted.run(Method("conditional", lambda current: calls.append(current), (claim,), uses=("value",)))
    assert not calls
    observed = hinted.observe(p=point)
    assert observed.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    repaired = observed.with_bound_hint(claim.name, upper=valid_hint)
    assert repaired.run(Method("repaired", lambda current: 1, (claim,), uses=("value",))).data == 1


def test_bound_source_refutations_precede_optional_certificates():
    class Curve(Modello):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        value = Add(-exp(-(x**2)), Mul(0, 1 / (p - 1), evaluate=False), evaluate=False)

    base = VerificationContext.for_model(Curve)
    context = base.assume(Eq(base.symbols.p, 1))
    claim = supremum_below("value", 1, wrt="x")
    for hint in (None, 0, 2, S.ImaginaryUnit):
        current = context if hint is None else context.with_bound_hint(claim.name, upper=hint)
        assert current.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly false"
    calls = []
    with pytest.raises(VerificationError) as failure:
        current.run(Method("undefined", lambda current: calls.append(current), (claim,), uses=("value",)))
    assert failure.value.report.explain(claim.name).verdict == "strictly false"
    assert not calls
    valid = base.observe(p=2).with_bound_hint(claim.name, upper=0)
    assert valid.run(Method("defined", lambda current: 1, (claim,), uses=("value",))).data == 1


def test_finite_piecewise_values_do_not_need_conditional_continuity():
    class Curve(Modello):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        q = InstanceDummy("q", real=True)
        value = Piecewise((0, x < 0), (p + q, True))

    base = VerificationContext.for_model(Curve)
    context = base.assume(Eq(base.symbols.p + base.symbols.q, 1))
    identity = equal_to("value", lambda reference: reference("value"), wrt="x")
    bound = supremum_below("value", 2, wrt="x")
    report = context.with_bound_hint(bound.name, upper=1).verify(obligations=(identity, bound))
    assert report.explain(identity.name).verdict == "strictly true"
    assert report.explain(bound.name).verdict != "strictly false"
    observed = base.observe(p=1, q=0).with_bound_hint(bound.name, upper=1)
    assert observed.run(Method("finite", lambda current: 1, (identity, bound), uses=("value",))).data == 1
    smooth = differentiable("value", wrt="x")
    assert observed.verify(obligations=(smooth,)).explain(smooth.name).verdict == "strictly false"


def test_uniform_hints_and_targets_share_the_subjects_parameter_assumptions():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        value = a * sin(x)

    context = VerificationContext.for_model(Curve)
    a = context.symbols.a
    context = context.assume(a > 0)
    claim = supremum_below("value", 2 * a, wrt="x")
    for upper, expected in ((3 * a, "unresolved"), (a, "strictly true")):
        evidence = context.with_bound_hint(claim.name, upper=upper).verify(obligations=(claim,)).explain(claim.name)
        assert evidence.verdict == expected


@pytest.mark.parametrize("point", (0, 1))
def test_optional_native_hint_conditions_cannot_refute_the_subject(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        value = sin(x)

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, context.symbols.p)
    hint = Add(1, Mul(0, native, evaluate=False), evaluate=False)
    claim = supremum_below("value", 2, wrt="x")
    context = context.assume(Eq(context.symbols.p, point)).with_bound_hint(claim.name, upper=hint)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == (
        "unresolved" if point == 0 else "strictly true"
    )

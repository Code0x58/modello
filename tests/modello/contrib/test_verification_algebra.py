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


def test_pure_defining_equations_bind_parameters_and_check_feasibility():
    class Implicit(VerificationModel):
        a = InstanceDummy("a", real=True)
        b = 2 * a
        a = b + 1

    case = VerificationContext.for_model(Implicit)
    assert case.expression("a") == -1
    claim = positive("a")
    evidence = case.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "strictly false"
    assert evidence.source_operands
    impossible = case.assume(case.symbols.a > 0)
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

    case = VerificationContext.for_model(Impossible)
    assert case.verify().consistent is False
    with pytest.raises(VerificationError):
        case.run(Method("calculate", lambda current: "must not execute"))


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

    case = VerificationContext.for_model(Impossible)
    assert case.verify().consistent is False
    with pytest.raises(VerificationError):
        case.run(Method("read", lambda current: current.expression("a"), uses=("a",)))


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

    case = VerificationContext.for_model(Scoped)
    assert Scoped.definition_report().consistent is (expected is not None)
    assert case.verify().consistent is (expected is not None)
    assert case.family.is_empty is (expected is None)
    calls = []
    method = Method(
        "read",
        lambda current: calls.append(current) or current.family.expression(current.symbols.value),
        uses=("value",),
    )
    if expected is None:
        with pytest.raises(VerificationError):
            case.run(method)
        assert not calls
    else:
        assert case.run(method).data == expected
        assert calls == [case]


@pytest.mark.parametrize("base", (Modello, VerificationModel))
def test_unsupported_cyclic_calculus_remains_inspectable_and_blocks_execution(base):
    class Cyclic(base):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = a * x
        a = b + x
        slope = Derivative(b, x, evaluate=False)

    case = VerificationContext.for_model(Cyclic)
    report = case.verify()
    assert report.consistent is None
    assert report.checks and all(item.verdict == "unresolved" for item in report.checks)
    assert "unresolved" in report.to_html()
    assert report.to_dict()["checks"]
    with pytest.raises(VerificationError):
        case.run(Method("read", lambda current: "must not execute", uses=("slope",)))


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

    case = VerificationContext.for_model(Scoped)
    method = Method("calculate", lambda current: current.expression("value").doit(), uses=("value",))
    assert case.run(method).data == expected
    assert case.assume(case.symbols.x > 0).run(method).data == expected


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
@pytest.mark.parametrize("source", ("definition", "native", "observation"))
@pytest.mark.parametrize("point", (0, 1))
def test_empty_reductions_check_native_endpoints_before_certifying_identities(operator, identity, source, point):
    class Scoped(Modello):
        x = Coordinate("x", real=True)
        n = Coordinate("n", integer=True)
        endpoint = Subs(Derivative(Abs(x), x), x, point)
        reduction = operator(
            1, (x if operator is Integral else n, endpoint, endpoint if operator is Integral else endpoint - 1)
        )
        value = reduction if source == "definition" else InstanceDummy("value", real=True)

    case = VerificationContext.for_model(Scoped)
    subject = case.expression("reduction") if source == "native" else "value"
    if source == "observation":
        case = case.observe(value=case.expression("reduction"))
    claim = equal_to(subject, identity)
    evidence = case.verify(obligations=(claim,)).explain(claim.name)
    calls = []
    method = Method("identity", lambda current: calls.append(current) or identity, (claim,))
    if point == 0:
        assert evidence.verdict == "strictly false"
        with pytest.raises(VerificationError):
            case.run(method)
        assert not calls
        with pytest.raises(ModelValidationError):
            case.sample(subject, wrt="x", points=(1,))
    else:
        assert evidence.verdict == "strictly true"
        assert case.run(method).data == identity
        assert calls == [case]
        assert tuple(map(float, case.sample(subject, wrt="x", points=(1,)).values)) == (float(identity),)


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
def test_valid_empty_reductions_do_not_check_an_inactive_undefined_derivative(operator, identity):
    class Scoped(Modello):
        x = Coordinate("x", real=True)
        n = Coordinate("n", integer=True)
        invalid = Subs(Derivative(Abs(x), x), x, 0)
        value = operator(invalid, (x if operator is Integral else n, 1, 1 if operator is Integral else 0))

    case = VerificationContext.for_model(Scoped)
    claims = (equal_to("value", identity), nonnegative("value"))
    result = case.run(Method("empty", lambda current: current.expression("value").doit(), claims))
    assert result.data == identity
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
def test_valid_empty_reductions_do_not_require_a_definition_for_their_body(operator, identity):
    class Scoped(Modello):
        x = Coordinate("x", real=True) if operator is Integral else Coordinate("x", integer=True)
        value = operator(Function("f")(x), (x, 1, 1 if operator is Integral else 0))

    case = VerificationContext.for_model(Scoped)
    claims = (equal_to("value", identity), nonnegative("value"))
    result = case.run(Method("empty", lambda current: current.expression("value").doit(), claims))
    assert result.data == identity
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)


@pytest.mark.parametrize(("operator", "identity"), ((Sum, S.Zero), (Product, S.One), (Integral, S.Zero)))
def test_valid_empty_reductions_do_not_inspect_inactive_cyclic_algebra(operator, identity):
    class Scoped(Modello):
        x = Coordinate("x", real=True)
        y = InstanceDummy("y", real=True)
        y = y / 2
        value = operator(y, (x, 1, 1 if operator is Integral else 0))

    case = VerificationContext.for_model(Scoped)
    claims = (equal_to("value", identity), nonnegative("value"))
    result = case.run(Method("empty", lambda current: current.expression("value").doit(), claims))
    assert result.data == identity
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)


def test_native_local_symbols_are_owned_by_their_model_expression():
    index = Symbol("local_index", integer=True)

    class Scoped(VerificationModel):
        value = Sum(1 / index, (index, 1, 2))

    case = VerificationContext.for_model(Scoped)
    result = case.run(Method("calculate", lambda current: current.expression("value").doit(), uses=("value",)))
    assert result.data == S(3) / 2


@pytest.mark.parametrize("coordinate", (Coordinate, InstanceDummy))
@pytest.mark.parametrize("operator", (Sum, Product))
def test_free_variable_assumptions_cannot_certify_a_locally_singular_reduction(coordinate, operator):
    class Scoped(VerificationModel):
        x = coordinate("x", real=True)
        value = operator(1 / x, (x, 0, 1))

    case = VerificationContext.for_model(Scoped)
    local = case.expression("x", wrt="x")
    case = case.assume(local > 0)
    obligations = tuple(item for item in case.verify().checks if item.obligation.consumer == "value")
    assert obligations and all(item.verdict != "strictly true" for item in obligations)
    calls = []
    with pytest.raises(VerificationError):
        case.run(Method("calculate", lambda current: calls.append(current), uses=("value",)))
    assert not calls
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(1,))


def test_a_local_substitution_keeps_its_own_scope_inside_a_reduction():
    class Scoped(VerificationModel):
        x = Coordinate("x", real=True)
        value = Sum(Subs(Derivative(log(x), x), x, 1), (x, 0, 1))

    case = VerificationContext.for_model(Scoped)
    result = case.run(
        Method("calculate", lambda current: current.sample("value", wrt="x", points=(1,)), uses=("value",))
    )
    assert tuple(map(float, result.data.values)) == (2.0,)


def test_nested_reduction_endpoints_keep_their_outer_variable_scope():
    class Nested(VerificationModel):
        x = InstanceDummy("x", integer=True)
        y = InstanceDummy("y", integer=True)
        value = Sum(1, (y, 0, 1 / x), (x, 0, 1))

    case = VerificationContext.for_model(Nested)
    case = case.assume(case.symbols.x > 0)
    assert case.verify().explain("value/auto:1").verdict == "unresolved"
    with pytest.raises(VerificationError):
        case.run(Method("calculate", lambda current: "must not execute", uses=("value",)))


def test_coupled_derivative_scopes_cannot_borrow_free_observations_or_sample_points():
    class Coupled(VerificationModel):
        x = InstanceDummy("x", real=True)
        y = Coordinate("y", real=True)
        value = Sum(Subs(Derivative(x * Abs(y), y), y, 0), (x, 0, 1))

    case = VerificationContext.for_model(Coupled)
    observed = case.observe(x=0)
    assert observed.verify().explain("value/auto:1").verdict == "unresolved"
    with pytest.raises(VerificationError):
        observed.run(Method("calculate", lambda current: "must not execute", uses=("value",)))
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(0,))


@pytest.mark.parametrize("domain", (Interval(1, 2), FiniteSet(1, 2)))
def test_ordinary_derivative_certificates_need_neighborhoods_at_all_requested_points(domain):
    class OneSided(VerificationModel):
        x = Coordinate("x", real=True)
        curve = Piecewise((x, x >= 1), (S.NaN, True))
        value = Sum(Derivative(curve, x), (x, 1, 2))

    case = VerificationContext.for_model(OneSided)
    claim = differentiable("curve", wrt="x", domain=domain)
    report = case.verify(obligations=(claim,))
    assert report.explain(claim.name).verdict == "unresolved"
    assert report.explain("value/auto:1").verdict == "unresolved"
    interior = differentiable("curve", wrt="x", domain=Interval.open(1, 2), name="interior")
    assert case.verify(obligations=(interior,)).explain(interior.name).verdict == "strictly true"
    with pytest.raises(VerificationError):
        case.run(Method("calculate", lambda current: "must not execute", uses=("value",)))


@pytest.mark.parametrize("domain, verdict", ((Interval.open(1, 2), "strictly true"), (Interval(1, 2), "unresolved")))
def test_guarded_curvature_uses_the_defined_body_after_checking_ordinary_neighborhoods(domain, verdict):
    class OneSided(VerificationModel):
        x = Coordinate("x", real=True)
        curve = Piecewise((exp(x), x >= 1), (S.NaN, True))

    case = VerificationContext.for_model(OneSided)
    claim = derivative_ratio_bound("curve", wrt="x", domain=domain)
    assert case.verify(obligations=(claim,)).explain(claim.name).verdict == verdict


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

    case = VerificationContext.for_model(Curve)
    bound = derivative_ratio_bound("rate", wrt="x")
    if nested:
        conclusion = Obligation("target", "rate", "rule")
        case = case.with_rule(InferenceRule("test rule", conclusion, (bound,), "Explicit test rule"))
        case = case.with_obligations(conclusion)
    else:
        case = case.with_obligations(bound)
    report = case.with_bound_hint("rate/derivative_ratio_bound", upper=1).verify()
    assert report.explain("target" if nested else bound.name).verdict == "strictly true"
    assert not any("Unused proof hints" in issue for issue in report.issues)


@pytest.mark.parametrize("kind", ("uniform", "curvature"))
def test_a_loose_conditional_hint_cannot_refute_a_true_bound(kind):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        bound = InstanceDummy("B", real=True)
        value = sin(x) if kind == "uniform" else 1 / (1 + exp(x))

    case = VerificationContext.for_model(Curve)
    obligation = supremum_below if kind == "uniform" else derivative_ratio_bound
    claim = obligation("value", bound=case.symbols.bound, wrt="x")
    hinted = case.with_bound_hint(claim.name, upper=3)
    assert hinted.verify(obligations=(claim,)).explain(claim.name).verdict == "conditionally true"
    valid = hinted.assume(case.symbols.bound > 1, case.symbols.bound < 3)
    assert valid.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    with pytest.raises(VerificationError):
        valid.run(Method("certificate", lambda current: "must not execute", (claim,)))
    tight = valid.with_bound_hint(claim.name, upper=1)
    assert tight.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("point", (0, 1))
def test_optional_native_hint_conditions_cannot_refute_the_subject(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        value = sin(x)

    case = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, case.symbols.p)
    hint = Add(1, Mul(0, native, evaluate=False), evaluate=False)
    claim = supremum_below("value", 2, wrt="x")
    case = case.assume(Eq(case.symbols.p, point)).with_bound_hint(claim.name, upper=hint)
    assert case.verify(obligations=(claim,)).explain(claim.name).verdict == (
        "unresolved" if point == 0 else "strictly true"
    )

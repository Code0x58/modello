"""Declared domains and native premises survive every research input path."""

import pytest
from sympy import Abs, Add, Derivative, FiniteSet, Interval, Mul, Piecewise, Q, S, Subs, arg, assuming, log, sqrt

from modello import Coordinate, Index, InstanceDummy
from modello.contrib.research import (
    Method,
    ResearchCase,
    ResearchError,
    ResearchModel,
    differentiable,
    exact_form,
    nonzero,
    positive,
    smooth,
    uniform_bound,
)
from modello.errors import ModelValidationError


@pytest.mark.parametrize(
    ("assumptions", "valid", "invalid"),
    (
        ({"integer": True}, 2, S.Half),
        ({"rational": True}, S.Half, sqrt(2)),
        ({"irrational": True}, sqrt(2), S.Half),
        ({"even": True}, 2, 1),
        ({"odd": True}, 1, 2),
        ({"zero": True}, 0, 1),
        ({"prime": True}, 2, 4),
        ({"composite": True}, 4, 2),
    ),
)
def test_intrinsic_domains_apply_to_claims_methods_and_sampling(assumptions, valid, invalid):
    class Curve(ResearchModel):
        x = Coordinate("x", **assumptions)
        value = x

    case = ResearchCase.for_model(Curve)
    claim = nonzero(Curve.x - S.Half, wrt="x")
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
    with pytest.raises(ResearchError):
        case.run(Method("real_domain", lambda current: 0, (claim,)))
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(invalid,))
    assert float(case.sample("value", wrt="x", points=(valid,)).values[0]) == pytest.approx(float(valid))


def test_index_and_imaginary_coordinates_do_not_acquire_a_real_continuous_domain():
    class Axes(ResearchModel):
        index = Index("n")
        imaginary = Coordinate("z", imaginary=True)

    case = ResearchCase.for_model(Axes)
    for coordinate in ("index", "imaginary"):
        claim = smooth(case.expression(coordinate), wrt=coordinate)
        assert case.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
        with pytest.raises(ModelValidationError):
            case.sample(case.expression(coordinate), wrt=coordinate, points=(S.Half,))


@pytest.mark.parametrize("point", (0, 1))
@pytest.mark.parametrize("indirect", (False, True))
def test_symbolic_observations_retain_native_operation_premises(point, indirect):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        helper = InstanceDummy("h", real=True)
        value = a * b

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    case = case.observe(a=case.symbols.helper**2, helper=native, b=0) if indirect else case.observe(a=native, b=0)
    claim = smooth("value", wrt="x")
    report = case.check(requirements=(claim,))
    assert report.explain(claim.name).verdict == ("strictly false" if point == 0 else "strictly true")
    method = Method("observe", lambda current: current.expression("value").doit(), uses=("value",))
    if point == 0:
        with pytest.raises(ResearchError):
            case.run(method)
        with pytest.raises(ModelValidationError):
            case.sample("value", wrt="x", points=(1,))
    else:
        assert case.run(method).data == 0
        assert case.sample("value", wrt="x", points=(1,)).values == (0,)


def test_conflicting_and_cyclic_observations_remain_joint_equations():
    class Product(ResearchModel):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a * b
        x = Coordinate("x", real=True)
        slope = Derivative(a * x**2, x)

    case = ResearchCase.for_model(Product)
    conflicting = case.observe(a=1).observe(a=2)
    assert conflicting.check().consistent is False
    compatible = case.observe(a=case.symbols.b).observe(a=1)
    assert compatible.check(requirements=(exact_form("value", S.One),)).explain("exact_form").verdict == "strictly true"
    cyclic = case.observe(a=case.symbols.b, b=case.symbols.a)
    claim = exact_form("value", cyclic.symbols.b**2)
    assert cyclic.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    claim = exact_form("slope", 2 * cyclic.symbols.b * Product.x, wrt="x")
    assert cyclic.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


def test_observed_zero_inside_a_derivative_is_a_valid_specialization():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a * b

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(case.symbols.b * Abs(Curve.x), Curve.x), Curve.x, 0)
    case = case.observe(a=native, b=0)
    assert case.run(Method("zero", lambda current: current.expression("value").doit(), uses=("value",))).data == 0


@pytest.mark.parametrize("point", (0, 1))
def test_cyclic_observations_keep_native_premises_after_joint_elimination(point):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    case = case.observe(a=case.symbols.b, b=case.symbols.a + native - point)
    claim = exact_form("value", S.Zero, wrt="x")
    method = Method("cyclic", lambda current: 0, (claim,))
    with assuming(Q.positive(Curve.x)):
        report = case.check(requirements=(claim,))
        assert report.explain(claim.name).verdict == ("strictly false" if point == 0 else "strictly true")
        assert report.explain(claim.name).observation_sources
        audit = next(row for row in report.to_dict()["checks"] if row["id"] == claim.name)
        assert any("Derivative" in source for source in audit["observation_sources"])
        assert "Observation sources:" in report.to_html()
    if point == 0:
        with pytest.raises(ResearchError):
            case.run(method)
        with pytest.raises(ModelValidationError):
            case.sample("value", wrt="x", points=(1,))
    else:
        assert case.run(method).data == 0
        assert case.sample("value", wrt="x", points=(1,)).values == (0,)


def test_cyclic_observation_guards_keep_local_function_domains():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    case = ResearchCase.for_model(Curve)
    source = Add(case.symbols.a, Mul(0, log(Curve.x), evaluate=False), evaluate=False)
    case = case.observe(a=case.symbols.b, b=source)
    claim = smooth("value", wrt="x")
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"
    claim = smooth("value", wrt="x", domain=Interval.open(0, S.Infinity))
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    claim = differentiable("value", wrt="x", at=1)
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    assert case.sample("value", wrt="x", points=(1,)).values == (0,)
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(0,))


def test_residual_source_guards_preserve_nested_native_substitution_scopes():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        curve = log(x)
        value = a - b

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(case.symbols.curve, Curve.x), Curve.x, 1)
    case = case.observe(a=case.symbols.b, b=case.symbols.a + native - 1)
    claim = exact_form("value", S.Zero, wrt="x")
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    assert case.sample("value", wrt="x", points=(0, 1)).values == (0, 0)


@pytest.mark.parametrize("reverse", (False, True))
def test_proof_cache_distinguishes_a_constant_from_an_equal_unestablished_quantity(reverse):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, 0)
    case = case.observe(a=case.symbols.b, b=case.symbols.a + native)
    claims = (
        exact_form(S.Zero, S.Zero, wrt="x", name="constant"),
        exact_form("value", S.Zero, wrt="x", name="observation"),
    )
    report = case.check(requirements=claims[::-1] if reverse else claims)
    assert report.explain("constant").verdict == "strictly true"
    assert report.explain("observation").verdict == "strictly false"


@pytest.mark.parametrize("point", (0, 1))
def test_domain_parameters_retain_sources_lost_during_joint_solving(point):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    case = case.observe(a=case.symbols.b, b=case.symbols.a + native - point)
    claim = nonzero(Curve.x + 1, wrt="x", domain=FiniteSet(case.symbols.value))
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == (
        "strictly false" if point == 0 else "strictly true"
    )


@pytest.mark.parametrize("point", (0, 1))
def test_optional_hint_parameters_retain_sources_lost_during_joint_solving(point):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        delta = a - b
        value = x**2

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    case = case.observe(a=case.symbols.b, b=case.symbols.a + native - point)
    claim = uniform_bound("value", 2, wrt="x", domain=Interval(0, 1))
    case = case.with_hint(claim.name, upper=1 + case.symbols.delta)
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == (
        "unresolved" if point == 0 else "strictly true"
    )


@pytest.mark.parametrize("payload", ("bound", "point", "domain"))
@pytest.mark.parametrize("point", (0, 1))
def test_required_payloads_check_their_own_native_derivative_scopes(payload, point):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = x**2

    case = ResearchCase.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.y), Curve.y), Curve.y, point)
    constant = Add(2 if payload == "bound" else 1, Mul(0, native, evaluate=False), evaluate=False)
    if payload == "bound":
        claim = uniform_bound("value", constant, wrt="x", domain=Interval(0, 1))
        case = case.with_hint(claim.name, upper=1)
    elif payload == "point":
        claim = differentiable("value", wrt="x", at=constant)
    else:
        claim = positive("value", wrt="x", domain=FiniteSet(constant))
    evidence = case.check(requirements=(claim,)).explain(claim.name)
    assert evidence.verdict == ("strictly false" if point == 0 else "strictly true")
    method = Method("payload", lambda current: 0, (claim,))
    if point == 0:
        with pytest.raises(ResearchError):
            case.run(method)
    else:
        assert case.run(method).data == 0


@pytest.mark.parametrize("point", (0, 1))
def test_an_invalid_optional_hint_is_inapplicable_instead_of_refuting_the_bound(point):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = x**2

    native = Subs(Derivative(Abs(Curve.y), Curve.y), Curve.y, point)
    hint = Add(1, Mul(0, native, evaluate=False), evaluate=False)
    claim = uniform_bound("value", 2, wrt="x", domain=Interval(0, 1))
    case = ResearchCase.for_model(Curve).with_hint(claim.name, upper=hint)
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == (
        "unresolved" if point == 0 else "strictly true"
    )


@pytest.mark.parametrize("function", (arg, lambda x: 1 / x))
def test_ambient_coordinate_assumptions_cannot_erase_source_exclusions(function):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        source = function(x)
        value = a * source

    case = ResearchCase.for_model(Curve).observe(a=0)
    claim = smooth("value", wrt="x")
    with assuming(Q.positive(Curve.x)):
        assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"
    with assuming(Q.nonzero(Curve.x)):
        assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"


@pytest.mark.parametrize("mode", ("declaration", "direct", "cyclic"))
def test_valid_guarded_derivatives_do_not_consume_inactive_undefined_branches(mode):
    if mode == "declaration":

        class Declared(ResearchModel):
            x = Coordinate("x", real=True)
            curve = log(x)
            slope = Derivative(curve, x)
            value = slope - 1 / x

        case = ResearchCase.for_model(Declared)
    else:

        class Observed(ResearchModel):
            x = Coordinate("x", real=True)
            a = InstanceDummy("a", real=True)
            b = InstanceDummy("b", real=True)
            value = a - b

        case = ResearchCase.for_model(Observed)
        native = Derivative(log(Observed.x), Observed.x)
        case = (
            case.observe(a=native, b=1 / Observed.x)
            if mode == "direct"
            else case.observe(a=case.symbols.b, b=case.symbols.a + native - 1 / Observed.x)
        )
    claim = exact_form("value", S.Zero, wrt="x", domain=Interval.open(0, S.Infinity))
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    assert case.run(Method("guarded", lambda current: 0, (claim,))).data == 0
    assert case.sample("value", wrt="x", points=(1, 2)).values == (0, 0)
    with pytest.raises(ModelValidationError):
        case.sample("value", wrt="x", points=(0,))


def test_native_piecewise_premises_respect_branch_domains_and_parameter_guards():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        good = Derivative(x**2, x)
        bad = Subs(Derivative(Abs(y), y), y, 0)
        value = Piecewise((good, x > 0), (bad, True))
        selected = Piecewise((good, switch > 0), (bad, True))

    case = ResearchCase.for_model(Curve)
    claim = smooth("value", wrt="x", domain=Interval.open(0, S.Infinity))
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    assert tuple(map(float, case.sample("value", wrt="x", points=(1,)).values)) == (2.0,)
    claim = exact_form("value", 2 * Curve.x, wrt="x")
    with assuming(Q.positive(Curve.x)):
        assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly false"
    claim = exact_form("selected", 2 * Curve.x, wrt="x", domain=Interval.open(0, S.Infinity))
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
    positive_case = case.assume(case.symbols.switch > 0)
    assert positive_case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


def test_native_operations_in_unreachable_branch_conditions_do_not_refute_the_active_branch():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        bad = Subs(Derivative(Abs(y), y), y, 0)
        value = Piecewise((Derivative(x**2, x), switch > 0), (1, bad > 0), (2, True))

    case = ResearchCase.for_model(Curve)
    claim = smooth("value", wrt="x", order=1)
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
    assert (
        case.assume(case.symbols.switch > 0).check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"
    )
    with assuming(Q.positive(case.symbols.switch)):
        assert case.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


def test_an_unresolved_branch_premise_cannot_refute_a_valid_identity():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        value = Piecewise((Derivative(x**2, x, y, evaluate=False), switch > 0), (Derivative(x**2, x), True))
        expected = Piecewise((0, switch > 0), (2 * x, True))

    case = ResearchCase.for_model(Curve)
    claim = exact_form("value", case.symbols.expected, wrt="x")
    assert case.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
    selected = case.assume(case.symbols.switch > 0)
    assert selected.check(requirements=(claim,)).explain(claim.name).verdict == "unresolved"
    established = case.observe(switch=-1)
    assert established.check(requirements=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("operation", ("derivative", "division"))
def test_methods_consume_only_reachable_automatic_operations(operation):
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        a = InstanceDummy("a", real=True)
        value = Piecewise(
            (Derivative(x**2, x), switch > 0),
            (Subs(Derivative(Abs(y), y), y, 0) if operation == "derivative" else 1 / a, True),
        )

    case = ResearchCase.for_model(Curve).observe(a=0)
    method = Method("read", lambda current: "executed", uses=("value",))
    assert case.observe(switch=1).run(method).data == "executed"
    assert case.assume(case.symbols.switch > 0).run(method).data == "executed"
    with pytest.raises(ResearchError):
        case.run(method)
    obligations = tuple(item for item in case.check().checks if item.requirement.consumer == "value")
    assert any(item.verdict == "unresolved" for item in obligations)
    assert all(item.verdict in {"strictly true", "unresolved"} for item in obligations)
    assert all(not item.witness for item in obligations if item.verdict == "unresolved")
    assert all(item.kind != "counterexample" for item in obligations if item.verdict == "unresolved")
    with pytest.raises(ResearchError):
        case.observe(switch=-1).run(method)


@pytest.mark.parametrize("coordinate", (Coordinate, InstanceDummy))
def test_coordinate_assumptions_cannot_select_away_automatic_branch_operations(coordinate):
    class Curve(ResearchModel):
        x = coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = Piecewise((Derivative(x**2, x), x > 0), (Subs(Derivative(Abs(y), y), y, 0), True))

    case = ResearchCase.for_model(Curve)
    method = Method("read", lambda current: "must not execute", uses=("value",))
    with assuming(Q.positive(case.expression("x", wrt="x"))):
        obligations = tuple(item for item in case.check().checks if item.requirement.consumer == "value")
        assert any(item.verdict == "unresolved" for item in obligations)
        assert all(item.verdict in {"strictly true", "unresolved"} for item in obligations)
        with pytest.raises(ResearchError):
            case.run(method)


def test_globally_certified_operations_do_not_need_branch_selection():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        switch = InstanceDummy("s", real=True)
        value = Piecewise((Derivative(x**2, x), switch > 0), (Derivative(x**3, x), True))

    case = ResearchCase.for_model(Curve)
    method = Method("read", lambda current: "executed", uses=("value",))
    assert case.run(method).data == "executed"


def test_an_unconditional_use_keeps_its_automatic_requirement_after_a_conditional_use():
    class Curve(ResearchModel):
        x = Coordinate("x", real=True)
        switch = InstanceDummy("s", real=True)
        native = Subs(Derivative(Abs(x), x), x, 0)
        value = Add(Subs(Piecewise((native, switch > 0), (1, True)), x, 1), native, evaluate=False)

    case = ResearchCase.for_model(Curve)
    obligations = tuple(item for item in case.check().checks if item.requirement.consumer == "value")
    assert any(item.verdict == "strictly false" for item in obligations)
    with pytest.raises(ResearchError):
        case.run(Method("read", lambda current: "must not execute", uses=("value",)))

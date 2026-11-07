"""Declared domains and native premises survive every verification input path."""

from contextlib import nullcontext

import pytest
from sympy import (
    Abs,
    Add,
    And,
    Derivative,
    Eq,
    FiniteSet,
    Function,
    Integral,
    Interval,
    Mul,
    Piecewise,
    Pow,
    Q,
    Range,
    S,
    Subs,
    Sum,
    arg,
    assuming,
    exp,
    log,
    sqrt,
)

from modello import Coordinate, Index, InstanceDummy
from modello.contrib.verification import (
    InferenceRule,
    Method,
    Obligation,
    VerificationContext,
    VerificationError,
    VerificationModel,
    continuously_differentiable,
    derivative_ratio_bound,
    differentiable,
    equal_to,
    nonnegative,
    nonzero,
    positive,
    predicate,
    specifies,
    supremum_below,
)
from modello.contrib.verification._proof import derivative_premises
from modello.errors import ModelDefinitionError, ModelValidationError


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
    class Curve(VerificationModel):
        x = Coordinate("x", **assumptions)
        value = x

    context = VerificationContext.for_model(Curve)
    claim = nonzero(Curve.x - S.Half, wrt="x")
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    with pytest.raises(VerificationError):
        context.run(Method("real_domain", lambda current: 0, (claim,)))
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(invalid,))
    assert float(context.sample("value", wrt="x", points=(valid,)).values[0]) == pytest.approx(float(valid))


def test_index_and_imaginary_coordinates_do_not_acquire_a_real_continuous_domain():
    class Axes(VerificationModel):
        index = Index("n")
        imaginary = Coordinate("z", imaginary=True)

    context = VerificationContext.for_model(Axes)
    for coordinate in ("index", "imaginary"):
        claim = continuously_differentiable(context.expression(coordinate), wrt=coordinate)
        assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
        with pytest.raises(ModelValidationError):
            context.sample(context.expression(coordinate), wrt=coordinate, points=(S.Half,))


@pytest.mark.parametrize("point", (0, 1))
@pytest.mark.parametrize("indirect", (False, True))
def test_symbolic_observations_retain_native_operation_premises(point, indirect):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        helper = InstanceDummy("h", real=True)
        value = a * b

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    context = (
        context.observe(a=context.symbols.helper**2, helper=native, b=0) if indirect else context.observe(a=native, b=0)
    )
    claim = continuously_differentiable("value", wrt="x")
    report = context.verify(obligations=(claim,))
    assert report.explain(claim.name).verdict == ("strictly false" if point == 0 else "strictly true")
    method = Method("observe", lambda current: current.expression("value").doit(), uses=("value",))
    if point == 0:
        with pytest.raises(VerificationError):
            context.run(method)
        with pytest.raises(ModelValidationError):
            context.sample("value", wrt="x", points=(1,))
    else:
        assert context.run(method).data == 0
        assert context.sample("value", wrt="x", points=(1,)).values == (0,)


def test_conflicting_and_cyclic_observations_remain_joint_equations():
    class Product(VerificationModel):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a * b
        x = Coordinate("x", real=True)
        slope = Derivative(a * x**2, x)

    context = VerificationContext.for_model(Product)
    conflicting = context.observe(a=1).observe(a=2)
    assert conflicting.verify().consistent is False
    compatible = context.observe(a=context.symbols.b).observe(a=1)
    assert compatible.verify(obligations=(equal_to("value", S.One),)).explain("equal_to").verdict == "strictly true"
    cyclic = context.observe(a=context.symbols.b, b=context.symbols.a)
    claim = equal_to("value", cyclic.symbols.b**2)
    assert cyclic.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"
    claim = equal_to("slope", 2 * cyclic.symbols.b * Product.x, wrt="x")
    assert cyclic.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"


def test_observed_zero_inside_a_derivative_is_a_valid_specialization():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a * b

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(context.symbols.b * Abs(Curve.x), Curve.x), Curve.x, 0)
    context = context.observe(a=native, b=0)
    assert context.run(Method("zero", lambda current: current.expression("value").doit(), uses=("value",))).data == 0


@pytest.mark.parametrize("point", (0, 1))
def test_cyclic_observations_keep_native_premises_after_joint_elimination(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    context = context.observe(a=context.symbols.b, b=context.symbols.a + native - point)
    claim = equal_to("value", S.Zero, wrt="x")
    method = Method("cyclic", lambda current: 0, (claim,))
    with assuming(Q.positive(Curve.x)):
        report = context.verify(obligations=(claim,))
        assert report.explain(claim.name).verdict == ("strictly false" if point == 0 else "strictly true")
        assert report.explain(claim.name).source_operands
        audit = next(row for row in report.to_dict()["evidence"] if row["id"] == claim.name)
        assert any("Derivative" in source for source in audit["source_operands"])
        assert "Source operands:" in report.to_html()
    if point == 0:
        with pytest.raises(VerificationError):
            context.run(method)
        with pytest.raises(ModelValidationError):
            context.sample("value", wrt="x", points=(1,))
    else:
        assert context.run(method).data == 0
        assert context.sample("value", wrt="x", points=(1,)).values == (0,)


def test_cyclic_observation_guards_keep_local_function_domains():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    context = VerificationContext.for_model(Curve)
    source = Add(context.symbols.a, Mul(0, log(Curve.x), evaluate=False), evaluate=False)
    context = context.observe(a=context.symbols.b, b=source)
    claim = continuously_differentiable("value", wrt="x")
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly false"
    claim = continuously_differentiable("value", wrt="x", domain=Interval.open(0, S.Infinity))
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"
    claim = differentiable("value", wrt="x", at=1)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"
    assert context.sample("value", wrt="x", points=(1,)).values == (0,)
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(0,))


def test_residual_source_guards_preserve_nested_native_substitution_scopes():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        curve = log(x)
        value = a - b

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(context.symbols.curve, Curve.x), Curve.x, 1)
    context = context.observe(a=context.symbols.b, b=context.symbols.a + native - 1)
    claim = equal_to("value", S.Zero, wrt="x")
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"
    assert context.sample("value", wrt="x", points=(0, 1)).values == (0, 0)


@pytest.mark.parametrize("reverse", (False, True))
def test_proof_cache_distinguishes_a_constant_from_an_equal_unestablished_quantity(reverse):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, 0)
    context = context.observe(a=context.symbols.b, b=context.symbols.a + native)
    claims = (
        equal_to(S.Zero, S.Zero, wrt="x", name="constant"),
        equal_to("value", S.Zero, wrt="x", name="observation"),
    )
    report = context.verify(obligations=claims[::-1] if reverse else claims)
    assert report.explain("constant").verdict == "strictly true"
    assert report.explain("observation").verdict == "strictly false"


@pytest.fixture
def substitution_scopes():
    class Axes(VerificationModel):
        real = Coordinate("x", real=True)
        positive = Coordinate("p", positive=True)
        observed = InstanceDummy("observed", real=True)

    values = {
        name: Subs(Derivative(S.Zero, variable, evaluate=False), variable, 0)
        for name, variable in (("valid", Axes.real), ("outside_domain", Axes.positive))
    }
    assert values["valid"] == values["outside_domain"]
    return VerificationContext.for_model(Axes), values


def substitution_identity(value, payload, name):
    return equal_to(value, S.Zero, name=name) if payload == "subject" else equal_to(S.Zero, value, name=name)


@pytest.mark.parametrize("payload", ("subject", "expected"))
@pytest.mark.parametrize("reverse", (False, True))
def test_equal_substitutions_keep_their_own_domains_across_cached_verification(substitution_scopes, payload, reverse):
    context, values = substitution_scopes
    claims = tuple(substitution_identity(value, payload, name) for name, value in values.items())
    if reverse:
        claims = claims[::-1]
    report = context.verify(obligations=claims)
    assert report.explain("valid").verdict == "strictly true"
    assert report.explain("outside_domain").verdict == "unresolved"
    calls = []
    for claim in claims:
        method = Method(claim.name, lambda current: calls.append(current) or 0, (claim,))
        if claim.name == "valid":
            assert context.run(method).data == 0
        else:
            with pytest.raises(VerificationError):
                context.run(method)
    assert calls == [context]


@pytest.mark.parametrize("payload", ("subject", "expected"))
def test_rules_and_duplicate_ids_distinguish_native_substitution_scopes(substitution_scopes, payload):
    context, values = substitution_scopes
    valid = substitution_identity(values["valid"], payload, "identity")
    invalid = substitution_identity(values["outside_domain"], payload, "identity")
    with pytest.raises(ModelDefinitionError, match="Obligation IDs"):
        context.with_obligations(valid).verify(obligations=(invalid,))
    with pytest.raises(ModelDefinitionError, match="premise names"):
        context.with_rule(InferenceRule("duplicate", valid, (valid, invalid), source="Test rule"))
    trusted = context.with_rule(InferenceRule("real derivative", valid, (), source="Derivative of a constant"))
    assert trusted.verify(obligations=(valid,)).explain("identity").kind == "rule"
    evidence = trusted.verify(obligations=(invalid,)).explain("identity")
    assert evidence.verdict == "unresolved"
    assert evidence.kind == "symbolic"


@pytest.mark.parametrize("reverse", (False, True))
def test_equal_observations_retain_both_native_source_scopes(substitution_scopes, reverse):
    context, values = substitution_scopes
    observations = tuple(values.values())
    for value in observations[::-1] if reverse else observations:
        context = context.observe(observed=value)
    claim = equal_to("observed", S.Zero)
    report = context.verify(obligations=(claim,))
    evidence = report.explain(claim.name)
    assert evidence.verdict == "unresolved"
    assert {value.variables for value in evidence.source_operands} == {value.variables for value in observations}
    with pytest.raises(VerificationError):
        context.run(Method("observation", lambda current: 0, (claim,)))
    with pytest.raises(ModelValidationError):
        context.sample("observed", wrt="real", points=(1,))


@pytest.mark.parametrize("substituted", (False, True))
def test_predicate_binding_preserves_native_derivative_operands(substituted):
    class Axes(VerificationModel):
        x = Coordinate("x", real=True)
        p = Coordinate("p", positive=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    context = VerificationContext.for_model(Axes).observe(a=37 if substituted else 1, b=37 if substituted else 1)
    left = Derivative(context.symbols.a * (Axes.x if substituted else Abs(Axes.x)), Axes.x, evaluate=False)
    right = Derivative(
        context.symbols.b * (Axes.p if substituted else Abs(Axes.x)), Axes.p if substituted else Axes.x, evaluate=False
    )
    if substituted:
        left, right = Subs(left, Axes.x, 0), Subs(right, Axes.p, 0)
        assert (
            context.verify(obligations=(equal_to(exp(left), exp(37)),)).explain("equal_to").verdict == "strictly true"
        )
        unsafe = equal_to(exp(right), exp(37))
        assert context.verify(obligations=(unsafe,)).explain(unsafe.name).verdict == "unresolved"
    claim = predicate(Eq(left, right, evaluate=False))
    assert context.verify(obligations=(claim,)).explain(claim.name).condition is not S.true
    with pytest.raises(VerificationError):
        context.run(Method("predicate", lambda current: 0, (claim,)))


@pytest.mark.parametrize("point", (0, 1))
def test_domain_parameters_retain_sources_lost_during_joint_solving(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        value = a - b

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    context = context.observe(a=context.symbols.b, b=context.symbols.a + native - point)
    claim = nonzero(Curve.x + 1, wrt="x", domain=FiniteSet(context.symbols.value))
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == (
        "strictly false" if point == 0 else "strictly true"
    )


@pytest.mark.parametrize("point", (0, 1))
def test_optional_hint_parameters_retain_sources_lost_during_joint_solving(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)
        delta = a - b
        value = x**2

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x), Curve.x, point)
    context = context.observe(a=context.symbols.b, b=context.symbols.a + native - point)
    claim = supremum_below("value", 2, wrt="x", domain=Interval(0, 1))
    context = context.with_bound_hint(claim.name, upper=1 + context.symbols.delta)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == (
        "unresolved" if point == 0 else "strictly true"
    )


@pytest.mark.parametrize("payload", ("bound", "point", "domain"))
@pytest.mark.parametrize("point", (0, 1))
def test_required_payloads_verify_their_own_native_derivative_scopes(payload, point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = x**2

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.y), Curve.y), Curve.y, point)
    constant = Add(2 if payload == "bound" else 1, Mul(0, native, evaluate=False), evaluate=False)
    if payload == "bound":
        claim = supremum_below("value", constant, wrt="x", domain=Interval(0, 1))
        context = context.with_bound_hint(claim.name, upper=1)
    elif payload == "point":
        claim = differentiable("value", wrt="x", at=constant)
    else:
        claim = positive("value", wrt="x", domain=FiniteSet(constant))
    evidence = context.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == ("strictly false" if point == 0 else "strictly true")
    method = Method("payload", lambda current: 0, (claim,))
    if point == 0:
        with pytest.raises(VerificationError):
            context.run(method)
    else:
        assert context.run(method).data == 0


@pytest.mark.parametrize("point", (0, 1))
def test_an_invalid_optional_hint_is_inapplicable_instead_of_refuting_the_bound(point):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = x**2

    native = Subs(Derivative(Abs(Curve.y), Curve.y), Curve.y, point)
    hint = Add(1, Mul(0, native, evaluate=False), evaluate=False)
    claim = supremum_below("value", 2, wrt="x", domain=Interval(0, 1))
    context = VerificationContext.for_model(Curve).with_bound_hint(claim.name, upper=hint)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == (
        "unresolved" if point == 0 else "strictly true"
    )


@pytest.mark.parametrize("function", (arg, lambda x: 1 / x))
def test_ambient_coordinate_assumptions_cannot_erase_source_exclusions(function):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        source = function(x)
        value = a * source

    context = VerificationContext.for_model(Curve).observe(a=0)
    claim = continuously_differentiable("value", wrt="x")
    with assuming(Q.positive(Curve.x)):
        assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly false"
    with assuming(Q.nonzero(Curve.x)):
        assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly false"


@pytest.mark.parametrize("mode", ("declaration", "direct", "cyclic"))
def test_valid_guarded_derivatives_do_not_consume_inactive_undefined_branches(mode):
    if mode == "declaration":

        class Declared(VerificationModel):
            x = Coordinate("x", real=True)
            curve = log(x)
            slope = Derivative(curve, x)
            value = slope - 1 / x

        context = VerificationContext.for_model(Declared)
    else:

        class Observed(VerificationModel):
            x = Coordinate("x", real=True)
            a = InstanceDummy("a", real=True)
            b = InstanceDummy("b", real=True)
            value = a - b

        context = VerificationContext.for_model(Observed)
        native = Derivative(log(Observed.x), Observed.x)
        context = (
            context.observe(a=native, b=1 / Observed.x)
            if mode == "direct"
            else context.observe(a=context.symbols.b, b=context.symbols.a + native - 1 / Observed.x)
        )
    claim = equal_to("value", S.Zero, wrt="x", domain=Interval.open(0, S.Infinity))
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"
    assert context.run(Method("guarded", lambda current: 0, (claim,))).data == 0
    assert context.sample("value", wrt="x", points=(1, 2)).values == (0, 0)
    with pytest.raises(ModelValidationError):
        context.sample("value", wrt="x", points=(0,))


def test_native_piecewise_premises_respect_branch_domains_and_parameter_guards():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        good = Derivative(x**2, x)
        bad = Subs(Derivative(Abs(y), y), y, 0)
        value = Piecewise((good, x > 0), (bad, True))
        selected = Piecewise((good, switch > 0), (bad, True))

    context = VerificationContext.for_model(Curve)
    claim = continuously_differentiable("value", wrt="x", domain=Interval.open(0, S.Infinity))
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"
    assert tuple(map(float, context.sample("value", wrt="x", points=(1,)).values)) == (2.0,)
    claim = equal_to("value", 2 * Curve.x, wrt="x")
    with assuming(Q.positive(Curve.x)):
        assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly false"
    claim = equal_to("selected", 2 * Curve.x, wrt="x", domain=Interval.open(0, S.Infinity))
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    positive_context = context.assume(context.symbols.switch > 0)
    assert positive_context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"


def test_native_operations_in_unreachable_branch_conditions_do_not_refute_the_active_branch():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        bad = Subs(Derivative(Abs(y), y), y, 0)
        value = Piecewise((Derivative(x**2, x), switch > 0), (1, bad > 0), (2, True))

    context = VerificationContext.for_model(Curve)
    claim = continuously_differentiable("value", wrt="x", order=1)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    assert (
        context.assume(context.symbols.switch > 0).verify(obligations=(claim,)).explain(claim.name).verdict
        == "strictly true"
    )
    with assuming(Q.positive(context.symbols.switch)):
        assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"


def test_an_unresolved_branch_premise_cannot_refute_a_valid_identity():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        value = Piecewise((Derivative(x**2, x, y, evaluate=False), switch > 0), (Derivative(x**2, x), True))
        expected = Piecewise((0, switch > 0), (2 * x, True))

    context = VerificationContext.for_model(Curve)
    claim = equal_to("value", context.symbols.expected, wrt="x")
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    selected = context.assume(context.symbols.switch > 0)
    assert selected.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    established = context.observe(switch=-1)
    assert established.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true"


@pytest.mark.parametrize("operation", ("derivative", "division"))
def test_methods_consume_only_reachable_automatic_operations(operation):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        switch = InstanceDummy("s", real=True)
        a = InstanceDummy("a", real=True)
        value = Piecewise(
            (Derivative(x**2, x), switch > 0),
            (Subs(Derivative(Abs(y), y), y, 0) if operation == "derivative" else 1 / a, True),
        )

    context = VerificationContext.for_model(Curve).observe(a=0)
    method = Method("read", lambda current: "executed", uses=("value",))
    assert context.observe(switch=1).run(method).data == "executed"
    assert context.assume(context.symbols.switch > 0).run(method).data == "executed"
    with pytest.raises(VerificationError):
        context.run(method)
    obligations = tuple(item for item in context.verify().evidence if item.obligation.consumer == "value")
    assert any(item.verdict == "unresolved" for item in obligations)
    assert all(item.verdict in {"strictly true", "unresolved"} for item in obligations)
    assert all(not item.witness for item in obligations if item.verdict == "unresolved")
    assert all(item.kind != "counterexample" for item in obligations if item.verdict == "unresolved")
    with pytest.raises(VerificationError):
        context.observe(switch=-1).run(method)


@pytest.mark.parametrize("coordinate", (Coordinate, InstanceDummy))
def test_coordinate_assumptions_cannot_select_away_automatic_branch_operations(coordinate):
    class Curve(VerificationModel):
        x = coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = Piecewise((Derivative(x**2, x), x > 0), (Subs(Derivative(Abs(y), y), y, 0), True))

    context = VerificationContext.for_model(Curve)
    method = Method("read", lambda current: "must not execute", uses=("value",))
    with assuming(Q.positive(context.expression("x", wrt="x"))):
        obligations = tuple(item for item in context.verify().evidence if item.obligation.consumer == "value")
        assert any(item.verdict == "unresolved" for item in obligations)
        assert all(item.verdict in {"strictly true", "unresolved"} for item in obligations)
        with pytest.raises(VerificationError):
            context.run(method)


def test_globally_certified_operations_do_not_need_branch_selection():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        switch = InstanceDummy("s", real=True)
        value = Piecewise((Derivative(x**2, x), switch > 0), (Derivative(x**3, x), True))

    context = VerificationContext.for_model(Curve)
    method = Method("read", lambda current: "executed", uses=("value",))
    assert context.run(method).data == "executed"


def test_an_unconditional_use_keeps_its_automatic_obligation_after_a_conditional_use():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        switch = InstanceDummy("s", real=True)
        native = Subs(Derivative(Abs(x), x), x, 0)
        value = Add(Subs(Piecewise((native, switch > 0), (1, True)), x, 1), native, evaluate=False)

    context = VerificationContext.for_model(Curve)
    obligations = tuple(item for item in context.verify().evidence if item.obligation.consumer == "value")
    assert any(item.verdict == "strictly false" for item in obligations)
    with pytest.raises(VerificationError):
        context.run(Method("read", lambda current: "must not execute", uses=("value",)))


@pytest.mark.parametrize("ambient", (False, True))
@pytest.mark.parametrize("form", ("direct", "alias", "predicate"))
def test_parameter_branch_selection_uses_resolved_assumptions(form, ambient):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        switch = InstanceDummy("switch", real=True)
        doubled = 2 * switch
        value = Piecewise((Derivative(Abs(x), x, evaluate=False), switch < 0), (1, True))

    context = VerificationContext.for_model(Curve)
    condition = {
        "direct": context.symbols.switch > 0,
        "alias": context.symbols.doubled > 0,
        "predicate": Q.positive(context.symbols.doubled),
    }[form]
    assumption_scope = assuming(condition) if ambient else nullcontext()
    context = context if ambient else context.assume(condition)
    with assumption_scope:
        result = context.run(Method("read", lambda current: 1, uses=("value",)))
    assert result.data == 1
    assert not any(item.obligation.consumer == "value" for item in result.report.evidence)


@pytest.mark.parametrize("ambient", (False, True))
@pytest.mark.parametrize("predicates", (False, True))
def test_coordinate_coupled_assumptions_cannot_select_away_local_operations(ambient, predicates):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        switch = InstanceDummy("switch", real=True)
        value = Piecewise((Derivative(Abs(x), x, evaluate=False), switch < 0), (1, True))

    context = VerificationContext.for_model(Curve)
    conditions = (
        (Q.zero(context.symbols.switch - Curve.x), Q.positive(Curve.x))
        if predicates
        else (Eq(context.symbols.switch, Curve.x), Curve.x > 0)
    )
    assumption_scope = assuming(*conditions) if ambient else nullcontext()
    context = context if ambient else context.assume(*conditions)
    with assumption_scope:
        with pytest.raises(VerificationError) as failure:
            context.run(Method("read", lambda current: "must not execute", uses=("value",)))
    obligations = tuple(item for item in failure.value.report.evidence if item.obligation.consumer == "value")
    assert obligations and all(item.verdict == "unresolved" for item in obligations)


@pytest.fixture(params=("self", "mutual"))
def cyclic_coordinate_context(request):
    class Implicit(VerificationModel):
        x = Coordinate("x", real=True)
        a = InstanceDummy("a", real=True)
        if request.param == "mutual":
            b = 2 * a
            a = b + 1
        else:
            a = 2 * a
        value = a**2

    return VerificationContext.for_model(Implicit), 1 if request.param == "mutual" else 0


@pytest.mark.parametrize("operation", ("expression", "verify", "sample", "method"))
@pytest.mark.parametrize("native", (False, True))
def test_cyclic_definitions_cannot_supply_independent_coordinates(cyclic_coordinate_context, operation, native):
    context, expected = cyclic_coordinate_context
    coordinate = context.symbols.a if native else "a"
    calls = []
    for subject in ("value", context.symbols.a**2):
        claim = equal_to(subject, expected, wrt=coordinate)
        with pytest.raises(ModelDefinitionError, match="reparameterize derived quantities"):
            if operation == "expression":
                context.expression(subject, wrt=coordinate)
            elif operation == "verify":
                context.verify(obligations=(claim,))
            elif operation == "sample":
                context.sample(subject, wrt=coordinate, points=(2,))
            else:
                context.run(Method("identity", lambda current: calls.append(current), (claim,)))
    assert not calls


def test_cyclic_definition_dependents_are_not_independent_coordinates(cyclic_coordinate_context):
    context, _ = cyclic_coordinate_context
    with pytest.raises(ModelDefinitionError, match="reparameterize derived quantities"):
        context.expression("a", wrt="value")


def test_cyclic_coordinates_are_rejected_when_model_obligations_are_attached(cyclic_coordinate_context):
    context, _ = cyclic_coordinate_context
    model = context.problem.model
    previous = model.definition_report()
    with pytest.raises(ModelDefinitionError, match="reparameterize derived quantities"):
        specifies(model, continuously_differentiable("value", wrt="a"))
    assert model.definition_report() is previous


def test_implicit_quantities_remain_available_under_an_independent_coordinate(cyclic_coordinate_context):
    context, expected = cyclic_coordinate_context
    claims = tuple(
        equal_to(subject, expected, wrt="x", name=name)
        for name, subject in (("named", "value"), ("native", context.symbols.a**2))
    )
    result = context.run(Method("identity", lambda current: current.expression("value"), claims))
    assert result.data == expected
    assert all(result.report.explain(claim.name).verdict == "strictly true" for claim in claims)
    assert tuple(map(float, context.sample("value", wrt="x", points=(0, 1)).values)) == (float(expected),) * 2


@pytest.mark.parametrize("coordinate", ("a", "alias"))
@pytest.mark.parametrize("observed", ("numeric", "cyclic"))
def test_observations_preserve_independent_coordinates_and_their_aliases(coordinate, observed):
    class Parameters(VerificationModel):
        a = InstanceDummy("a", real=True)
        alias = a
        b = InstanceDummy("b", real=True)
        value = a**2

    context = VerificationContext.for_model(Parameters)
    context = (
        context.observe(a=3) if observed == "numeric" else context.observe(a=context.symbols.b, b=context.symbols.a)
    )
    expected = context.symbols.a**2
    assert context.expression("value", wrt=coordinate) == expected
    claim = equal_to("value", expected, wrt=coordinate)
    assert (
        context.run(Method("identity", lambda current: current.expression("value", wrt=coordinate), (claim,))).data
        == expected
    )
    assert tuple(map(float, context.sample("value", wrt=coordinate, points=(2,)).values)) == (4.0,)


@pytest.mark.parametrize("observed", (False, True))
@pytest.mark.parametrize("binding", ("direct", "observed_alias", "cyclic_observation"))
@pytest.mark.parametrize("native", (False, True))
def test_joint_state_solutions_cannot_freeze_a_protected_implicit_curve(binding, observed, native):
    class Implicit(VerificationModel):
        b = InstanceDummy("b", real=True)
        c = InstanceDummy("c", real=True)
        a = InstanceDummy("a", real=True)
        a = a / 2 + (b if binding == "direct" else c)
        value = a**2

    context = VerificationContext.for_model(Implicit)
    if binding != "direct":
        c = context.symbols.b if binding == "observed_alias" else 2 * context.symbols.c + context.symbols.b
        context = context.observe(c=c)
    if observed:
        context = context.observe(b=1)
        assert context.expression("value") == 4
    subject = context.symbols.a**2 if native else "value"
    claim = equal_to(subject, 4 * context.symbols.b**2, wrt="b")
    evidence = context.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "unresolved"
    assert "cyclic model definitions" in " ".join(evidence.steps)
    assert not evidence.witness
    with pytest.raises(NotImplementedError, match="cyclic model definitions"):
        context.expression(subject, wrt="b")
    with pytest.raises(NotImplementedError, match="cyclic model definitions"):
        context.sample(subject, wrt="b", points=(2,))
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("implicit curve", lambda current: calls.append(current), (claim,)))
    assert not calls


@pytest.mark.parametrize("observed", (False, True))
@pytest.mark.parametrize("binding", ("direct", "nested", "equivalent_equations"))
@pytest.mark.parametrize("native", (False, True))
def test_observation_queries_preserve_implicit_curve_scopes(binding, observed, native):
    class Implicit(VerificationModel):
        x = Coordinate("x", real=True)
        b = InstanceDummy("b", real=True)
        a = InstanceDummy("a", real=True)
        a = a / 2 + b
        middle = InstanceDummy("middle", real=True)
        output = InstanceDummy("output", real=True)

    context = VerificationContext.for_model(Implicit)
    if binding == "nested":
        context = context.observe(middle=context.symbols.a, output=context.symbols.middle**2)
    else:
        context = context.observe(output=context.symbols.a**2)
        if binding == "equivalent_equations":
            context = context.observe(output=2 * context.symbols.a * context.symbols.b)
    if observed:
        context = context.observe(b=1)
        assert context.expression("output") == 4
        assert context.expression("output", wrt="x") == 4
        independent = equal_to("output", 4, wrt="x", name="independent")
        assert context.verify(obligations=(independent,)).explain("independent").verdict == "strictly true"
    subject = context.symbols.output if native else "output"
    claim = equal_to(subject, 4, wrt="b")
    evidence = context.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == "unresolved"
    assert "cyclic model definitions" in " ".join(evidence.steps)
    assert not evidence.witness
    with pytest.raises(NotImplementedError, match="cyclic model definitions"):
        context.expression(subject, wrt="b")
    with pytest.raises(NotImplementedError, match="cyclic model definitions"):
        context.sample(subject, wrt="b", points=(2,))
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("observed implicit curve", lambda current: calls.append(current), (claim,)))
    assert not calls


@pytest.mark.parametrize(
    ("point", "expected", "verdict"),
    (
        (S.NegativeInfinity, -1, "unresolved"),
        (S.Infinity, 1, "unresolved"),
        (S.ImaginaryUnit, S.ImaginaryUnit, "unresolved"),
        (-1, -1, "strictly true"),
        (1, 1, "strictly true"),
        (0, 0, "strictly false"),
    ),
)
def test_native_local_derivative_premises_verify_point_domains(point, expected, verdict):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = x

    context = VerificationContext.for_model(Curve)
    native = Subs(Derivative(Abs(Curve.x), Curve.x, evaluate=False), Curve.x, point)
    claim = equal_to(native, expected)
    evidence = context.verify(obligations=(claim,)).explain(claim.name)
    assert evidence.verdict == verdict
    calls = []
    method = Method("local derivative", lambda current: calls.append(current), (claim,))
    if verdict == "strictly true":
        context.run(method)
        assert calls == [context]
    else:
        with pytest.raises(VerificationError):
            context.run(method)
        assert not calls


@pytest.mark.parametrize(("assumptions", "domain"), (({"integer": True}, S.Integers), ({"real": True}, FiniteSet(1))))
def test_local_derivatives_need_a_real_neighborhood_in_their_domain(assumptions, domain):
    class Curve(VerificationModel):
        x = Coordinate("x", **assumptions)
        value = Abs(x)

    context = VerificationContext.for_model(Curve)
    claim = differentiable("value", wrt="x", domain=domain, at=1)
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    if domain is S.Integers:
        native = equal_to(Subs(Derivative(context.symbols.value, Curve.x), Curve.x, 1), 1)
        assert context.verify(obligations=(native,)).explain(native.name).verdict == "unresolved"


@pytest.mark.parametrize(
    ("assumptions", "domain", "point"),
    (
        ({"integer": True}, S.Integers, S.One),
        ({"rational": True}, S.Rationals, S.Half),
        ({"irrational": True}, FiniteSet(sqrt(2)), sqrt(2)),
        ({"even": True}, FiniteSet(2), S(2)),
        ({"odd": True}, FiniteSet(1), S.One),
        ({"zero": True}, FiniteSet(0), S.Zero),
        ({"prime": True}, FiniteSet(2), S(2)),
        ({"composite": True}, FiniteSet(4), S(4)),
    ),
)
def test_intrinsically_noncontinuous_coordinates_do_not_supply_ordinary_derivatives(assumptions, domain, point):
    class Curve(VerificationModel):
        x = Coordinate("x", **assumptions)
        value = x**2
        cusp = Abs(x)
        slope = Derivative(value, x)

    context = VerificationContext.for_model(Curve)
    claims = (
        continuously_differentiable("value", wrt="x", domain=domain),
        differentiable("value", wrt="x", domain=domain),
        differentiable("cusp", wrt="x", domain=domain, name="cusp"),
        derivative_ratio_bound("value", wrt="x", domain=domain),
        equal_to("slope", 2 * Curve.x),
    )
    for claim in claims:
        evidence = context.verify(obligations=(claim,)).explain(claim.name)
        assert evidence.verdict == "unresolved"
        assert not evidence.witness
        calls = []
        with pytest.raises(VerificationError):
            context.run(Method("ordinary calculus", lambda current: calls.append(current), (claim,)))
        assert not calls
    automatic = context.verify().evidence
    assert automatic and all(item.verdict == "unresolved" for item in automatic)
    with pytest.raises(VerificationError):
        context.run(Method("slope", lambda current: current.expression("slope").doit(), uses=("slope",)))
    constrained = specifies(Curve, claims[0])
    assert constrained.definition_report().explain("continuously_differentiable").verdict == "unresolved"
    values = (nonnegative("value"), equal_to("value", Curve.x**2))
    assert all(context.verify(obligations=(claim,)).explain(claim.name).verdict == "strictly true" for claim in values)
    assert float(context.sample("value", wrt="x", points=(point,)).values[0]) == pytest.approx(float(point**2))


@pytest.mark.parametrize("domain", (FiniteSet(1, 2), Range(1, 3), S.Integers))
def test_real_curves_keep_ordinary_derivatives_on_discrete_evaluation_sets(domain):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        value = x**2
        slope = Derivative(value, x)
        total = Sum(slope, (x, 1, 2))

    context = VerificationContext.for_model(Curve)
    claims = (
        continuously_differentiable("value", wrt="x", domain=domain),
        differentiable("value", wrt="x", domain=domain),
        equal_to("slope", 2 * Curve.x, wrt="x", domain=domain),
    )
    result = context.run(
        Method("sum slopes", lambda current: current.expression("total").doit(), claims, uses=("total",))
    )
    assert result.data == 6
    assert all(item.verdict == "strictly true" for item in result.report.evidence)


@pytest.mark.parametrize("nonnegative_coordinate", (False, True))
def test_ordinary_derivative_certificates_preserve_closed_real_intervals(nonnegative_coordinate):
    class Curve(VerificationModel):
        x = Coordinate("x", nonnegative=True) if nonnegative_coordinate else Coordinate("x", real=True)
        value = x**2
        slope = Derivative(value, x)

    context = VerificationContext.for_model(Curve)
    claims = (
        continuously_differentiable("value", wrt="x", domain=Interval(0, 1)),
        differentiable("value", wrt="x", domain=Interval(0, 1)),
    )
    result = context.run(Method("slope", lambda current: current.expression("slope").doit(), claims, uses=("slope",)))
    assert result.data == 2 * Curve.x
    assert all(item.verdict == "strictly true" for item in result.report.evidence)


@pytest.mark.parametrize(
    ("order", "offset", "valid"),
    (
        ("parameter", 0, False),
        ("parameter", 1, True),
        ("parameter_first", 0, False),
        ("parameter_first", 1, True),
        ("coordinate_first", 0, False),
        ("coordinate_first", 1, False),
    ),
)
def test_native_substitutions_do_not_capture_the_derivative_coordinate(order, offset, valid):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        _slope = Derivative(Abs(x - y), x)
        if order == "parameter":
            value = Subs(_slope, y, x - offset)
        elif order == "parameter_first":
            value = Subs(_slope, (y, x), (x - offset, y))
        else:
            value = Subs(_slope, (x, y), (y, x - offset))

    expected = "strictly true" if valid else "strictly false"
    assert Curve.definition_report().explain("value/auto:1").verdict == expected
    context = VerificationContext.for_model(Curve)
    identity = equal_to("value", int(valid), name="identity")
    report = context.verify(obligations=(identity,))
    assert report.explain(identity.name).verdict == expected
    assert report.to_dict()["evidence_sha256"] == context.verify(obligations=(identity,)).to_dict()["evidence_sha256"]
    calls = []

    def execute(current):
        calls.append(current)
        return current.expression("value").doit()

    methods = (Method("automatic", execute, uses=("value",)), Method("explicit", execute, (identity,)))
    for method in methods:
        if valid:
            assert context.run(method).data == 1
        else:
            with pytest.raises(VerificationError):
                context.run(method)
    assert len(calls) == (2 if valid else 0)


def test_a_constant_function_remains_smooth_under_overlapping_native_bindings():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        a = InstanceDummy("a", real=True)
        value = Subs(Derivative(Abs(a * x - y), x), y, a * x)

    assert Curve.definition_report().explain("value/auto:1").verdict == "conditionally true"
    context = VerificationContext.for_model(Curve)
    calls = []
    method = Method(
        "slope", lambda current: calls.append(current) or current.expression("value").doit(), uses=("value",)
    )
    assert context.observe(a=0).run(method).data == 0
    with pytest.raises(VerificationError) as failure:
        context.observe(a=1).run(method)
    assert failure.value.report.explain("value/auto:1").verdict == "strictly false"
    assert len(calls) == 1


def test_compound_native_bindings_preserve_denominator_sources():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        value = Subs(1 / (x**2 - 1), (x, x**2), (p, 2))

    context = VerificationContext.for_model(Curve)
    invalid = context.assume(Eq(context.symbols.p, 1))
    report = invalid.verify()
    assert report.consistent is True
    assert report.explain("value/auto:1").condition is not S.true
    calls = []
    method = Method(
        "value", lambda current: calls.append(current) or current.expression("value").doit(), uses=("value",)
    )
    with pytest.raises(VerificationError):
        invalid.run(method)
    assert not calls
    valid = context.observe(p=2)
    assert valid.run(method).data == S.One / 3
    assert calls == [valid]


def test_compound_substitutions_support_polynomials_but_do_not_certify_derivative_scopes():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        polynomial = Subs(x**2 + y**2, x**2, 2)
        slope = Subs(Derivative(Abs(x**2 - 1), x), x**2, 2)

    context = VerificationContext.for_model(Curve)
    report = context.verify()
    assert report.explain("slope/auto:1").verdict == "unresolved"
    method = Method("polynomial", lambda current: current.expression("polynomial").doit(), uses=("polynomial",))
    assert context.run(method).data == Curve.y**2 + 2
    with pytest.raises(VerificationError):
        context.run(Method("slope", lambda current: "must not execute", uses=("slope",)))


@pytest.mark.parametrize("layout", ("sibling", "inside", "compound_sibling", "compound_inside", "integrand"))
def test_native_premises_do_not_require_finiteness_of_unrelated_symbolic_values(layout):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        upper = InstanceDummy("upper", positive=True)
        _slope = Derivative(x + y, x)
        _formal = Function("f")(x)
        if layout == "sibling":
            value = Subs(_slope, y, x) + _formal
        elif layout == "inside":
            value = Subs(_slope + _formal, y, x)
        elif layout == "compound_sibling":
            value = Subs(x**2, x**2, 2) + _formal
        elif layout == "compound_inside":
            value = Subs(x**2 + _formal, x**2, 2)
        else:
            value = Integral(Subs(_slope, y, x) + _formal, (x, 0, upper))

    context = VerificationContext.for_model(Curve)
    method = Method("inspect", lambda current: current.expression("value"), uses=("value",))
    result = context.run(method)
    assert result.data == context.expression("value")
    assert all(item.verdict == "strictly true" for item in result.report.evidence)


@pytest.mark.parametrize("selector", ("square", "fourth", "observed_alias"))
def test_compound_substitutions_do_not_prune_a_native_derivative_using_the_unsubstituted_selector(selector):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        p = InstanceDummy("p", real=True)
        a = InstanceDummy("a", real=True)
        value = Subs(
            Piecewise(
                (Derivative(Abs(x), x), Eq(p**2 if selector == "square" else p**4 if selector == "fourth" else a, 0)),
                (1, True),
            ),
            p**2,
            0,
        )

    context = VerificationContext.for_model(Curve)
    context = context.observe(a=context.symbols.p**4, p=1) if selector == "observed_alias" else context.observe(p=1)
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("slope", lambda current: calls.append(current), uses=("value",)))
    assert not calls


def test_native_range_witnesses_restore_the_original_coordinate_in_keys_and_values():
    x, y, z = (Coordinate(name, real=True) for name in ("x", "y", "z"))
    body = Subs(Derivative(Abs(x - y - z), x), y, x - z)
    source = Subs(Sum(body, (x, -1, 1)), z, x)
    failure = next(item for item in derivative_premises(source, x, S.Reals) if item.condition is S.false)
    assert failure.witness == ((x, x),)


def test_a_ranged_denominator_does_not_identify_its_axis_with_an_external_value():
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = Subs(Sum(1 / (x - y), (x, 1, 2)), y, x - 1)

    context = VerificationContext.for_model(Curve).assume(Eq(Curve.x, 2))
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("sum", lambda current: calls.append(current), uses=("value",)))
    assert not calls


@pytest.mark.parametrize("ranged", (False, True))
def test_unrelated_compound_patterns_keep_inactive_native_operations_inactive(ranged):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        a = InstanceDummy("a", real=True)
        q = InstanceDummy("q", real=True)
        if ranged:
            value = Subs(Sum(Piecewise((1 / q, And(a > 0, y > 0)), (1, True)), (y, 1, 2)), x**2, 2)
        else:
            value = Subs(Piecewise((1 / q, a > 0), (1, True)), x**2, 2)

    context = VerificationContext.for_model(Curve).observe(a=-1, q=0)
    result = context.run(Method("inspect", lambda current: current.expression("value"), uses=("value",)))
    assert result.data.doit() == (2 if ranged else 1)
    assert result.report.consistent is True


@pytest.mark.parametrize("pointed", (False, True))
@pytest.mark.parametrize("operation", ("derivative", "division"))
def test_captured_native_operands_keep_sources_before_observations_can_collapse_them(pointed, operation):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        a = InstanceDummy("a", real=True)
        _body = Mul(a, y + log(x), evaluate=False)
        _operation = (
            Derivative(_body, x, evaluate=False)
            if operation == "derivative"
            else Pow(Add(1, _body, evaluate=False), -1, evaluate=False)
        )
        value = Subs(_operation, (x, y), (1, x)) if pointed else Subs(_operation, y, x)

    context = VerificationContext.for_model(Curve).observe(a=0)
    calls = []
    method = Method(
        "read", lambda current: calls.append(current) or current.expression("value").doit(), uses=("value",)
    )
    if pointed:
        assert context.run(method).data == (0 if operation == "derivative" else 1)
        assert calls == [context]
    else:
        with pytest.raises(VerificationError):
            context.run(method)
        assert not calls


def test_an_unregistered_rule_does_not_evaluate_unsupported_native_operations():
    class Parameters(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        q = InstanceDummy("q", real=True)

    context = VerificationContext.for_model(Parameters).observe(q=0)
    subject = Subs(Sum(1 / context.symbols.q, (Parameters.y, 1, 2)), Parameters.x**2, 2)
    claim = Obligation("unproved", subject, "rule")
    assert context.verify(obligations=(claim,)).explain(claim.name).verdict == "unresolved"
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("unproved", lambda current: calls.append(current), (claim,)))
    assert not calls


@pytest.mark.parametrize("ranged", (False, True))
def test_unrelated_compound_patterns_preserve_established_selector_domains(ranged):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        q = InstanceDummy("q", real=True)
        if ranged:
            value = Subs(Sum(Piecewise((2, And(Eq(1 / q, 1), y > 0)), (1, True)), (y, 1, 2)), x**2, 2)
        else:
            value = Subs(Piecewise((2, Eq(1 / q, 1 / q, evaluate=False)), (1, True), evaluate=False), x**2, 2)

    context = VerificationContext.for_model(Curve).observe(q=1)
    result = context.run(Method("inspect", lambda current: current.expression("value"), uses=("value",)))
    assert result.data.doit() == (4 if ranged else 2)
    assert all(item.verdict == "strictly true" for item in result.report.evidence)


def test_compound_selector_truth_cannot_erase_an_affected_operand_domain():
    class Curve(VerificationModel):
        p = InstanceDummy("p", real=True)
        value = Subs(
            Piecewise((2, Eq(log(p**2 - 1), log(p**2 - 1), evaluate=False)), (1, True), evaluate=False),
            p**2,
            1,
        )

    context = VerificationContext.for_model(Curve)
    calls = []
    with pytest.raises(VerificationError):
        context.run(Method("inspect", lambda current: calls.append(current), uses=("value",)))
    assert not calls


@pytest.mark.parametrize("operator", (Sum, Integral))
def test_unrelated_compound_patterns_preserve_native_range_metadata(operator):
    class Curve(VerificationModel):
        x = Coordinate("x", real=True)
        y = Coordinate("y", real=True)
        value = Subs(operator(1 / y, (y, 1, 2)), x**2, 2)

    context = VerificationContext.for_model(Curve)
    result = context.run(Method("value", lambda current: current.expression("value").doit(), uses=("value",)))
    assert result.data == (S(3) / 2 if operator is Sum else log(2))
    assert all(item.verdict == "strictly true" for item in result.report.evidence)

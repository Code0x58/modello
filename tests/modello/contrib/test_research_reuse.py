"""Reusing a case preserves identities through component changes."""

import pytest
from sympy import Derivative, Eq, Integral, Limit, Product, S, Subs, Sum

from modello import Coordinate, InstanceDummy, Modello, Vector, component
from modello.contrib.research import ResearchCase, exact_form
from modello.errors import ModelDefinitionError


@pytest.fixture
def curves():
    class Old(Modello):
        x = Coordinate("x", real=True)
        value = x**2

    class New(Modello):
        x = Coordinate("z", real=True)
        value = -(x**2)

    return Old, New


@pytest.mark.parametrize(
    "operation, expected",
    (
        (lambda value, x: Derivative(value, x), lambda x: -2 * x),
        (lambda value, x: Integral(value, (x, 0, 1)), lambda x: -S.One / 3),
        (lambda value, x: Sum(value, (x, 1, 2)), lambda x: -5),
        (lambda value, x: Product(value, (x, 1, 2)), lambda x: 4),
        (lambda value, x: Limit(value, x, 1), lambda x: -1),
        (lambda value, x: Subs(value, x, 2), lambda x: -4),
        (lambda value, x: Subs(value, x**2, 1), lambda x: -1),
    ),
    ids=("derivative", "integral", "sum", "product", "limit", "substitution", "compound_substitution"),
)
def test_replacement_rebinds_coordinates_used_only_inside_native_operators(curves, operation, expected):
    old, new = curves

    class Parent(Modello):
        curve = component(old)
        value = operation(curve.value, old.x)

    source = ResearchCase.for_model(Parent)
    changed = source.replace("curve", new)
    expression = changed.expression("value")
    assert not expression.has(old.x)
    assert expression.doit() == expected(new.x)
    assert source.expression("value").has(old.x)


def test_replacement_rebinds_native_observations_and_requirement_payloads(curves):
    old, new = curves

    class Parent(Modello):
        curve = component(old)
        observed = InstanceDummy("observed", real=True)

    source = ResearchCase.for_model(Parent)
    slope = Derivative(source.symbols.curve.value, old.x)
    claim = exact_form(slope, 0, name="stationary")
    changed = source.observe(observed=slope).require(claim).replace("curve", new)
    assert changed.expression("observed").doit() == -2 * new.x
    assert changed.requirements[0].subject == Derivative(changed.symbols.curve.value, new.x)
    evidence = changed.check().explain("stationary")
    assert evidence.expression.doit() == -2 * new.x
    assert evidence.verdict == "conditionally true"
    expected = exact_form("observed", slope, name="observed_slope")
    changed = source.observe(observed=slope).require(expected).replace("curve", new)
    assert changed.check().explain("observed_slope").verdict == "strictly true"


def test_replacement_rebinds_native_coordinates_in_vector_definitions(curves):
    old, new = curves

    class Parent(Modello):
        curve = component(old)
        values = Vector((Derivative(curve.value, old.x),))

    changed = ResearchCase.for_model(Parent).replace("curve", new)
    assert changed.expression("values.0").doit() == -2 * new.x


@pytest.mark.parametrize("where", ("declaration", "assumption", "requirement", "expected", "hint"))
def test_replacement_rejects_ambiguous_shared_native_coordinates(curves, where):
    old, new = curves

    class Parent(Modello):
        a = component(old)
        b = component(old)
        if where == "declaration":
            slope = Derivative(a.value, old.x)

    case = ResearchCase.for_model(Parent)
    shared = Derivative(case.symbols.b.value, old.x)
    if where == "assumption":
        case = case.assume(Eq(shared, 0))
    elif where == "requirement":
        case = case.require(exact_form(case.symbols.a.value + shared, 0, name="joined"))
    elif where == "expected":
        case = case.require(exact_form("a.value", shared, name="joined"))
    elif where == "hint":
        case = case.with_hint("bound", upper=case.symbols.a.value + shared)
    with pytest.raises(ModelDefinitionError, match="shared|joins"):
        case.replace("a", new)


@pytest.mark.parametrize("where", ("requirement", "hint"))
def test_replacement_rejects_ambiguous_expanded_native_payloads(curves, where):
    old, new = curves

    class Parent(Modello):
        a = component(old)
        b = component(old)

    source = ResearchCase.for_model(Parent)
    expanded = Derivative(source.expression("b.value"), old.x)
    if where == "requirement":
        source = source.require(exact_form(expanded, 0))
    else:
        source = source.with_hint("bound", upper=expanded)
    with pytest.raises(ModelDefinitionError, match="joins"):
        source.replace("a", new)


def test_replacement_preserves_a_named_unchanged_siblings_hint_scope(curves):
    old, new = curves

    class Parent(Modello):
        a = component(old)
        b = component(old)

    source = ResearchCase.for_model(Parent)
    hint = Derivative(source.symbols.b.value, old.x)
    changed = source.with_hint("bound", upper=hint).replace("a", new)
    assert dict(changed.hints)["bound"] == Derivative(changed.symbols.b.value, old.x)


@pytest.mark.parametrize("observed", (False, True))
def test_replacement_rejects_splitting_aliased_fields(observed):
    class Old(Modello):
        a = InstanceDummy("a", real=True)
        b = a

    class New(Modello):
        a = InstanceDummy("a", real=True)
        b = InstanceDummy("b", real=True)

    class Parent(Modello):
        curve = component(Old)

    source = ResearchCase.for_model(Parent)
    if observed:
        source = source.observe(curve=Old("old", a=2))
    with pytest.raises(ModelDefinitionError, match="splits aliased fields"):
        source.replace("curve", New)
    assert source.problem.model is Parent


def test_replacement_preserves_aliases_and_their_observations():
    class Old(Modello):
        a = InstanceDummy("a", real=True)
        b = a

    class New(Modello):
        a = InstanceDummy("a", real=True)
        b = a

    class Parent(Modello):
        curve = component(Old)

    source = ResearchCase.for_model(Parent).observe(curve=Old("old", a=2))
    changed = source.replace("curve", New)
    assert changed.symbols.curve.a == changed.symbols.curve.b
    assert changed.expression("curve.a") == changed.expression("curve.b") == 2

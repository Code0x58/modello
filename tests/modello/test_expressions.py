"""Scoped native operators agree across construction, families, and audits."""

from types import new_class

import pytest
from sympy import (
    Add,
    Derivative,
    Eq,
    FiniteSet,
    Function,
    Integral,
    Limit,
    Product,
    Rational,
    S,
    Sum,
    Symbol,
    exp,
    fibonacci,
)

from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model
from modello._expressions import _evaluate
from modello.errors import ModelInputError


@pytest.mark.parametrize("first,last", [(0, 3), (-4, -1), (2, 1), (2, 0), (3, 0)])
def test_finite_vector_reductions_retain_native_empty_negative_and_reversed_limit_semantics(first, last):
    values = Vector(10, 20, 30, 40)
    i = Index("i")
    for operator in (Sum, Product):
        expression = operator(values[i], (i, first, last))
        assert _evaluate(expression) == expression.doit()


def test_nested_finite_reductions_keep_dependent_bounds_and_product_order():
    i, j = Index("i"), Index("j")
    values = Vector(10, 20, 30, 40)
    expression = Sum(values[i] * values[j], (i, 0, j), (j, 0, 3))
    assert _evaluate(expression) == 6500
    a, b = Symbol("a", commutative=False), Symbol("b", commutative=False)
    ordered = Vector(a, b)
    assert _evaluate(ordered.product(i)) == a * b
    assert _evaluate(Product(ordered[i], (i, 2, -1))) == (a * b) ** -1
    periodic = Vector(1, 2, 3)
    assert _evaluate(Sum(periodic[i % 3], (i, 0, 299))) == 600


def test_undefined_vector_selections_end_reduction_and_ordinary_symbolic_sums_stay_closed_form():
    i = Index("i")
    values = Vector(1, 2, 3)
    assert _evaluate(Sum(values[i], (i, 0, 10**100))) is S.NaN
    assert _evaluate(Product(values[i], (i, 0, 10**100))) is S.NaN
    count = 1_000_000
    assert _evaluate(Sum(i**2, (i, 0, count))) == count * (count + 1) * (2 * count + 1) // 6


@pytest.mark.parametrize("indirect", [False, True])
def test_calculus_rejects_indirect_cyclic_dependencies(indirect):
    class Cyclic(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        y = y + x
        shifted = y + 1
        slope = Derivative(shifted if indirect else y, x)

    with pytest.raises(ModelInputError, match="cyclic"):
        inspect_model(Cyclic, "cycle").solutions()
    with pytest.raises(ModelInputError, match="cyclic"):
        Cyclic("cycle", x=0)


def test_calculus_does_not_choose_arbitrarily_between_joint_definitions():
    class Square(Modello):
        x = InstanceDummy("x", positive=True)
        value = x**2

    class Joined(Modello):
        square = component(Square)
        alias = square.value
        alias = square.x + 1

    case = Joined("joint")
    assert case.square.x == (1 + S(5) ** Rational(1, 2)) / 2

    class Differentiated(Joined):
        shifted = Joined.alias + 5
        slope = Derivative(shifted, Joined.square.x)

    with pytest.raises(ModelInputError, match="multiply defined"):
        inspect_model(Differentiated, "ambiguous").solutions()
    with pytest.raises(ModelInputError, match="multiply defined"):
        Differentiated("ambiguous")

    class LocalCoordinate(Joined):
        slope = Derivative(Joined.alias**2, Joined.alias)

    local = LocalCoordinate("local")
    assert (local.slope - 2 * local.alias).simplify() == 0

    class SameDefinition(Modello):
        square = component(Square)
        alias = square.value
        alias = square.x**2
        slope = Derivative(alias, square.x)

    assert SameDefinition("same", square=Square("square", x=3)).slope == 6


def test_family_queries_and_conditions_differentiate_model_formulas_before_observations():
    class Sensitivity(Modello):
        x = InstanceDummy("x", real=True)
        square = x**2

    problem = inspect_model(Sensitivity, "formula")
    symbols = problem.symbols
    family = problem.solutions()
    gradient = Derivative(symbols.square, symbols.x)
    assert family.expression(gradient) == 2 * symbols.x
    assert family.where(Eq(gradient, 6)).one().x == 3
    observed = family.observe(x=3)
    assert observed.expression(gradient) == 6
    assert observed.values(gradient) == FiniteSet(6)
    reciprocal = 1 / gradient
    assert observed.expression(reciprocal) == Rational(1, 6)
    assert observed.values(reciprocal) == FiniteSet(Rational(1, 6))
    assert observed.where(Eq(reciprocal, Rational(1, 6))).one().x == 3
    assert family.observe(x=0).expression(reciprocal) is S.NaN
    preferred = problem.solutions(symbols.square)
    assert preferred.expression(symbols.square + gradient) == symbols.square + 2 * symbols.x


def test_reduced_response_derivative_follows_a_constrained_vector_family():
    class Allocation(Modello):
        weights = Vector("w", size=2, nonnegative=True)
        total = weights.sum()
        concentration = weights.dot(weights)

    problem = inspect_model(Allocation, "allocation", total=1)
    family = problem.solutions(problem.symbols.weights)
    coordinate = family.parameters[0].symbol
    declared = Derivative(family.symbols.concentration, coordinate)
    response = family.expression(family.symbols.concentration)
    constrained = Derivative(response, coordinate)

    assert family.expression(declared) == 2 * coordinate
    assert family.expression(constrained) == 4 * coordinate - 2
    balanced = family.where(Eq(constrained, 0))
    assert balanced.one().weights == Vector(Rational(1, 2), Rational(1, 2))
    assert balanced.values(family.symbols.concentration) == FiniteSet(Rational(1, 2))


def test_differentiation_expands_definitions_before_substituting_observations():
    class Sensitivity(Modello):
        offset = InstanceDummy("offset", real=True)
        x = offset + 1
        squared = x**2
        slope = Derivative(squared, x)

    p = inspect_model(Sensitivity, "derivative", offset=2)
    assert p.constraints[-2].right.has(Derivative)
    assert Sensitivity("point", offset=2).slope == 6
    assert p.solutions().one().slope == 6
    assert p.analyse().status == "consistent"


def test_calculus_definitions_are_retained_and_solved_in_both_directions():
    t = Symbol("t", real=True)

    class Exposure(Modello):
        rate = InstanceDummy("rate", real=True)
        horizon = InstanceDummy("horizon", positive=True)
        cumulative = Integral(rate * t, (t, 0, horizon))

    p = inspect_model(Exposure, "formula")
    assert isinstance(p.constraints[0].right, Integral)
    assert Exposure("forward", rate=2, horizon=3).cumulative == 9
    assert Exposure("inverse", horizon=3, cumulative=9).rate == 2
    assert inspect_model(Exposure, "family", horizon=3, cumulative=9).solutions().one().rate == 2


def test_limits_apply_observed_endpoints_before_evaluation():
    x = Symbol("x", real=True)

    class Response(Modello):
        point = InstanceDummy("point", real=True)
        scale = InstanceDummy("scale", real=True)
        response = Limit(scale * (exp(x) - 1) / x, x, point)

    problem = inspect_model(Response, "removable", point=0, scale=3)
    assert isinstance(problem.constraints[0].right, Limit)
    assert Response("point", point=0, scale=3).response == 3
    assert problem.solutions().one().response == 3
    assert problem.analyse().status == "consistent"
    assert inspect_model(Response, "conflict", point=0, scale=3, response=4).analyse().status == "inconsistent"


@pytest.mark.parametrize("coefficient,expected", [(0, S.One), (2, Rational(1, 2))])
def test_parameterized_limits_retain_exceptional_coefficient_values(coefficient, expected):
    x = Symbol("x", real=True)

    class Response(Modello):
        coefficient = InstanceDummy("coefficient", real=True)
        shifted = coefficient + 1
        response = Limit((x + shifted) / (x + 2 * shifted), x, 0)

    observed = coefficient - 1
    problem = inspect_model(Response, "exception", coefficient=observed)
    assert Response("point", coefficient=observed).response == expected
    assert problem.solutions().one().response == expected
    assert problem.analyse().status == "consistent"
    assert Response("open").response.has(Limit)
    with pytest.raises(ModelInputError, match="bound-variable"):
        inspect_model(Response, "unresolved").solutions()


def test_limits_defer_enclosing_calculus_without_blocking_independent_siblings():
    x, a = Symbol("x", real=True), Symbol("a", real=True)
    deferred = Limit((x + a) / (x + 2 * a), x, 0)
    derivative = Derivative(deferred, a)
    assert _evaluate(derivative) == derivative
    assert _evaluate(Add(deferred, Derivative(x**2, x))) == deferred + 2 * x
    assert _evaluate(Sum(deferred, (x, 0, 1))) == Sum(deferred, (x, 0, 1))


def test_family_limit_queries_and_conditions_resolve_observations():
    x = Symbol("x", real=True)

    class Point(Modello):
        endpoint = InstanceDummy("endpoint", real=True)

    family = inspect_model(Point, "point").solutions()
    query = Limit((exp(x) - 1) / x, x, family.symbols.endpoint)
    with pytest.raises(ModelInputError, match="bound-variable"):
        family.expression(query)
    assert family.observe(endpoint=0).expression(query) == 1
    assert family.where(Eq(family.symbols.endpoint, 0), Eq(query, 1)).one().endpoint == 0
    assert family.where(Eq(family.symbols.endpoint, 0), Eq(query, 2)).is_empty is True
    t = Symbol("t", real=True)
    nested = Limit(Limit(x + t, t, 0), x, family.symbols.endpoint)
    assert family.observe(endpoint=0).expression(nested) == 0
    with pytest.raises(ModelInputError, match="commutative scalar"):
        family.expression(Limit(Symbol("operator", commutative=False) / x, x, 0))


def test_shared_formula_dependencies_do_not_require_expanding_every_path():
    def declare(namespace):
        namespace["x"] = InstanceDummy("x", real=True)
        namespace["f0"] = namespace["x"]
        namespace["f1"] = namespace["x"]
        for i in range(2, 41):
            namespace[f"f{i}"] = namespace[f"f{i - 1}"] + namespace[f"f{i - 2}"]

    Fibonacci = new_class("Fibonacci", (Modello,), exec_body=declare)
    case = Fibonacci("shared", x=1)
    assert case.f40 == fibonacci(41)


def test_calculus_through_a_cycle_reports_an_explicit_limitation():
    class Cyclic(Modello):
        x = InstanceDummy("x", real=True)
        expression = InstanceDummy("expression", real=True)
        expression = expression + x
        slope = Derivative(expression, x)

    problem = inspect_model(Cyclic, "cycle")
    assert problem.constraints[-1].right.has(Derivative)
    with pytest.raises(ModelInputError, match="cyclic"):
        Cyclic("cycle", x=0)
    with pytest.raises(ModelInputError, match="cyclic"):
        problem.solutions()


def test_unresolved_derivatives_keep_diagnostic_observations_without_claiming_a_solution():
    class Opaque(Modello):
        x = InstanceDummy("x", real=True)
        gradient = Derivative(Function("f")(x), x)

    problem = inspect_model(Opaque, "opaque", x=3)
    analysis = problem.analyse()
    assert analysis.status == "unknown"
    assert analysis.deductions.candidate.value(problem.field("x")) == 3
    assert "Diagnostic check raised" not in analysis.reason
    assert Opaque("point", x=3).gradient.has(Derivative)
    with pytest.raises(ModelInputError, match="bound-variable"):
        problem.solutions()


def test_local_formula_variables_are_not_printed_as_external_inputs():
    t = Symbol("t", real=True)

    class Area(Modello):
        height = InstanceDummy("height", real=True)
        integral = Integral(t, (t, 0, height))

    p = inspect_model(Area, "local")
    assert p.format(p.constraints[0].right) == "Integral(t, (t, 0, height))"
    assert "external[t]" in p.format(t + p.constraints[0].right)
    assert "Integral(t, (t" in p.format(t + p.constraints[0].right)

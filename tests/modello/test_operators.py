"""Scoped native operators agree across construction, families, and audits."""

from types import new_class

import pytest
from sympy import Derivative, Eq, FiniteSet, Function, Integral, Product, Rational, S, Sum, Symbol, fibonacci

from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model
from modello._expressions import _evaluate
from modello.errors import ModelInputError, SolverBackendError


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
    with pytest.raises(SolverBackendError) as caught:
        Cyclic("cycle", x=0)
    assert isinstance(caught.value.backend_error, NotImplementedError)


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
    with pytest.raises(SolverBackendError) as caught:
        Differentiated("ambiguous")
    assert isinstance(caught.value.backend_error, NotImplementedError)

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
    assert family.observe(x=3).expression(gradient) == 6
    assert family.observe(x=3).values(gradient) == FiniteSet(6)
    preferred = problem.solutions(symbols.square)
    assert preferred.expression(symbols.square + gradient) == symbols.square + 2 * symbols.x


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
    with pytest.raises(SolverBackendError) as caught:
        Cyclic("cycle", x=0)
    assert isinstance(caught.value.backend_error, NotImplementedError)
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

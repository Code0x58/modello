"""Indexed formulas share scalar identities, domains, and joint model solving."""

from copy import deepcopy
from types import new_class

import pytest
from sympy import (
    Derivative,
    Eq,
    FiniteSet,
    Function,
    Integral,
    Interval,
    MatrixSymbol,
    Product,
    Rational,
    S,
    Sum,
    Symbol,
    Tuple,
    fibonacci,
)

from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model, vector
from modello.errors import ModelDefinitionError, ModelInputError, SolutionCountError, SolverBackendError


class Window(Modello):
    samples = Vector("x", size=3, real=True)
    i = Index("i")
    total = Sum(samples[i], (i, 0, len(samples) - 1))
    growth = Product(1 + samples[i], (i, 0, len(samples) - 1))
    changes = vector(samples[i] - samples[i - 1], (i, 1, len(samples) - 1))


def test_vector_values_indexing_and_sympy_reconstruction():
    x = Vector("x", size=3, real=True)
    i = Index("i")
    assert len(x) == 3
    assert x.shape == (3,)
    assert Vector().shape == (0,)
    assert tuple(x[::-1]) == (x[2], x[1], x[0])
    assert x[-1] == x[2]
    assert x[i].subs(i, 1) == x[1]
    assert x[i].free_symbols == set(x) | {i}
    assert x.sum(i).free_symbols == set(x)
    assert x.sum(i).xreplace({x[1]: S(5)}).doit() == x[0] + 5 + x[2]
    assert deepcopy(x) == x
    assert deepcopy(i) == i
    assert Vector([1, "1/2", 3]) == Vector(1, Rational(1, 2), 3)
    with pytest.raises(IndexError):
        x[3]
    assert x[i].subs(i, 3) is S.NaN
    assert x[i].subs(i, Rational(1, 2)) is S.NaN
    assert x[i].is_real is None


def test_forward_inverse_and_partial_vector_observations():
    forward = Window("forward", samples=[1, 2, 3])
    assert forward.samples == Vector(1, 2, 3)
    assert (forward.total, forward.growth) == (6, 24)
    assert forward.changes == Vector(1, 1)
    inverse = Window("inverse", samples=[1, None, 3], total=6)
    assert inverse.samples == forward.samples
    assert inverse.changes == forward.changes
    symbolic = Window("open", samples=[1, None, 3])
    assert symbolic.total == symbolic.samples[1] + 4
    assert symbolic.samples[1] not in Window("other").samples.free_symbols


def test_inspection_preserves_formulas_and_whole_vector_input_groups(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Inspection must not evaluate or solve formulas")

    monkeypatch.setattr(Sum, "doit", unexpected)
    p = inspect_model(Window, "raw", samples=[1, None, 3])
    assert isinstance(p.constraints[0].right, Sum)
    assert isinstance(p.constraints[1].right, Product)
    assert p.symbols.samples[1] == p.field("samples", "1").symbol
    assert p.symbols.samples[1] == p.field("samples", 1).symbol
    assert p.vector("samples") == p.symbols.samples
    assert [c.path for c in p.constraints if c.input == "samples"] == [("samples", "0"), ("samples", "2")]
    assert not any(c.input == "samples" for c in p.without("samples").constraints)
    assert "Sum(samples[i]" in str(p)


def test_vector_schema_is_read_only_and_retains_nested_aliases_and_empty_fields():
    class Parent(Modello):
        window = component(Window)
        alias = window.samples
        empty = Vector("empty", size=0)

    p = inspect_model(Parent, "schema")
    assert p.vector("window", "samples") == p.vector("alias")
    assert len(p.vector("empty")) == 0
    assert set(p.vectors) == {("window", "samples"), ("window", "changes"), ("alias",), ("empty",)}
    with pytest.raises(TypeError):
        p.vectors[("alias",)] = Vector(1, 2, 3)


def test_component_refinement_cannot_silently_change_an_inherited_vector_shape():
    class Wider(Modello):
        samples = Vector("x", size=4)

    class Parent(Modello):
        window = component(Window)
        total = window.samples.sum()

    with pytest.raises(ModelDefinitionError, match=r"window\.samples"):

        class Invalid(Parent):
            window = component(Wider)


def test_vector_aliases_and_nested_component_copies():
    class Pair(Modello):
        left = component(Window)
        right = component(Window)
        alias = left.samples
        difference = left.total - right.total

    source = Window("source", samples=[1, None, 3])
    pair = Pair("joined", left=source, right=source, alias=[None, 2, None], difference=1)
    assert pair.left.samples == Vector(1, 2, 3)
    assert pair.right.samples == Vector(1, 1, 3)
    assert pair.alias == pair.left.samples
    assert pair.left is not source
    assert source.total == source.samples[1] + 4
    with pytest.raises(SolutionCountError):
        Pair("conflict", left=Window("fixed", samples=[1, 2, 3]), alias=[1, 4, 3])


def test_vector_overrides_rebind_inherited_formulas_and_aliases():
    class PositiveWindow(Window):
        samples = Vector("x", size=3, positive=True)

    class Envelope(Modello):
        window = component(Window)
        squared = window.samples[1] ** 2

    class PositiveEnvelope(Envelope):
        window = component(PositiveWindow)

    result = PositiveEnvelope("positive", squared=4)
    assert result.window.samples[1] == 2
    assert result.window.total == sum(result.window.samples)
    with pytest.raises(SolutionCountError):
        PositiveWindow("bad", samples=[1, -1, 2])
    with pytest.raises(ModelDefinitionError, match="size"):

        class WrongShape(Window):
            samples = Vector("x", size=4)


def test_exact_family_and_diagnostics_use_the_same_indexed_equations():
    p = inspect_model(Window, "exact", samples=[1, None, 3], total=6)
    assert p.analyse().status == "consistent"
    assert p.solutions().one().samples == Vector(1, 2, 3)
    conflict = inspect_model(Window, "conflict", samples=[1, 2, 3], total=7)
    analysis = conflict.analyse(conflicts=True)
    assert analysis.status == "inconsistent"
    assert analysis.conflict is not None
    assert any(c.path == ("total",) for c in analysis.conflict.constraints)


def test_vector_arithmetic_and_contractions_have_explicit_shapes():
    x = Vector(1, 2, 3)
    y = Vector(3, 2, 1)
    assert tuple(x + y) == (4, 4, 4)
    assert tuple(x - y) == (-2, 0, 2)
    assert tuple(-x) == (-1, -2, -3)
    assert tuple(2 * x) == (2, 4, 6)
    assert tuple(x / 2) == (Rational(1, 2), 1, Rational(3, 2))
    assert (x @ y).doit() == 10
    assert x.sum().doit() == 6
    assert x.product().doit() == 6
    assert Vector().sum().doit() == 0
    assert Vector().product().doit() == 1
    assert (Vector() @ Vector()).doit() == 0
    with pytest.raises(ModelInputError, match="same size"):
        x + Vector(1, 2)
    with pytest.raises(ModelInputError, match="same size"):
        x @ Vector(1)


def test_noncommutative_entries_keep_their_product_order():
    a, b = Symbol("a", commutative=False), Symbol("b", commutative=False)
    values = Vector(a, b)
    assert values[Index("i")].is_commutative is False
    assert values.product().doit() == a * b
    assert values.dot(Vector(b, a)).doit() == a * b + b * a
    c = Symbol("c", commutative=False)
    assert (c * values).doit() == Vector(c * a, c * b)
    assert (values * c).doit() == Vector(a * c, b * c)


def test_index_substitution_also_updates_coordinates_when_they_depend_on_it():
    i = Index("i")
    values = Vector(i, i**2)
    assert values[i].subs(i, 1) == 1
    assert values.sum(i).doit() == 1


def test_comprehension_retains_a_finite_domain_and_zero_based_result():
    i = Index("i")
    result = vector(i**2, (i, 2, 4))
    assert len(result) == 3
    assert result == result.func(*result.args)
    assert result[0] == 4
    assert result[-1] == 16
    assert tuple(result) == (4, 9, 16)
    assert result.free_symbols == set()
    assert result.sum().doit() == 29
    assert result.doit() == Vector(4, 9, 16)


def test_comprehension_queries_are_lazy_and_respect_their_local_index():
    i = Index("i")
    coefficient = Symbol("a")
    result = vector(coefficient * i**2, (i, 0, 1_000_000))
    assert result.shape == (1_000_001,)
    assert result[0] == 0
    assert result[-1] == coefficient * 1_000_000**2
    assert result[:2] == Vector(0, coefficient)
    assert result.subs(i, 2) == result
    assert result.subs(coefficient, 3)[2] == 12
    with pytest.raises(IndexError):
        result[1_000_001]


def test_automatic_indices_avoid_names_already_used_in_the_vector():
    i = Index("i")
    values = vector(i**2, (i, 0, 2))
    assert values.sum().variables[0].name == "j"
    assert values.sum().doit() == 5


def test_matrix_entries_and_scaling_are_rejected_at_the_vector_boundary():
    matrix = MatrixSymbol("A", 2, 2)
    with pytest.raises(ModelInputError, match="scalar"):
        Vector(matrix)
    with pytest.raises(ModelInputError, match="scalar"):
        Vector(1, 2) * matrix
    with pytest.raises(ModelInputError):
        vector(matrix, (Index("i"), 0, 1))


def test_vector_family_queries_preserve_order_and_coupled_values():
    class Allocation(Modello):
        weights = Vector("w", size=2, nonnegative=True)
        total = weights.sum()

    p = inspect_model(Allocation, "family", total=1)
    family = p.solutions(p.symbols.weights)
    weights = family.expression(family.symbols.weights)
    assert isinstance(weights, Vector)
    assert sum(weights) == 1
    observed = family.observe(weights=[Rational(1, 4), None])
    assert observed.values(family.symbols.weights) == FiniteSet(Tuple(Rational(1, 4), Rational(3, 4)))
    assert observed.one().weights == Vector(Rational(1, 4), Rational(3, 4))


def test_large_vector_observations_close_an_exact_family_even_when_preferred():
    class Large(Modello):
        samples = Vector("x", size=1024, real=True)
        total = samples.sum()

    problem = inspect_model(Large, "large", samples=[1] * 1024)
    family = problem.solutions(problem.symbols.samples)
    assert family.parameters == ()
    assert family.expression(family.symbols.total) == 1024
    assert family.expression(family.symbols.samples) == Vector([1] * 1024)


@pytest.mark.parametrize("count, total, growth", [(0, 0, 1), (1, 1, 2), (3, 6, 24), (4, 10, 120)])
def test_observed_bounds_evaluate_before_exact_family_validation(count, total, growth):
    class Prefix(Modello):
        samples = Vector("x", size=4, real=True)
        count = InstanceDummy("count", integer=True, nonnegative=True)
        i = Index("i")
        total = Sum(samples[i], (i, 0, count - 1))
        growth = Product(1 + samples[i], (i, 0, count - 1))

    family = inspect_model(Prefix, "prefix", samples=[1, 2, 3, 4], count=count).solutions()
    case = family.one()
    assert (case.total, case.growth) == (total, growth)
    symbols = family.symbols
    query = Sum(symbols.samples[Prefix.i], (Prefix.i, 0, symbols.count - 1))
    assert family.expression(query) == total
    assert family.values(query) == FiniteSet(total)


def test_free_vector_indices_are_explicitly_unsupported_by_exact_families():
    class Selection(Modello):
        samples = Vector("x", size=2, real=True)
        index = InstanceDummy("index", integer=True)
        selected = samples[index]

    problem = inspect_model(Selection, "selection", samples=[10, 20])
    with pytest.raises(ModelInputError, match="indices"):
        problem.solutions()
    assert inspect_model(Selection, "valid", samples=[10, 20], index=-1).solutions().one().selected == 20
    for index in (-3, 2):
        invalid = inspect_model(Selection, "invalid", samples=[10, 20], index=index).solutions()
        assert invalid.is_empty is True
        assert invalid.values(invalid.symbols.selected) is S.EmptySet


def test_family_sections_and_plot_windows_accept_vector_coordinates():
    from examples.solution_views import ModelView

    class Pair(Modello):
        samples = Vector("x", size=2, real=True)
        total = samples.sum()

    problem = inspect_model(Pair, "section")
    family = problem.solutions(problem.symbols.samples)
    section = family.section(problem.symbols.samples[0], through=Pair("anchor", samples=[2, 3]))
    assert section.expression(problem.symbols.samples) == Vector(problem.symbols.samples[0], 3)
    assert section.expression(problem.symbols.total) == problem.symbols.samples[0] + 3
    assert "Coordinates: samples[0]" in str(section)
    view = ModelView(section.parameters, section)
    assert view._windows(Pair("lower", samples=[0, 3]), Pair("upper", samples=[4, 3])) == (Interval(0, 4),)
    reference = Pair("missing", samples=[2, 3])
    reference.samples = Vector(2)
    with pytest.raises(ModelInputError, match=r"samples\[1\]"):
        family.section(through=reference)


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


@pytest.mark.parametrize("replacement", [Vector(1, 2), Vector(1, 2, 3, 4), "1,2,3", None])
def test_component_copies_reject_overwritten_vector_shapes(replacement):
    class Parent(Modello):
        window = component(Window)

    source = Window("source", samples=[1, 2, 3])
    source.samples = replacement
    with pytest.raises(ModelInputError, match=r"window\.samples"):
        Parent("copy", window=source)


@pytest.mark.parametrize("raw", [1, "1, 2, 3", {}, [1, 2], [1, 2, 3, 4], [1, object(), 3]])
def test_invalid_vector_observations_report_owned_input_errors(raw):
    with pytest.raises(ModelInputError, match="samples"):
        Window("invalid", samples=raw)


@pytest.mark.parametrize("size", [-1, True, 1.5, "3"])
def test_invalid_shapes_fail_at_declaration(size):
    with pytest.raises(ModelInputError, match="size"):
        Vector("x", size=size)


@pytest.mark.parametrize(
    "position", [True, Rational(1, 2), 1.0, 1.5, S.NaN, S.Infinity, MatrixSymbol("A", 2, 2), object()]
)
def test_invalid_coordinate_indices_are_rejected(position):
    with pytest.raises(ModelInputError, match="index"):
        Vector(1, 2)[position]

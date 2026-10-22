"""Public construction jointly solves and selects admissible candidates."""

import pytest
from sympy import S, Symbol, simplify

from modello import (
    BoundInstanceDummy,
    InstanceDummy,
    Modello,
    ModelSolveError,
    SolutionCountError,
    _select_candidate,
    component,
)

from ._models import Pair, PairParent, Reading


def test_no_constraints():
    """A model with no constraints just has dummy attributes."""

    class ExampleClass(Modello):
        thing = InstanceDummy("thing")

    instance = ExampleClass("Example")
    assert isinstance(instance.thing, BoundInstanceDummy)

    instance = ExampleClass("Example", thing=1)
    expected = simplify(1)
    assert isinstance(instance.thing, type(expected))
    assert instance.thing == expected


@pytest.mark.parametrize(
    "values, retained, expected",
    [
        pytest.param((), 0, None, id="no-reported-candidates"),
        pytest.param((S.One,), 1, S.One, id="one-positive-value"),
        pytest.param((Symbol("unknown"),), 1, Symbol("unknown"), id="undecided-assumption"),
        pytest.param((S.Zero,), 0, None, id="zero-outside-positive-domain"),
        pytest.param((S.NegativeOne,), 0, None, id="negative-outside-positive-domain"),
        pytest.param((S.NaN,), 0, None, id="undefined-value"),
        pytest.param((S.ComplexInfinity,), 0, None, id="pole"),
        pytest.param((S.One, S(2)), 2, None, id="multiple-admissible-values"),
        pytest.param((S.NaN, S.One), 1, S.One, id="valid-value-alongside-pole"),
        pytest.param((S.NegativeOne, S.Zero), 0, None, id="reported-values-all-rejected"),
    ],
)
def test_constructor_candidate_selection(positive_problem, values, retained, expected):
    """Constructor filtering preserves symbolic values and original failure counts."""
    symbol = positive_problem.symbols.x
    reported = tuple({symbol: value} for value in values)
    if retained == 1:
        selected = _select_candidate(positive_problem, reported)
        assert selected[symbol] == expected
        assert any(selected is candidate for candidate in reported)
    else:
        with pytest.raises(SolutionCountError) as caught:
            _select_candidate(positive_problem, reported)
        assert caught.value.solution_count == retained
        assert caught.value.returned_count == len(values)
        assert all(
            tuple(item.field for item in candidate.values) == positive_problem.fields
            for candidate in caught.value.problem.candidates
        )


def test_missing_nested_inputs_remain_shared_parameters():
    """Underdetermined trees retain free inputs rather than choosing zeroes."""

    first = PairParent("P", child=Pair("C", a=2))
    second = PairParent("P", child=Pair("C", a=2))
    assert isinstance(first.child.b, BoundInstanceDummy)
    assert first.child.b != second.child.b
    assert first.child.total == first.child.b + 2
    assert first.total == first.child.total + 1


def test_external_symbols_are_parameters_not_solver_unknowns():
    """Solving a model must not assign values to a caller's external parameters."""

    class Child(Modello):
        x = InstanceDummy("x")
        doubled = x * 2

    class Parent(Modello):
        child = Child
        total = child.doubled + 1

    parameter = Symbol("parameter")
    parent = Parent("P", total=parameter)
    assert parent.child.x == (parameter - 1) / 2
    assert parent.total == parameter


def test_inconsistent_and_ambiguous_nested_systems_are_rejected():
    """Nested constraints retain assumptions and the unique-solution requirement."""

    class Parent(Modello):
        child = Reading
        total = child.squared + 1

    with pytest.raises(ModelSolveError, match="2 solutions") as error:
        Parent("P", total=5)
    assert "Parent('P')" in str(error.value)
    assert "child: Reading\n    c1: squared = x**2" in str(error.value.problem)
    assert "total = 5" in str(error.value.problem)
    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Parent("P", child=Reading("C", x=2), total=10)
    assert "x = 2" in str(error.value.problem)
    assert "total = 10" in str(error.value.problem)


def test_derived_field_assumptions_filter_multiple_roots():
    """Domain assumptions still select solutions after formula expansion."""

    class Child(Modello):
        positive = InstanceDummy("positive", positive=True)
        x = InstanceDummy("x", real=True)
        squared = x**2
        positive = x + 2

    class Parent(Modello):
        child = component(Child)
        squared = child.squared

    parent = Parent("P", squared=9)
    assert (parent.child.x, parent.child.positive) == (3, 5)


def test_undefined_derived_values_are_not_solutions():
    """Substituting a value at a pole must not create a successful model."""

    class Child(Modello):
        x = InstanceDummy("x")
        inverse = 1 / x

    class Parent(Modello):
        child = component(Child)
        x = child.x

    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Parent("P", x=0)
    assert "1/x" in str(error.value.problem)
    assert "x = 0" in str(error.value.problem)


def test_cyclic_definitions_are_solved_as_simultaneous_equations():
    """Cycles are handled by the joint solver instead of recursive expansion."""

    class Child(Modello):
        x = InstanceDummy("x")
        y = x + 1
        x = y / 2

    class Parent(Modello):
        child = Child
        total = child.x + child.y

    parent = Parent("P")
    assert (parent.child.x, parent.child.y, parent.total) == (1, 2, 3)


def test_constants_and_positive_inputs_cannot_be_contradicted():
    """Fixed derived values and independent input assumptions remain constraints."""

    class Child(Modello):
        x = InstanceDummy("x", positive=True)
        fixed = S.One

    class Parent(Modello):
        child = component(Child)
        x = child.x
        fixed = child.fixed

    with pytest.raises(ValueError, match="0 solutions"):
        Parent("P", x=0)
    with pytest.raises(ValueError, match="0 solutions"):
        Parent("P", fixed=2)


def test_undefined_candidates_are_filtered_before_selecting_a_root():
    """A pole excludes one polynomial root without discarding the valid solution."""

    class Child(Modello):
        x = InstanceDummy("x")
        inverse = 1 / x
        polynomial = x * (x - 1)

    class Parent(Modello):
        child = component(Child)
        polynomial = child.polynomial

    parent = Parent("P", polynomial=0)
    assert (parent.child.x, parent.child.inverse) == (1, 1)

"""Joint solving selects candidates and populates independent component trees."""

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
    "declare",
    [pytest.param(component, id="typed"), pytest.param(lambda model: model, id="raw-class")],
)
def test_component_declarations_accept_preconstructed_children(declare):
    """Both declaration forms bind an independently constructed child."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        c = a + b

    child = Child("C", a=2, b=3)

    class Parent(Modello):
        child = declare(Child)
        d = InstanceDummy("d")
        e = child.c + d

    instance = Parent("P", child=child, d=4)
    assert instance.child.c == 5
    assert instance.e == 9


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


def test_nested_partial_values():
    """Unspecified nested attributes are solved with the parent."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        c = a + b

    class Parent(Modello):
        child = Child
        c_total = child.c + 1

    instance = Parent("P", child=Child("C", a=2))
    # b defaults to dummy but derived c should resolve using constraints
    assert instance.child.c == instance.child.a + instance.child.b
    assert instance.c_total == instance.child.c + 1


def test_nested_multiple_levels():
    """Nested models work across more than one level."""

    class Leaf(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        total = a + b

    class Branch(Modello):
        leaf = Leaf
        offset = InstanceDummy("offset")
        total = leaf.total + offset

    class Tree(Modello):
        left = Branch
        right = Branch
        whole = left.total + right.total

    left = Branch("L", leaf=Leaf("LL", a=1, b=2), offset=1)
    right = Branch("R", leaf=Leaf("RL", a=2, b=3), offset=2)
    tree = Tree("T", left=left, right=right)
    assert tree.left.leaf.total == 3
    assert tree.left.total == 4
    assert tree.right.leaf.total == 5
    assert tree.right.total == 7
    assert tree.whole == 11
    assert (tree.left.leaf.a, tree.left.leaf.b) == (1, 2)
    assert (tree.right.leaf.a, tree.right.leaf.b) == (2, 3)
    assert tree.left is not left
    assert tree.left.leaf is not left.leaf


def test_nested_geometry_forward_and_inverse():
    """Parent constraints solve descendants, including their own derived fields."""
    from examples.geometry import RightAngleTriangle
    from examples.nested_geometry import Box
    from examples.nested_line import Line, Point

    line = Line("L", start=Point("start", x=0, y=0), end=Point("end", x=3, y=4))
    assert line.length == 5
    assert isinstance(line.start, Point)
    assert line.start is not line.end

    box = Box("B", base=RightAngleTriangle("base", b=4), height=12, diagonal=13)
    assert (box.base.a, box.base.b, box.base.c) == (3, 4, 5)


def test_deep_instances_and_attribute_paths():
    """Instances and expressions resolve every level of a repeated subtree."""

    class Leaf(Modello):
        x = InstanceDummy("x")
        y = x + 1

    class Branch(Modello):
        leaf = Leaf
        doubled = leaf.y * 2

    class Tree(Modello):
        left = Branch
        right = Branch
        total = left.doubled + right.doubled
        deep = left.leaf.x + right.leaf.y

    tree = Tree("T", left=Branch("L", leaf=Leaf("LL", x=2)), right=Branch("R", leaf=Leaf("RL", x=5)))
    assert (tree.left.leaf.x, tree.left.leaf.y, tree.left.doubled) == (2, 3, 6)
    assert (tree.right.leaf.x, tree.right.leaf.y, tree.right.doubled) == (5, 6, 12)
    assert (tree.total, tree.deep) == (18, 8)


def test_missing_nested_inputs_remain_shared_parameters():
    """Underdetermined trees retain free inputs rather than choosing zeroes."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        total = a + b

    class Parent(Modello):
        child = Child
        total = child.total + 1

    first = Parent("P", child=Child("C", a=2))
    second = Parent("P", child=Child("C", a=2))
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


def test_empty_and_unconstrained_child_instances():
    """Empty and unconstrained children can be supplied or omitted."""

    class Empty(Modello):
        pass

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        empty = Empty
        child = Child

    parent = Parent("P", child=Child("C", x=5), empty=Empty("E"))
    assert parent.child.x == 5
    assert isinstance(parent.empty, Empty)
    assert isinstance(Parent("P").child.x, BoundInstanceDummy)


def test_inconsistent_and_ambiguous_nested_systems_are_rejected():
    """Nested constraints retain assumptions and the unique-solution requirement."""

    class Child(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    class Parent(Modello):
        child = Child
        total = child.squared + 1

    with pytest.raises(ModelSolveError, match="2 solutions") as error:
        Parent("P", total=5)
    assert "Parent('P')" in str(error.value)
    assert "child: Child\n    c1: squared = x**2" in str(error.value.problem)
    assert "total = 5" in str(error.value.problem)
    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Parent("P", child=Child("C", x=2), total=10)
    assert "x = 2" in str(error.value.problem)
    assert "total = 10" in str(error.value.problem)


def test_error_labels_preserve_noncommutative_relationships():
    """Readable names must not turn an ordered product difference into zero."""

    class Ordered(Modello):
        scale = InstanceDummy("scale", positive=True)
        x = InstanceDummy("x", commutative=False)
        y = InstanceDummy("y", commutative=False)
        commutator = x * y - y * x

    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Ordered("O", scale=0)
    assert "commutator = x*y - y*x" in str(error.value.problem)


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


def test_nested_construction_uses_declarations_without_child_initializers():
    """Composition solves declarations directly instead of independently rebuilding children."""
    calls = []

    class Child(Modello):
        x = InstanceDummy("x")

        def __init__(self, name, **values):
            """Record explicit standalone construction."""
            calls.append(name)
            super().__init__(name, **values)

    class Parent(Modello):
        child = Child

    source = Child("C", x=2)
    parent = Parent("P", child=source)
    assert parent.child.x == 2
    assert calls == ["C"]


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

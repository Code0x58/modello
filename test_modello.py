"""Functional tests for Modello instances."""
from modello import BoundInstanceDummy, InstanceDummy, Modello
from sympy import simplify


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


def test_multiple_inheritance_expr_conflict():
    """Overrided modello attributes are replaced with new values."""

    class ExampleA(Modello):
        conflicted = InstanceDummy("conflicted")
        a = conflicted

    class ExampleB(Modello):
        conflicted = InstanceDummy("conflicted")
        b = conflicted

    class ExampleC(ExampleA, ExampleB):
        pass

    instance = ExampleC("Example")
    assert instance.a == instance.b  # dummy is different but value is the same
    assert instance.a == instance.conflicted

    assert ExampleC.conflicted == ExampleB.conflicted
    assert ExampleC.conflicted != ExampleA.conflicted


def test_solver_unique_solution():
    class Linear(Modello):
        x = InstanceDummy("x")
        y = InstanceDummy("y")
        total = x + y

    instance = Linear("L", x=2, total=5)
    assert instance.y == 3


def test_solver_multiple_solution_permissive_and_selector():
    class Branches(Modello):
        x = InstanceDummy("x")
        y = x**2

    permissive = Branches("B1", y=4, solution_mode="permissive")
    assert isinstance(permissive.x, BoundInstanceDummy)

    selected = Branches(
        "B2",
        y=4,
        solution_selector=lambda solutions: max(
            solutions, key=lambda sol: list(sol.values())[0]
        ),
    )
    assert selected.x == 2


def test_solver_underdetermined_set_strategy():
    class Under(Modello):
        x = InstanceDummy("x")
        y = x

    instance = Under("U", solver_strategy="set", solution_mode="permissive")
    assert isinstance(instance.x, BoundInstanceDummy)
    assert instance.y == instance.x

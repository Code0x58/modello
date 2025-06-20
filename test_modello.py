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


def test_nested_models_dict():
    """Nested models accept dictionaries of values."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        c = a + b

    class Parent(Modello):
        child = Child
        d = InstanceDummy("d")
        e = child.c + d

    instance = Parent("P", child={"a": 3, "b": 4}, d=5)
    assert instance.child.c == 7
    assert instance.e == 12


def test_nested_models_instance():
    """Nested models accept pre-instantiated children."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        c = a + b

    child = Child("C", a=2, b=3)

    class Parent(Modello):
        child = Child
        d = InstanceDummy("d")
        e = child.c + d

    instance = Parent("P", child=child, d=4)
    assert instance.child.c == 5
    assert instance.e == 9


def test_nested_partial_values():
    """Unspecified nested attributes are solved with the parent."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        c = a + b

    class Parent(Modello):
        child = Child
        c_total = child.c + 1

    instance = Parent("P", child={"a": 2})
    # b defaults to dummy but derived c should resolve using constraints
    assert instance.child.c == instance.child.a + instance.child.b
    assert instance.c_total == instance.child.c + 1

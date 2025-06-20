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


def test_helper_parse_nested_values():
    """_parse_nested_values splits nested data from values."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child
        y = InstanceDummy("y")

    instance = object.__new__(Parent)
    values = {"child": {"x": 5}, "y": 3}
    nested = instance._parse_nested_values(values)
    assert nested == {"child": {"x": 5}}
    assert values == {"y": 3}


def test_helper_create_instance_dummies():
    """_create_instance_dummies binds dummies for the instance."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child
        y = InstanceDummy("y")

    instance = object.__new__(Parent)
    dummies, nested = instance._create_instance_dummies("X")
    assert dummies[Parent._modello_namespace.dummies["y"]].name.startswith("X_")
    child_dummy = Child._modello_namespace.dummies["x"]
    assert child_dummy in nested["child"]
    assert nested["child"][child_dummy].name.startswith("X_")

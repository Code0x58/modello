"""Class declarations preserve aliases, inheritance, and component shape."""

import pytest
from sympy import Symbol

from modello import InstanceDummy, Modello


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


def test_overriding_a_child_rebinds_inherited_expressions():
    """Replacing a component updates inherited references without changing bases."""

    class Child(Modello):
        x = InstanceDummy("x")
        y = x + 1

    class Replacement(Modello):
        x = InstanceDummy("x")
        y = x + 10

    class Parent(Modello):
        child = Child
        total = child.y + 1
        alias = child.x

    class Derived(Parent):
        child = Replacement
        doubled = child.y * 2

    class Restored(Derived):
        child = Child

    parent = Derived("D", child=Replacement("C", x=2))
    assert (parent.child.y, parent.total, parent.doubled, parent.alias) == (12, 13, 24, 2)
    assert Parent("P", child=Child("C", x=2)).total == 4
    assert Restored("R", child=Child("C", x=2)).total == 4


def test_nested_overrides_follow_existing_multiple_base_precedence():
    """Later model bases override earlier model bases throughout the field tree."""

    class Child(Modello):
        x = InstanceDummy("x")
        y = x + 1

    class Replacement(Modello):
        x = InstanceDummy("x")
        y = x + 10

    class First(Modello):
        child = Child
        a = child.y

    class Second(Modello):
        child = Replacement
        b = child.y

    class Combined(First, Second):
        pass

    instance = Combined("C", child=Replacement("R", x=2))
    assert (instance.child.y, instance.a, instance.b) == (12, 12, 12)


def test_removed_referenced_child_fields_are_rejected():
    """An inherited expression cannot silently keep a symbol from a removed field."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Replacement(Modello):
        y = InstanceDummy("y")

    class Parent(Modello):
        child = Child
        total = child.x + 1

    with pytest.raises(ValueError, match="child.x"):

        class Invalid(Parent):
            child = Replacement

    class Valid(Parent):
        child = Replacement
        total = child.y + 1

    assert Valid("V", child=Replacement("R", y=2)).total == 3


def test_overridden_scalar_assumptions_reach_inherited_expressions():
    """Replacing an independent scalar rebinds formulas and preserves new assumptions."""

    class Base(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    class Positive(Base):
        x = InstanceDummy("x", positive=True)

    assert Positive("P", squared=4).x == 2
    with pytest.raises(ValueError, match="2 solutions"):
        Base("B", squared=4)


def test_ordinary_method_inheritance_and_super_survive_nesting():
    """Model inheritance preserves ordinary mixins, class cells, and methods."""

    class Labels:
        def label(self):
            """Return the base label."""
            return "base"

    class Child(Labels, Modello):
        x = InstanceDummy("x")

    class Derived(Child):
        def label(self):
            """Extend the base label using normal Python super semantics."""
            return super().label() + " derived"

    class Grandchild(Derived):
        pass

    class Parent(Modello):
        child = Grandchild

    assert Parent("P", child=Grandchild("G", x=1)).child.label() == "base derived"


def test_alias_definitions_add_constraints_instead_of_overwriting_them():
    """A parent alias can constrain a child's existing derived relationship."""

    class Child(Modello):
        x = InstanceDummy("x")
        y = x + 1

    class Parent(Modello):
        child = Child
        alias = child.y
        alias = child.x * 2

    parent = Parent("P")
    assert (parent.child.x, parent.child.y, parent.alias) == (1, 2, 2)


def test_nested_reference_unknown_attributes():
    """Unknown class reference attributes have a clear validation error."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child

    with pytest.raises(AttributeError, match="Child.*nope"):
        Parent.child.nope


def test_scalar_and_nested_fields_cannot_change_kinds():
    """Replacing a field with another kind cannot leave hidden stale equations."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child
        x = InstanceDummy("x")

    with pytest.raises(ValueError, match="Cannot replace nested model"):

        class Scalar(Parent):
            child = Symbol("scalar")

    with pytest.raises(ValueError, match="Cannot replace scalar"):

        class Nested(Parent):
            x = Child

    with pytest.raises(ValueError, match="non-model value"):

        class Plain(Parent):
            child = None


def test_incompatible_field_kinds_in_multiple_bases_are_rejected():
    """Multiple bases cannot define the same name as both a scalar and a child."""

    class Child(Modello):
        value = InstanceDummy("value")

    class Scalar(Modello):
        x = InstanceDummy("x")

    class Nested(Modello):
        x = Child

    with pytest.raises(ValueError, match="scalar and nested.*x"):

        class Invalid(Scalar, Nested):
            pass


def test_breaking_referenced_child_aliases_is_rejected():
    """An old aliased symbol cannot be silently mapped to two new inputs."""

    class Child(Modello):
        x = InstanceDummy("x")
        y = x

    class Replacement(Modello):
        x = InstanceDummy("x")
        y = InstanceDummy("y")

    class Parent(Modello):
        child = Child
        total = child.x + 1

    with pytest.raises(ValueError, match="changed field aliases"):

        class Invalid(Parent):
            child = Replacement


def test_metadata_prefix_is_reserved_for_model_fields():
    """Symbolic fields cannot overwrite the model's internal metadata."""
    with pytest.raises(ValueError, match="prefix is reserved"):

        class Invalid(Modello):
            _modello_symbol = Symbol("reserved")


def test_reference_helpers_do_not_shadow_public_model_fields():
    """Public field names are not taken over by the component reference machinery."""

    class Child(Modello):
        create = InstanceDummy("create")

    class Parent(Modello):
        child = Child
        total = child.create + 1

    assert Parent("P", child=Child("C", create=2)).total == 3


def test_class_body_component_aliases_require_an_explicit_model_class():
    """Copying a component reference must not leak an unbound template to instances."""

    class Child(Modello):
        x = InstanceDummy("x")

    with pytest.raises(TypeError, match="using its model class"):

        class Invalid(Modello):
            child = Child
            alias = child

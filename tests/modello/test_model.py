"""Model declarations, references, and population of component trees."""

import pytest
from sympy import (
    S,
    Symbol,
)

from modello import (
    BoundInstanceDummy,
    InstanceDummy,
    Modello,
    component,
    inspect_model,
)

from ._models import Increment, Pair, Reading, Scalar, ScalarComponent


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

    class Replacement(Modello):
        x = InstanceDummy("x")
        y = x + 10

    class Parent(Modello):
        child = Increment
        total = child.y + 1
        alias = child.x

    class Derived(Parent):
        child = Replacement
        doubled = child.y * 2

    class Restored(Derived):
        child = Increment

    parent = Derived("D", child=Replacement("C", x=2))
    assert (parent.child.y, parent.total, parent.doubled, parent.alias) == (12, 13, 24, 2)
    assert Parent("P", child=Increment("C", x=2)).total == 4
    assert Restored("R", child=Increment("C", x=2)).total == 4


def test_nested_overrides_follow_existing_multiple_base_precedence():
    """Later model bases override earlier model bases throughout the field tree."""

    class Replacement(Modello):
        x = InstanceDummy("x")
        y = x + 10

    class First(Modello):
        child = Increment
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

    class Replacement(Modello):
        y = InstanceDummy("y")

    class Parent(Modello):
        child = Scalar
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

    class Positive(Reading):
        x = InstanceDummy("x", positive=True)

    assert Positive("P", squared=4).x == 2
    with pytest.raises(ValueError, match="2 solutions"):
        Reading("B", squared=4)


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

    class Parent(Modello):
        child = Increment
        alias = child.y
        alias = child.x * 2

    parent = Parent("P")
    assert (parent.child.x, parent.child.y, parent.alias) == (1, 2, 2)


def test_nested_reference_unknown_attributes():
    """Unknown class reference attributes have a clear validation error."""

    with pytest.raises(AttributeError, match="Scalar.*nope"):
        ScalarComponent.child.nope


def test_scalar_and_nested_fields_cannot_change_kinds():
    """Replacing a field with another kind cannot leave hidden stale equations."""

    class Parent(Modello):
        child = Scalar
        x = InstanceDummy("x")

    with pytest.raises(ValueError, match="Cannot replace nested model"):

        class InvalidScalar(Parent):
            child = Symbol("scalar")

    with pytest.raises(ValueError, match="Cannot replace scalar"):

        class Nested(Parent):
            x = Scalar

    with pytest.raises(ValueError, match="non-model value"):

        class Plain(Parent):
            child = None


def test_incompatible_field_kinds_in_multiple_bases_are_rejected():
    """Multiple bases cannot define the same name as both a scalar and a child."""

    class Child(Modello):
        value = InstanceDummy("value")

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

    with pytest.raises(TypeError, match="using its model class"):

        class Invalid(Modello):
            child = Scalar
            alias = child


@pytest.mark.parametrize(
    "declare",
    [pytest.param(component, id="typed"), pytest.param(lambda model: model, id="raw-class")],
)
def test_component_declarations_accept_preconstructed_children(declare):
    """Both declaration forms bind an independently constructed child."""

    child = Pair("C", a=2, b=3)

    class Parent(Modello):
        child = declare(Pair)
        d = InstanceDummy("d")
        e = child.total + d

    instance = Parent("P", child=child, d=4)
    assert instance.child.total == 5
    assert instance.e == 9


def test_repeated_subtrees_preserve_deep_paths_and_source_identity():
    """Repeated subtrees retain their values and resolve cross-branch paths."""

    class Branch(Modello):
        leaf = Pair
        offset = InstanceDummy("offset")
        total = leaf.total + offset

    class Tree(Modello):
        left = Branch
        right = Branch
        whole = left.total + right.total
        cross_branch = left.leaf.a + right.leaf.total

    left = Branch("L", leaf=Pair("LL", a=1, b=2), offset=1)
    right = Branch("R", leaf=Pair("RL", a=2, b=3), offset=2)
    tree = Tree("T", left=left, right=right)
    assert tree.left.leaf.total == 3
    assert tree.left.total == 4
    assert tree.right.leaf.total == 5
    assert tree.right.total == 7
    assert tree.whole == 11
    assert tree.cross_branch == 6
    assert (tree.left.leaf.a, tree.left.leaf.b) == (1, 2)
    assert (tree.right.leaf.a, tree.right.leaf.b) == (2, 3)
    assert tree.left is not left
    assert tree.left.leaf is not left.leaf


def test_empty_and_unconstrained_child_instances():
    """Empty and unconstrained children can be supplied or omitted."""

    class Empty(Modello):
        pass

    class Parent(Modello):
        empty = Empty
        child = Scalar

    parent = Parent("P", child=Scalar("C", x=5), empty=Empty("E"))
    assert parent.child.x == 5
    assert isinstance(parent.empty, Empty)
    assert isinstance(Parent("P").child.x, BoundInstanceDummy)


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


def test_nested_refinement_keeps_both_definitions_of_an_aliased_quantity():
    """An alias and its component equation retain separate origins when nested."""

    class Doubled(Modello):
        x = InstanceDummy("x", real=True)
        total = 2 * x

    class Joined(Modello):
        reading = component(Doubled)
        observed = reading.total
        observed = reading.x + 3

    class Tripled(Doubled):
        total = 3 * Doubled.x

    class Refined(Joined):
        reading = component(Tripled)

    class Root(Modello):
        left = component(Refined)
        right = component(Joined)

    problem = inspect_model(Root, "origins")
    symbols = problem.symbols
    assert [(c.path, c.left, c.right) for c in problem.constraints] == [
        (("left", "observed"), symbols.left.reading.total, symbols.left.reading.x + 3),
        (("left", "reading", "total"), symbols.left.reading.total, 3 * symbols.left.reading.x),
        (("right", "observed"), symbols.right.reading.total, symbols.right.reading.x + 3),
        (("right", "reading", "total"), symbols.right.reading.total, 2 * symbols.right.reading.x),
    ]
    for case in (Root("constructed"), problem.solutions().one()):
        assert isinstance(case.left.reading, Tripled)
        assert (case.left.reading.x, case.left.observed) == (S(3) / 2, S(9) / 2)
        assert (case.right.reading.x, case.right.observed) == (3, 6)

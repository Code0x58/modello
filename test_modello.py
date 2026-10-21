"""Functional tests for Modello instances."""

from collections import UserDict
from decimal import Decimal
from fractions import Fraction

import pytest
from sympy import S, Symbol, simplify

from modello import BoundInstanceDummy, InstanceDummy, Modello, ModelSolveError, component


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


def test_typed_component_declarations():
    """Typed component declarations accept explicitly constructed children."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        c = a + b

    class Parent(Modello):
        child = component(Child)
        d = InstanceDummy("d")
        e = child.c + d

    instance = Parent("P", child=Child("C", a=3, b=4), d=5)
    assert instance.child.c == 7
    assert instance.e == 12


def test_nested_models_instance():
    """Raw model class declarations still accept pre-instantiated children."""

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


@pytest.mark.parametrize("value", [2, 0.5, 1 + 2j, "x + 1", Decimal("0.5"), Fraction(1, 2), Symbol("x")])
def test_scalar_input_types(value):
    """The typed scalar input boundary preserves ordinary SymPy-compatible values."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = component(Child)

    assert Parent("P", child=Child("C", x=value)).child.x == simplify(value)


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


def test_partial_instance_can_be_completed_without_mutating_source():
    """A parent's equations can complete a symbolic snapshot of a child."""

    class Child(Modello):
        a = InstanceDummy("a")
        b = InstanceDummy("b")
        total = a + b

    class Parent(Modello):
        child = Child
        total = child.total + 1

    source = Child("C", a=2)
    parent = Parent("P", child=source, total=10)
    assert (parent.child.a, parent.child.b, parent.child.total) == (2, 7, 9)
    assert isinstance(source.b, BoundInstanceDummy)
    assert source.total == source.b + 2
    assert parent.child is not source


def test_reusing_symbolic_instances_creates_independent_children():
    """Reusing an input instance copies its values into independent symbol scopes."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        left = Child
        right = Child
        difference = left.x - right.x

    source = Child("C")
    parent = Parent("P", left=source, right=source)
    assert parent.left.x != parent.right.x
    assert parent.left.x != source.x
    assert parent.right.x != source.x
    assert parent.difference == parent.left.x - parent.right.x


def test_input_expressions_resolve_relative_and_sibling_symbols():
    """Input expressions can use child-class symbols or the parent's full paths."""

    class Child(Modello):
        x = InstanceDummy("x")
        y = InstanceDummy("y")

    class Parent(Modello):
        left = Child
        right = Child
        total = left.x + right.x

    parent = Parent("P", left=Child("L", x=Child.y + Parent.right.x, y=3), right=Child("R", x=2))
    assert parent.left.x == 5
    assert parent.total == 7


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


@pytest.mark.parametrize("value", [None, 1, [], {}, {"x": 5}, UserDict(x=5), object()])
def test_invalid_nested_values_are_rejected(value):
    """Invalid nested values are never silently treated as missing inputs."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child

    with pytest.raises(TypeError, match="child must be a Child instance"):
        Parent("P", child=value)


def test_unknown_attributes_and_deep_mappings_are_rejected():
    """Unknown keywords and mappings at every constructor boundary are invalid."""

    class Leaf(Modello):
        x = InstanceDummy("x")

    class Branch(Modello):
        leaf = Leaf

    class Tree(Modello):
        branch = Branch

    with pytest.raises(TypeError, match="branch must be a Branch instance"):
        Tree("T", branch={"leaf": {"x": 1}})
    with pytest.raises(TypeError, match="leaf must be a Leaf instance"):
        Tree("T", branch=Branch("B", leaf={"x": 1}))
    with pytest.raises(TypeError, match="Unknown model attribute 'typo'"):
        Leaf("L", typo=1)
    with pytest.raises(TypeError, match="Unknown model attribute 'typo'"):
        Tree("T", typo=1)


def test_child_instances_must_have_a_compatible_model_type():
    """Structurally similar but unrelated model instances are rejected."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Other(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child

    with pytest.raises(TypeError, match="child.*Child"):
        Parent("P", child=Other("O", x=3))

    class Derived(Child):
        y = Child.x + 1

    assert Parent("P", child=Derived("D", x=3)).child.x == 3


def test_scalar_aliases_keep_all_supplied_constraints():
    """Aliases share values, and contradictory inputs cannot overwrite each other."""

    class Child(Modello):
        x = InstanceDummy("x")
        alias = x

    class Parent(Modello):
        child = Child
        alias = child.x

    parent = Parent("P", alias=3)
    assert (parent.child.x, parent.child.alias, parent.alias) == (3, 3, 3)
    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Child("C", x=1, alias=2)
    assert "x = 1" in str(error.value.problem)
    assert "alias = 2" in str(error.value.problem)
    with pytest.raises(ValueError, match="0 solutions"):
        Parent("P", child=Child("C", x=1), alias=2)


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
    assert "child.squared = child.x**2" in str(error.value.problem)
    assert "total = 5" in str(error.value.problem)
    with pytest.raises(ModelSolveError, match="0 solutions") as error:
        Parent("P", child=Child("C", x=2), total=10)
    assert "child.x = 2" in str(error.value.problem)
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


@pytest.mark.parametrize("value", [None, True, [], {}, object()])
def test_invalid_scalar_values_identify_the_field(value):
    """Malformed scalar values report their path rather than failing inside SymPy."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child

    source = Child("C")
    source.x = value
    with pytest.raises(TypeError, match="child.x.*scalar"):
        Parent("P", child=source)


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


def test_instance_subclasses_must_retain_the_declared_child_shape():
    """An instance subclass cannot omit paths expected by its declared component."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Replacement(Modello):
        y = InstanceDummy("y")

    class Base(Modello):
        child = Child

    class Derived(Base):
        child = Replacement

    class Parent(Modello):
        payload = Base

    with pytest.raises(TypeError, match="missing payload.child.x"):
        Parent("P", payload=Derived("D", child=Replacement("R", y=3)))


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


def test_external_bound_symbols_remain_owned_by_the_source_instance():
    """Passing an external symbol as a scalar keeps an explicit symbolic dependency."""

    class Child(Modello):
        x = InstanceDummy("x")

    class Parent(Modello):
        child = Child
        doubled = child.x * 2

    source = Child("external")
    parent = Parent("P", child=Child("C", x=source.x))
    assert parent.child.x == source.x
    assert parent.doubled == 2 * source.x
    assert isinstance(source.x, BoundInstanceDummy)

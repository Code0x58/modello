"""Scalar and component observations retain types, identities, and provenance."""

from decimal import Decimal
from fractions import Fraction

import pytest
from sympy import S, Symbol, simplify

from modello import (
    BoundInstanceDummy,
    ConstraintKind,
    InstanceDummy,
    Modello,
    ModelSolveError,
    component,
    inspect_model,
)

from ._models import Check, Envelope, Pair, PairParent, PositiveReading, Reading, Scalar, ScalarComponent


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(2, id="integer"),
        pytest.param(0.5, id="float"),
        pytest.param(1 + 2j, id="complex"),
        pytest.param("x + 1", id="string-expression"),
        pytest.param(Decimal("0.5"), id="decimal"),
        pytest.param(Fraction(1, 2), id="fraction"),
        pytest.param(Symbol("x"), id="external-symbol"),
    ],
)
def test_scalar_input_types(value):
    """Capture accepts ordinary scalars before construction converts their values."""

    problem = inspect_model(Scalar, "scalar", x=value)
    (observation,) = tuple(c for c in problem.constraints if c.input is not None)
    assert observation.kind is ConstraintKind.INPUT
    assert observation.right == simplify(value)
    assert Scalar("constructed", x=value).x == simplify(value)


def test_partial_instance_can_be_completed_without_mutating_source():
    """A parent's equations can complete a symbolic snapshot of a child."""

    source = Pair("C", a=2)
    parent = PairParent("P", child=source, total=10)
    assert (parent.child.a, parent.child.b, parent.child.total) == (2, 7, 9)
    assert isinstance(source.b, BoundInstanceDummy)
    assert source.total == source.b + 2
    assert parent.child is not source


def test_reusing_symbolic_instances_creates_independent_children():
    """Reusing an input instance copies its values into independent symbol scopes."""

    class Parent(Modello):
        left = Scalar
        right = Scalar
        difference = left.x - right.x

    source = Scalar("C")
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


@pytest.mark.parametrize(
    "value",
    [None, {"x": 5}, object()],
    ids=["none", "field-mapping", "opaque-object"],
)
def test_invalid_nested_values_are_rejected(value):
    """Invalid nested values are never silently treated as missing inputs."""

    with pytest.raises(TypeError, match="child must be a Scalar instance"):
        ScalarComponent("P", child=value)


@pytest.mark.parametrize(
    "boundary, message",
    [
        pytest.param("root-mapping", "branch must be a Branch instance", id="root-mapping"),
        pytest.param("nested-mapping", "leaf must be a Scalar instance", id="nested-mapping"),
        pytest.param("parent-keyword", "Unknown model attribute 'typo'", id="parent-keyword"),
    ],
)
def test_unknown_attributes_and_deep_mappings_are_rejected(boundary, message):
    """Unknown keywords and mappings at every constructor boundary are invalid."""

    class Branch(Modello):
        leaf = Scalar

    class Tree(Modello):
        branch = Branch

    operations = {
        "root-mapping": lambda: Tree("T", branch={"leaf": {"x": 1}}),
        "nested-mapping": lambda: Tree("T", branch=Branch("B", leaf={"x": 1})),
        "parent-keyword": lambda: Tree("T", typo=1),
    }
    with pytest.raises(TypeError, match=message):
        operations[boundary]()


def test_child_instances_must_have_a_compatible_model_type():
    """Structurally similar but unrelated model instances are rejected."""

    class Other(Modello):
        x = InstanceDummy("x")

    with pytest.raises(TypeError, match="child.*Scalar"):
        ScalarComponent("P", child=Other("O", x=3))

    class Derived(Scalar):
        y = Scalar.x + 1

    assert ScalarComponent("P", child=Derived("D", x=3)).child.x == 3


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


@pytest.mark.parametrize(
    "value", [None, True, [], {}, object()], ids=["none", "boolean", "list", "mapping", "opaque-object"]
)
def test_invalid_scalar_values_identify_the_field(value):
    """Malformed scalar values report their path rather than failing inside SymPy."""

    with pytest.raises(TypeError, match="x.*scalar"):
        Scalar("C", x=value)


def test_invalid_copied_scalar_identifies_its_nested_path():
    source = Reading("source", x=2)
    source.squared = object()
    with pytest.raises(TypeError, match=r"reading\.squared.*scalar"):
        Check("invalid", reading=source)


def test_instance_subclasses_must_retain_the_declared_child_shape():
    """An instance subclass cannot omit paths expected by its declared component."""

    class Replacement(Modello):
        y = InstanceDummy("y")

    class Derived(ScalarComponent):
        child = Replacement

    class Parent(Modello):
        payload = ScalarComponent

    with pytest.raises(TypeError, match="missing payload.child.x"):
        Parent("P", payload=Derived("D", child=Replacement("R", y=3)))


def test_external_bound_symbols_remain_owned_by_the_source_instance():
    """Passing an external symbol as a scalar keeps an explicit symbolic dependency."""

    class Parent(Modello):
        child = Scalar
        doubled = child.x * 2

    source = Scalar("external")
    parent = Parent("P", child=Scalar("C", x=source.x))
    assert parent.child.x == source.x
    assert parent.doubled == 2 * source.x
    assert isinstance(source.x, BoundInstanceDummy)


@pytest.mark.parametrize(
    "copy_from, observations, supplied, inferred",
    [
        pytest.param("original", {"x": 2}, "x", "squared", id="forward"),
        pytest.param("original", {"squared": 4}, "squared", "x", id="inverse"),
        pytest.param("original", {"alias": 2}, "x", "squared", id="alias"),
        pytest.param("populated-child", {"squared": 4}, "squared", "x", id="populated-child"),
        pytest.param("whole-parent", {"squared": 4}, "squared", "x", id="whole-parent"),
    ],
)
def test_copied_values_retain_construction_inputs_in_both_solve_directions(observations, supplied, inferred, copy_from):
    source = PositiveReading("source", **observations)
    if copy_from == "original":
        problem = inspect_model(Check, "direct", reading=source)
    else:
        parent = Check("parent", reading=source)
        if copy_from == "populated-child":
            problem = inspect_model(Check, "descendant", reading=parent.reading)
        else:
            problem = inspect_model(Envelope, "nested", check=parent)

    prefix = ("check", "reading") if copy_from == "whole-parent" else ("reading",)
    group = "check" if copy_from == "whole-parent" else "reading"
    expected = {
        (*prefix, supplied): ConstraintKind.INPUT,
        (*prefix, inferred): ConstraintKind.INFERRED,
    }
    if copy_from == "whole-parent":
        expected[("check", "total")] = ConstraintKind.INFERRED
    assert {c.path: c.kind for c in problem.constraints if c.input == group} == expected
    assert f"{inferred} = " in str(problem)
    assert f"[inferred in {group!r}]" in str(problem)
    assert f"[input {group!r}]" in str(problem)


def test_jointly_inferred_component_fields_remain_inferred():
    class PositiveCheck(Check):
        reading = component(PositiveReading)

    joined = PositiveCheck("joint", total=5)
    copied = inspect_model(Check, "copy_joint", reading=joined.reading)
    assert {c.kind for c in copied.constraints if c.input == "reading"} == {"inferred"}


@pytest.mark.parametrize("replacement", (9, S(9)))
def test_changed_component_results_are_inputs_instead_of_stale_inferences(replacement):
    source = Reading("source", x=2)
    source.squared = replacement
    p = inspect_model(Check, "changed", reading=source)
    copied = next(c for c in p.constraints if c.input and c.path == ("reading", "squared"))
    assert copied.kind == "input"
    assert copied.right == 9
    assert p.analyse().status == "inconsistent"

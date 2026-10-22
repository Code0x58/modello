"""Extensions annotate declarations without participating in their algebra."""

import subprocess
import sys
from dataclasses import dataclass

import pytest
from sympy import Basic

from modello import Annotation, InstanceDummy, Modello, annotate, annotate_model, component, inspect_model
from modello.errors import ModelDefinitionError, SolutionCountError


@dataclass(frozen=True)
class Description(Annotation):
    text: str


@dataclass(frozen=True)
class Unit(Annotation):
    name: str


def test_annotations_are_unwrapped_before_later_class_body_expressions():
    class Plain(Modello):
        x = InstanceDummy("x", positive=True)
        doubled = 2 * x

    class Described(Modello):
        x = annotate(InstanceDummy("x", positive=True), Description("Length"), Unit("m"))
        doubled = annotate(2 * x, Description("Twice the length"))

    assert isinstance(Described.x, InstanceDummy)
    assert isinstance(Described.doubled, InstanceDummy)
    assert Described("case", x=3).doubled == Plain("case", x=3).doubled == 6
    assert Described("case", doubled=6).x == 3
    with pytest.raises(SolutionCountError):
        Described("case", x=-1)
    plain, described = (inspect_model(model, "case") for model in (Plain, Described))
    assert [(c.kind, c.path, plain.format(c.right)) for c in plain.constraints] == [
        (c.kind, c.path, described.format(c.right)) for c in described.constraints
    ]
    assert Described.__modello_annotations__[Description]["x"] == Description("Length")
    assert Unit.for_model(Described)["x"].name == "m"


def test_stacked_extensions_keep_distinct_annotation_types():
    class NestedDescription(Description):
        pass

    class Model(Modello):
        value = annotate(
            annotate(InstanceDummy("value"), Description("Inner"), Unit("m")),
            NestedDescription("Separate kind"),
        )

    assert Description.for_model(Model)["value"].text == "Inner"
    assert Unit.for_model(Model)["value"].name == "m"
    assert NestedDescription.for_model(Model)["value"].text == "Separate kind"
    assert not Annotation.for_model(Model)


@pytest.mark.parametrize("stacked", [False, True])
def test_duplicate_annotations_raise_even_when_the_metadata_is_identical(stacked):
    info = Description("Same")
    with pytest.raises(ModelDefinitionError, match="Duplicate Description annotation"):
        if stacked:
            annotate(annotate(InstanceDummy("value"), info), info)
        else:
            annotate(InstanceDummy("value"), info, info)


def test_reannotating_a_field_in_one_class_body_is_rejected():
    with pytest.raises(ModelDefinitionError, match=r"Duplicate Description annotation on Invalid\.value"):

        class Invalid(Modello):
            value = annotate(InstanceDummy("value"), Description("Original"))
            value = annotate(value, Description("Second"))


def test_duplicate_model_annotations_are_rejected_without_partial_updates():
    class Model(Modello):
        value = InstanceDummy("value")

    annotate_model(Model, Description("Original"))
    with pytest.raises(ModelDefinitionError, match="Duplicate Description annotation on Model"):
        annotate_model(Model, Unit("m"), Description("Second"))
    assert Description.for_model(Model)[""].text == "Original"
    assert not Unit.for_model(Model)
    annotate_model(Model, Unit("m"))
    assert Unit.for_model(Model)[""].name == "m"


def test_annotations_stay_with_named_fields_and_aliases_keep_separate_labels():
    class Model(Modello):
        value = annotate(InstanceDummy("value", positive=True), Description("First"))
        alias = annotate(value, Description("Second"))
        doubled = annotate(InstanceDummy("doubled", positive=True), Unit("m"))
        doubled = 2 * value

    assert Model.value == Model.alias
    assert Description.for_model(Model)["value"].text == "First"
    assert Description.for_model(Model)["alias"].text == "Second"
    assert Unit.for_model(Model)["doubled"].name == "m"
    assert Model.doubled.is_positive
    assert Model("case", alias=3).doubled == 6


def test_annotations_inherit_by_name_without_mutating_parent_registries():
    class Parent(Modello):
        value = annotate(InstanceDummy("value"), Description("Parent"), Unit("m"))
        doubled = 2 * value

    annotate_model(Parent, Description("Parent model"))

    class Child(Parent):
        value = annotate(InstanceDummy("child_value"), Description("Child"))

    class PlainOverride(Parent):
        value = InstanceDummy("replacement")

    assert Description.for_model(Parent)["value"].text == "Parent"
    assert Description.for_model(Child)["value"].text == "Child"
    assert Description.for_model(PlainOverride)["value"].text == "Parent"
    assert Description.for_model(Child)[""].text == "Parent model"
    assert Unit.for_model(Child)["value"].name == "m"
    assert Child("case", value=3).doubled == 6
    with pytest.raises(TypeError):
        Child.__modello_annotations__[Description]["value"] = Description("Mutation")
    with pytest.raises(TypeError):
        Child.__modello_annotations__[Description] = {}


def test_later_model_bases_override_metadata_of_the_same_kind():
    class Left(Modello):
        value = annotate(InstanceDummy("left"), Description("Left"), Unit("m"))

    class Right(Modello):
        value = annotate(InstanceDummy("right"), Description("Right"))

    class Combined(Left, Right):
        pass

    assert Description.for_model(Combined)["value"].text == "Right"
    assert Unit.for_model(Combined)["value"].name == "m"
    assert Combined.value == Right.value


def test_nested_annotations_are_scoped_and_replaced_components_drop_stale_paths():
    class Leaf(Modello):
        value = annotate(InstanceDummy("value"), Description("Leaf value"))

    annotate_model(Leaf, Description("Leaf model"))

    class Middle(Modello):
        leaf = component(Leaf)

    class Root(Modello):
        first = annotate(component(Middle), Description("First component"))
        second = component(Middle)

    class Replacement(Modello):
        other = annotate(InstanceDummy("other"), Description("Replacement value"))

    class Changed(Root):
        first = component(Replacement)

    original = Description.for_model(Root)
    assert original["first"].text == "First component"
    assert original["first.leaf"].text == "Leaf model"
    assert original["first.leaf.value"].text == "Leaf value"
    assert original["second.leaf.value"].text == "Leaf value"
    assert Root.first.leaf.value != Root.second.leaf.value
    changed = Description.for_model(Changed)
    assert changed["first"].text == "First component"
    assert changed["first.other"].text == "Replacement value"
    assert "first.leaf.value" not in changed and "first.leaf" not in changed
    assert original["first.leaf.value"].text == "Leaf value"


def test_model_annotations_keep_class_identity_and_support_component_overrides():
    class Child(Modello):
        value = InstanceDummy("value")

    assert annotate_model(Child, Description("Child")) is Child

    class Parent(Modello):
        first = annotate(component(Child), Description("Override child label"))
        second = component(Child)

    assert Description.for_model(Parent)["first"].text == "Override child label"
    assert Description.for_model(Parent)["second"].text == "Child"
    assert Parent("case", first=Child("child", value=3)).first.value == 3


def test_annotation_values_are_not_sympified_and_plain_attributes_keep_their_type():
    class Opaque(Annotation):
        def _sympy_(self):
            raise AssertionError("Metadata must not enter SymPy")

    class Model(Modello):
        constant = annotate(7, Description("Ordinary class constant"))
        value = annotate(InstanceDummy("value"), Opaque())
        scaled = value * constant

    assert type(Model.constant) is int
    assert Model.constant == 7
    assert Model("case", value=2).scaled == 14
    assert all(isinstance(c.right, Basic) for c in inspect_model(Model, "case").constraints)
    value = InstanceDummy("plain")
    assert annotate(value) is value


def test_invalid_metadata_and_reserved_registry_are_rejected():
    with pytest.raises(ModelDefinitionError, match="derive from Annotation"):
        annotate(InstanceDummy("value"), object())
    with pytest.raises(ModelDefinitionError, match="Modello class"):
        annotate_model(int, Description("Bad target"))
    with pytest.raises(ModelDefinitionError, match="derive from Annotation"):
        annotate_model(Modello, object())
    with pytest.raises(ModelDefinitionError, match="managed by Modello"):

        class Invalid(Modello):
            __modello_annotations__ = {}


def test_generic_annotations_do_not_import_gui_or_optional_dependencies():
    program = """
import sys
from dataclasses import dataclass
from modello import Annotation, InstanceDummy, Modello, annotate

@dataclass(frozen=True)
class Label(Annotation):
    text: str

class Model(Modello):
    value = annotate(InstanceDummy('value'), Label('A value'))
    doubled = 2 * value

assert Model('case', value=2).doubled == 4
assert Label.for_model(Model)['value'].text == 'A value'
assert not {'modello.gui', 'flask'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

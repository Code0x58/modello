"""GUI layout composes metadata, visibility, navigation, and authoring errors."""

import pytest

from modello import InstanceDummy, Modello, annotate, component, inspect_model
from modello.errors import ModelInputError
from modello.gui import Explorer, FieldInfo, ModelInfo, Scenario, gui
from tests.modello._gui_models import Estimate, Panel, rows


def test_external_metadata_can_hide_a_component():
    data = Explorer(Estimate, metadata={"panel": FieldInfo(hidden=True)}).snapshot(raw=True)
    assert all(not f["path"].startswith("panel.") for f in data["fields"])
    assert not data["components"]


def test_model_info_composes_model_component_and_field_labels():
    @ModelInfo(
        title="Panel estimate",
        description="A model description.",
        info="A model note.",
        fields={"panel": FieldInfo(title="Geometry"), "panel.height": FieldInfo(description="Height in metres.")},
    )
    class Described(Modello):
        panel = component(Panel)

    data = Explorer(Described, metadata={"panel.area": FieldInfo(title="Surface")}).snapshot(raw=True)
    assert (data["title"], data["description"], data["info"]) == (
        "Panel estimate",
        "A model description.",
        "A model note.",
    )
    assert data["components"][0]["title"] == "Geometry"
    assert rows(data)["panel.width"]["description"] == "Width in metres."
    assert rows(data)["panel.height"]["description"] == "Height in metres."
    assert rows(data)["panel.area"]["title"] == "Surface"


def test_model_info_rejects_unsupported_declarations():
    @ModelInfo(title="Use annotations")
    class Legacy(Modello):
        __gui__ = ModelInfo(title="Old style")

    @ModelInfo(fields={"": FieldInfo(title="Title")})
    class EmptyPath(Modello):
        pass

    with pytest.raises(ModelInputError, match=r"Legacy.__gui__ is unsupported; use @ModelInfo"):
        Explorer(Legacy)
    with pytest.raises(ModelInputError, match="empty field path"):
        Explorer(EmptyPath)


def test_decorator_and_external_overrides_take_precedence_over_inline_metadata():
    @ModelInfo(title="Decorated", fields={"width": FieldInfo(title="Model override")})
    class Model(Modello):
        width = gui(InstanceDummy("width", positive=True), title="Inline")

    data = Explorer(Model).snapshot(raw=True)
    assert data["title"] == "Decorated"
    assert rows(data)["width"]["title"] == "Model override"
    overridden = Explorer(Model, metadata={"width": FieldInfo(title="Application override")}).snapshot(raw=True)
    assert rows(overridden)["width"]["title"] == "Application override"


def test_gui_annotations_follow_inherited_fields_and_replaced_components():
    class First(Modello):
        width = gui(InstanceDummy("width"), title="First width")

    class Second(Modello):
        height = gui(InstanceDummy("height"), title="Second height")

    class Parent(Modello):
        panel = gui(component(First), title="Geometry")
        price = gui(InstanceDummy("price"), title="Price")

    class Child(Parent):
        panel = component(Second)

    data = Explorer(Child).snapshot(raw=True)
    assert data["components"][0]["title"] == "Geometry"
    assert rows(data)["panel.height"]["title"] == "Second height"
    assert rows(data)["price"]["title"] == "Price"
    assert "panel.width" not in FieldInfo.for_model(Child)


def test_flat_registry_metadata_and_override_precedence_across_nested_models():
    @ModelInfo(title="Original model", fields={"value": FieldInfo(title="Child override")})
    class Child(Modello):
        value = gui(InstanceDummy("value"), title="Inline")

    @ModelInfo(fields={"child.value": FieldInfo(title="Parent override")})
    class Parent(Modello):
        child = annotate(
            component(Child), ModelInfo(title="Scoped model", fields={"value": FieldInfo(title="Scoped override")})
        )

    data = Explorer(Parent).snapshot(raw=True)
    assert data["components"][0]["title"] == "Scoped model"
    assert rows(data)["child.value"]["title"] == "Parent override"
    data = Explorer(Parent, metadata={"child.value": FieldInfo(title="Application override")}).snapshot(raw=True)
    assert rows(data)["child.value"]["title"] == "Application override"
    assert rows(Explorer(Child).snapshot(raw=True))["value"]["title"] == "Child override"


def test_hidden_components_propagate_by_path_without_hiding_similarly_named_siblings():
    class Child(Modello):
        value = InstanceDummy("value")
        alias = value

    class Parent(Modello):
        part = gui(component(Child), hidden=True)
        partial = component(Child)

    data = Explorer(Parent, inputs={"part.value": 2}).snapshot(raw=True)
    assert set(rows(data)) == {"partial.value", "partial.alias"}
    assert [component["path"] for component in data["components"]] == ["partial"]
    assert {field["path"] for field in data["symbols"]} == {"part.value", "part.alias"}
    assert data["hidden_inputs"] == 1


def test_empty_components_remain_in_the_raw_model_structure():
    class Empty(Modello):
        """No quantities yet."""

    class Root(Modello):
        empty = gui(component(Empty), title="Placeholder")

    data = Explorer(Root).snapshot(raw=True)
    assert not data["fields"]
    assert data["components"][0]["title"] == "Placeholder"


def test_hidden_fields_cannot_be_highlighted():
    @ModelInfo(highlights=("price",))
    class Invalid(Estimate):
        pass

    with pytest.raises(ModelInputError, match="hidden field 'price' cannot be highlighted"):
        Explorer(Invalid)


@pytest.mark.parametrize(
    ("options", "location"),
    [
        ({"fields": {"panel.wdth": FieldInfo()}}, "ModelInfo.fields['panel.wdth'] on 'Invalid'"),
        ({"highlights": ("panel.wdth",)}, "ModelInfo.highlights"),
        ({"initial_field": "panel.wdth"}, "ModelInfo.initial_field"),
        (
            {"scenarios": (Scenario(title="Target", description="A target.", select="panel.wdth"),)},
            "Scenario 'Target'.select",
        ),
        (
            {"scenarios": (Scenario(title="Target", description="A target.", bindings={"panel.wdth": 2}),)},
            "Scenario 'Target'.bindings",
        ),
    ],
)
def test_authoring_errors_identify_options_and_suggest_full_nested_paths(options, location):
    @ModelInfo(**options)
    class Invalid(Estimate):
        pass

    with pytest.raises(ModelInputError) as error:
        Explorer(Invalid)
    assert str(error.value) == f"{location}: Unknown field path 'panel.wdth'. Did you mean 'panel.width'?"


def test_nested_model_metadata_errors_locate_relative_and_full_paths():
    @ModelInfo(fields={"wdth": FieldInfo()})
    class InvalidPanel(Panel):
        pass

    class Project(Modello):
        panel = component(InvalidPanel)

    with pytest.raises(ModelInputError) as error:
        Explorer(Project)
    assert str(error.value) == (
        "ModelInfo.fields['wdth'] on 'panel': Unknown field path 'panel.wdth'. Did you mean 'panel.width'?"
    )


def test_metadata_value_errors_explain_expected_declarations():
    with pytest.raises(ModelInputError, match="metadata: expected FieldInfo at 'width'"):
        Explorer(Panel, metadata={"width": "Title"})
    with pytest.raises(ModelInputError, match="gui\\('width'\\): invalid number_format.*choose 'number' or 'percent'"):

        class Invalid(Panel):
            width = gui(InstanceDummy("width"), number_format="percentage")

        Explorer(Invalid)
    with pytest.raises(ModelInputError, match="gui\\('note'\\): Unknown field path 'note'"):

        class NonField(Panel):
            note = gui("Not a scalar or component", title="Note")

        Explorer(NonField)


def test_scenario_errors_identify_hidden_and_recorded_edit_policies():
    @ModelInfo(scenarios=(Scenario(title="Background", description="Edit a hidden input.", bindings={"price": 2}),))
    class Hidden(Estimate):
        pass

    with pytest.raises(ModelInputError, match="Scenario 'Background'.bindings: cannot edit hidden field 'price'"):
        Explorer(Hidden)

    @ModelInfo(scenarios=(Scenario(title="Copied", description="Edit a recorded result.", bindings={"panel.area": 6}),))
    class Recorded(Estimate):
        pass

    problem = inspect_model(Recorded, "copy", panel=Panel("source", width=2, height=3))
    with pytest.raises(ModelInputError, match="Scenario 'Copied'.bindings: cannot edit Recorded field 'panel.area'"):
        Explorer(problem)

    @ModelInfo(scenarios=("Use a Scenario object",))
    class Invalid(Panel):
        pass

    with pytest.raises(ModelInputError, match=r"ModelInfo.scenarios\[0\]: expected a Scenario declaration"):
        Explorer(Invalid)

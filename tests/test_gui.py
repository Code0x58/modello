"""Exploration preserves paths, input provenance, domains, and browser boundaries."""

import json
import subprocess
import sys
from xml.etree import ElementTree

import pytest
from sympy import Rational, Symbol, pi, sqrt

from modello import InstanceDummy, Modello, annotate, component, inspect_model
from modello._gui.inputs import parse_scalar
from modello.diagnostics import ConstraintKind
from modello.errors import ModelDefinitionError, ModelInputError
from modello.gui import BindingError, Explorer, FieldInfo, ModelInfo, Scenario, create_app, gui


class Panel(Modello):
    """A rectangular panel."""

    width = gui(InstanceDummy("width", positive=True), description="Width in metres.", symbol="w")
    height = InstanceDummy("height", positive=True)
    area = width * height


class Estimate(Modello):
    panel = component(Panel)
    price = gui(InstanceDummy("price", positive=True), symbol="p", hidden=True)
    total = panel.area * price


def rows(data):
    return {row["path"]: row for row in data["fields"]}


def test_raw_never_solves(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Raw rendering must not solve")

    monkeypatch.setattr("modello.solve", unexpected)
    data = Explorer(Estimate).snapshot(raw=True)
    assert data["status"] == "raw"
    assert rows(data)["panel.width"]["depth"] == 1
    assert rows(data)["panel.width"]["description"] == "Width in metres."
    assert rows(data)["total"]["kind"] == "derived"
    assert data["components"][0]["path"] == "panel"


def test_partial_binding_and_hidden_input():
    explorer = Explorer(Estimate, inputs={"panel.width": 2, "price": 10})
    data = explorer.snapshot()
    assert data["status"] == "conditional"
    assert rows(data)["panel.width"]["kind"] == "input"
    assert rows(data)["panel.height"]["kind"] == "free"
    assert rows(data)["total"]["value"] == "20*panel.height"
    assert data["hidden_inputs"] == 1
    assert "price" not in rows(data)
    assert data["symbols"][0]["value"] == "Hidden input"
    bound = explorer.snapshot({"panel.height": 3})
    assert bound["status"] == "checked"
    assert rows(bound)["total"]["value"] == "60"
    assert rows(bound)["total"]["kind"] == "derived"
    assert rows(bound)["total"]["value_state"] == "numeric"
    assert rows(explorer.snapshot())["panel.height"]["kind"] == "free"


def test_release_initial_input():
    explorer = Explorer(Panel, inputs={"width": 4, "height": 5})
    data = explorer.snapshot({"height": None})
    assert rows(data)["height"]["kind"] == "free"
    assert rows(data)["area"]["value"] == "4*height"
    assert rows(explorer.snapshot())["area"]["value"] == "20"


def test_override_drop_and_restore_individual_inputs():
    explorer = Explorer(Panel, inputs={"width": 4, "height": 5})
    initial = rows(explorer.snapshot())
    assert initial["width"]["initial_input"] == "4"
    assert initial["width"]["input_state"] == "initial"
    changed = rows(explorer.snapshot({"width": 6, "height": None}))
    assert changed["width"]["input_state"] == "overridden"
    assert changed["width"]["initial_input"] == "4"
    assert changed["height"]["input_state"] == "dropped"
    assert changed["height"]["kind"] == "free"
    assert changed["area"]["input_state"] == "unspecified"
    # Restoring width does not restore the separately dropped height.
    restored = rows(explorer.snapshot({"height": None}))
    assert restored["width"]["input_state"] == "initial"
    assert restored["height"]["input_state"] == "dropped"
    assert restored["area"]["value"] == "4*height"


def test_drop_input_and_observe_output_as_one_case():
    client = create_app(Panel, inputs={"width": 4, "height": 5}).test_client()
    contradictory = client.post("/api/explore", json={"bindings": {"area": "30"}})
    assert contradictory.status_code == 422
    response = client.post("/api/explore", json={"bindings": {"height": None, "area": "30"}})
    assert response.status_code == 200
    data = rows(response.json)
    # Dropping an observation does not require the quantity to stay free.
    assert data["height"]["input_state"] == "dropped"
    assert data["height"]["kind"] == "derived"
    assert data["height"]["value"] == "15/2"
    assert data["area"]["input_state"] == "added"
    # Switching back swaps the observations atomically in the other direction.
    restored = rows(client.post("/api/explore", json={"bindings": {"area": None}}).json)
    assert restored["height"]["input_state"] == "initial"
    assert restored["area"]["kind"] == "derived"
    assert restored["area"]["value"] == "20"


def test_input_history_uses_shared_aliases_and_hides_background_values():
    class Alias(Modello):
        value = InstanceDummy("value", positive=True)
        other = value

    explorer = Explorer(Alias, inputs={"value": 2})
    for field in rows(explorer.snapshot({"other": 3})).values():
        assert field["input_state"] == "overridden"
        assert field["initial_input"] == "2"
    for field in rows(explorer.snapshot({"other": None})).values():
        assert field["input_state"] == "dropped"
    hidden = Explorer(Estimate, inputs={"price": 1234567}).snapshot()["symbols"][0]
    assert hidden["input_state"] == "hidden"
    assert hidden["initial_input"] == ""
    assert "1234567" not in json.dumps(hidden)


def test_reopen_instance_uses_inputs_not_solved_outputs():
    original = Estimate("job", panel=Panel("panel", width=4, height=5), price=10)
    explorer = Explorer(original)
    data = explorer.snapshot({"panel.width": 6})
    assert rows(data)["total"]["value"] == "300"
    assert original.panel.width == 4 and original.total == 200
    assert all(c.kind is not ConstraintKind.INFERRED for c in explorer.problem.constraints)
    original.panel.width = 8
    changed = Explorer(original).snapshot()
    assert rows(changed)["total"]["value"] == "400"


def test_snapshot_keeps_inferred_component_observations():
    problem = inspect_model(Estimate, "snapshot", panel=Panel("source", width=4, height=5), price=10)
    data = Explorer(problem).snapshot()
    assert not rows(data)["panel.area"]["editable"]
    with pytest.raises(BindingError):
        Explorer(problem).snapshot({"panel.width": 6})


def test_reopen_solution_family_case_preserves_annotations_and_observed_targets():
    case = inspect_model(Panel, "target", height=3).solutions().observe(area=6).one()
    assert case.width == 2
    explorer = Explorer(case)
    initial = rows(explorer.snapshot())
    assert initial["width"]["description"] == "Width in metres."
    assert initial["width"]["kind"] == "derived"
    assert initial["area"]["kind"] == "input"
    changed = rows(explorer.snapshot({"height": 4}))
    assert changed["area"]["value"] == "6"
    assert changed["width"]["value"] == "3/2"
    assert case.width == 2 and case.height == 3


def test_alias_bindings_and_deep_paths():
    class Alias(Modello):
        value = InstanceDummy("value", real=True)
        other = value

    class Layer(Modello):
        leaf = component(Alias)

    class Root(Modello):
        layer = component(Layer)

    explorer = Explorer(Root)
    data = explorer.snapshot({"layer.leaf.other": 3})
    assert rows(data)["layer.leaf.value"]["value"] == "3"
    assert rows(data)["layer.leaf.value"]["depth"] == 2
    assert rows(data)["layer.leaf.other"]["aliases"] == ["layer.leaf.value", "layer.leaf.other"]
    with pytest.raises(BindingError, match="Aliases"):
        explorer.snapshot({"layer.leaf.value": 3, "layer.leaf.other": 4})


def test_validation_is_atomic_and_reports_indirect_domain_errors():
    class Difference(Modello):
        x = InstanceDummy("x", positive=True)
        remainder = InstanceDummy("remainder", positive=True)
        remainder = x - 2

    explorer = Explorer(Difference, inputs={"x": 3})
    with pytest.raises(BindingError, match="positive") as failure:
        explorer.snapshot({"x": -1})
    assert failure.value.paths == ("x",)
    with pytest.raises(BindingError) as failure:
        explorer.snapshot({"x": 1})
    assert "remainder" in failure.value.paths
    assert rows(explorer.snapshot())["remainder"]["value"] == "1"


def test_multiple_solutions_are_not_selected():
    class Square(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    with pytest.raises(BindingError, match="Multiple solutions"):
        Explorer(Square, inputs={"squared": 4}).snapshot()


def test_external_parameters_stay_conditional():
    class External(Modello):
        x = InstanceDummy("x", real=True)
        value = x * Symbol("parameter")

    data = Explorer(External, inputs={"x": 2}).snapshot()
    assert data["status"] == "conditional"
    assert "external[parameter]" in rows(data)["value"]["value"]


def test_mathml_is_valid_xml_and_escapes_metadata():
    data = Explorer(Estimate, metadata={"panel.width": FieldInfo(symbol='<script>"&')}).snapshot(raw=True)
    for equation in data["equations"]:
        root = ElementTree.fromstring(equation["mathml"])
        assert root.tag == "{http://www.w3.org/1998/Math/MathML}math"
        assert not root.findall(".//script")
    panel = next(e for e in data["equations"] if e["path"] == "panel.area")
    parsed = ElementTree.fromstring(panel["mathml"])
    width = next(element for element in parsed.iter() if element.get("data-path") == "panel.width")
    assert width.text == '<script>"&'
    assert 'data-path="panel.width"' in panel["mathml"]


@pytest.mark.parametrize(
    "text, value",
    [
        ("1.25", Rational(5, 4)),
        ("70%", Rational(7, 10)),
        (" 6 % ", Rational(3, 50)),
        ("1/2%", Rational(1, 200)),
        ("3/2", Rational(3, 2)),
        ("sqrt(2)", sqrt(2)),
        ("pi", pi),
        ("2**3 - 1", 7),
        ("-2 + +3", 1),
    ],
)
def test_exact_arithmetic(text, value):
    assert parse_scalar(text) == value


@pytest.mark.parametrize(
    "text",
    [
        "%",
        "10%%",
        "1%2",
        "True",
        "1/0",
        "float('inf')",
        "1e9999",
        "1e-999999999",
        "2**101",
        "(2**100)**100",
        "[1,2]",
        "x",
        "x.width",
        "__import__('os').getcwd()",
        "sqrt(x=2)",
        "sqrt(1, 2)",
        "1 +",
        "",
        "2" * 161,
    ],
)
def test_parser_rejects_code_and_unbounded_arithmetic(text):
    with pytest.raises(ModelInputError):
        parse_scalar(text)


def test_metadata_errors_and_component_hiding():
    with pytest.raises(ModelInputError, match="metadata"):
        Explorer(Panel, metadata={"missing": FieldInfo()})
    data = Explorer(Estimate, metadata={"panel": FieldInfo(hidden=True)}).snapshot(raw=True)
    assert all(not f["path"].startswith("panel.") for f in data["fields"])
    assert not data["components"]
    with pytest.raises(BindingError, match="Unknown field"):
        Explorer(Panel).snapshot({"missing": 2})


def test_app_updates_are_isolated_and_invalid_edits_preserve_previous_model():
    client = create_app(Estimate, inputs={"panel.width": 4, "panel.height": 5, "price": 10}).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/assets/explorer.js").status_code == 200
    assert client.get("/assets/explorer.css").status_code == 200
    response = client.post("/api/explore", json={"bindings": {"panel.width": "6"}})
    assert response.status_code == 200
    assert rows(response.json)["total"]["value"] == "300"
    response = client.post("/api/explore", json={"bindings": {"panel.width": "-2"}})
    assert response.status_code == 422 and response.json["paths"] == ["panel.width"]
    response = client.post("/api/explore", json={"bindings": {"price": "20"}})
    assert response.status_code == 422
    assert rows(client.post("/api/explore", json={}).json)["total"]["value"] == "200"
    assert (
        rows(client.post("/api/explore", json={"bindings": {"panel.height": None}}).json)["panel.height"]["kind"]
        == "free"
    )
    assert client.post("/api/explore", json={"raw": True}).json["raw"]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"x": 1},
        {"raw": "yes"},
        {"bindings": []},
        {"bindings": {"width": 2}},
        {"bindings": {"width": "__import__('os')"}},
    ],
)
def test_app_rejects_malformed_requests(payload):
    client = create_app(Panel).test_client()
    assert client.post("/api/explore", json=payload).status_code in (400, 422)
    assert client.post("/api/explore", data="not json").status_code == 400


def test_import_does_not_load_gui_runtime():
    program = "import sys; import modello; import modello.gui; assert 'flask' not in sys.modules"
    subprocess.run([sys.executable, "-c", program], check=True)


def test_hidden_observations_are_not_sent_to_browser():
    data = Explorer(Estimate, inputs={"price": 987654321}).snapshot(raw=True)
    assert "987654321" not in json.dumps(data)


def test_hidden_alias_cannot_release_or_overwrite_an_input():
    class Alias(Modello):
        value = InstanceDummy("value", real=True)
        other = value

    client = create_app(Alias, inputs={"value": 4}, metadata={"value": FieldInfo(hidden=True)}).test_client()
    assert client.post("/api/explore", json={}).json["fields"] == []
    for value in ("6", None):
        assert client.post("/api/explore", json={"bindings": {"other": value}}).status_code == 422


def test_browser_cannot_edit_a_copied_inferred_observation():
    problem = inspect_model(Estimate, "snapshot", panel=Panel("source", width=4, height=5), price=10)
    client = create_app(problem).test_client()
    response = client.post("/api/explore", json={"bindings": {"panel.area": "20"}})
    assert response.status_code == 422
    assert response.json["paths"] == ["panel.area"]


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


def test_inline_annotations_supply_model_component_and_scalar_presentation():
    @ModelInfo(title="A panel", description="Annotated geometry.")
    class AnnotatedPanel(Modello):
        width = gui(InstanceDummy("width", positive=True), title="Clear width", symbol="w", unit="m")
        height = gui(InstanceDummy("height", positive=True), title="Clear height", symbol="h", unit="m")
        area = gui(width * height, title="Surface", symbol="A", unit="m²")

    @ModelInfo(title="A project", highlights=("panel.area",), initial_field="panel.width")
    class Project(Modello):
        panel = gui(component(AnnotatedPanel), title="Geometry", description="A nested panel.")
        total = gui(panel.area * 10, title="Cost", unit="USD")

    data = Explorer(Project, inputs={"panel.width": 2, "panel.height": 3}).snapshot()
    assert data["title"] == "A project"
    assert data["highlights"] == ["panel.area"]
    assert data["components"][0] == {"path": "panel", "title": "Geometry", "description": "A nested panel."}
    assert rows(data)["panel.width"]["title"] == "Clear width"
    assert rows(data)["panel.area"]["unit"] == "m²"
    assert rows(data)["total"]["value"] == "60"
    assert FieldInfo.for_model(Project)["panel.width"].symbol == "w"
    assert Project.__modello_annotations__[ModelInfo]["panel"].title == "A panel"
    assert "__gui__" not in vars(Project) and "__gui__" not in vars(AnnotatedPanel)
    original = AnnotatedPanel("case", width=2, height=3)
    assert original.area == 6
    assert Project("case", panel=original).total == 60


def test_gui_metadata_and_scenario_arguments_require_names():
    with pytest.raises(TypeError):
        gui(InstanceDummy("value"), "A title")
    with pytest.raises(TypeError):
        FieldInfo("A title")
    with pytest.raises(TypeError):
        ModelInfo("A title")
    with pytest.raises(TypeError):
        Scenario("A question", "A description")
    scenario = Scenario(title="A question", description="A description", bindings={"value": 3}, select="value")
    assert scenario.bindings == {"value": 3}

    class Model(Modello):
        value = gui(field=InstanceDummy("value"), title="A title")

    assert FieldInfo.for_model(Model)["value"].title == "A title"


def test_gui_annotations_and_model_decorators_reject_duplicates():
    with pytest.raises(ModelDefinitionError, match="Duplicate FieldInfo annotation"):

        class DuplicateField(Modello):
            value = gui(gui(InstanceDummy("value"), title="Original"), title="Second")

    with pytest.raises(ModelDefinitionError, match="Duplicate ModelInfo annotation on DuplicateModel"):

        @ModelInfo(title="Second")
        @ModelInfo(title="Original")
        class DuplicateModel(Modello):
            value = InstanceDummy("value")


def test_inline_gui_hidden_metadata_still_protects_all_aliases():
    class Model(Modello):
        value = gui(InstanceDummy("value", positive=True), title="Background value", hidden=True)
        alias = value
        result = gui(value * 2, title="Visible result")

    client = create_app(Model, inputs={"value": 3}).test_client()
    data = client.post("/api/explore", json={}).json
    assert set(rows(data)) == {"result"}
    assert data["hidden_inputs"] == 1
    assert rows(data)["result"]["value"] == "6"
    for path in ("value", "alias"):
        assert client.post("/api/explore", json={"bindings": {path: None}}).status_code == 422


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


def test_field_provenance_distinguishes_calculations_from_recorded_observations():
    ordinary = rows(Explorer(Panel, inputs={"width": 2, "height": 3}).snapshot())
    assert ordinary["width"]["provenance"] == "Supplied input"
    assert (
        ordinary["area"]["kind"] == "derived" and ordinary["area"]["provenance"] == "Calculated from a model definition"
    )
    inverse = rows(Explorer(Panel, inputs={"area": 6, "height": 3}).snapshot())
    assert (
        inverse["width"]["kind"] == "derived"
        and inverse["width"]["provenance"] == "Inferred from the other observations"
    )
    copied = rows(Explorer(inspect_model(Estimate, "copy", panel=Panel("source", width=2, height=3))).snapshot())
    assert copied["panel.area"]["provenance"] == "Recorded component result"
    assert not copied["panel.area"]["editable"]


def test_empty_components_remain_in_the_raw_model_structure():
    class Empty(Modello):
        """No quantities yet."""

    class Root(Modello):
        empty = gui(component(Empty), title="Placeholder")

    data = Explorer(Root).snapshot(raw=True)
    assert not data["fields"]
    assert data["components"][0]["title"] == "Placeholder"


def test_python_implementation_is_not_served_as_a_frontend_asset():
    client = create_app(Panel).test_client()
    assert client.get("/assets/inputs.py").status_code == 404


def test_typeset_values_constraints_and_exact_units():
    explorer = Explorer(Panel, inputs={"height": 3}, metadata={"area": FieldInfo(symbol="A", unit="m²")})
    partial = rows(explorer.snapshot())
    assert partial["area"]["value_state"] == "symbolic"
    assert partial["area"]["depends_on"] == [{"path": "width", "title": "width"}]
    bound = rows(explorer.snapshot({"width": 2}))
    assert bound["area"]["display_value"] == "6" and bound["area"]["unit"] == "m²"
    assert not bound["area"]["approximate"]
    for name in ("symbol_mathml", "value_mathml", "exact_mathml"):
        ElementTree.fromstring(bound["area"][name])
    for constraint in bound["area"]["constraints"]:
        ElementTree.fromstring(constraint["mathml"])


def test_percent_editor_roundtrip_and_highlighted_target():
    @ModelInfo(
        initial_field="value",
        highlights=("doubled",),
        scenarios=(
            Scenario(
                title="Set 3%",
                description="Apply a known rate.",
                bindings={"value": Rational(3, 100)},
                select="doubled",
            ),
        ),
    )
    class Rate(Modello):
        value = gui(InstanceDummy("value", real=True), number_format="percent")
        doubled = gui(2 * value, number_format="percent")

    client = create_app(Rate).test_client()
    initial = client.post("/api/explore", json={}).json
    scenario = initial["scenarios"][0]
    assert scenario["bindings"] == {"value": "3%"}
    data = client.post("/api/explore", json={"bindings": scenario["bindings"]}).json
    assert data["initial_field"] == "value" and data["highlights"] == ["doubled"]
    assert rows(data)["value"]["input"] == "3%"
    assert rows(data)["doubled"]["display_value"] == "6%"
    assert rows(data)["doubled"]["kind"] == "derived"


@pytest.mark.parametrize(
    "options",
    [
        {"highlights": ("unknown",)},
        {"initial_field": "unknown"},
        {"scenarios": (Scenario(title="Hidden", description="Cannot change background data.", bindings={"price": 2}),)},
        {"scenarios": (Scenario(title="Unknown selection", description="Bad field.", select="unknown"),)},
        {"highlights": ("price",)},
    ],
)
def test_model_navigation_validates_paths_and_hidden_policy(options):
    @ModelInfo(**options)
    class Invalid(Estimate):
        pass

    with pytest.raises(ModelInputError):
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


def test_external_metadata_and_input_path_errors_have_context_and_suggestions():
    with pytest.raises(ModelInputError) as error:
        Explorer(Estimate, metadata={"panel.wdth": FieldInfo()})
    assert str(error.value) == "metadata: Unknown field path 'panel.wdth'. Did you mean 'panel.width'?"
    with pytest.raises(BindingError) as error:
        Explorer(Estimate, inputs={"panel.wdth": 2})
    assert str(error.value) == "inputs: Unknown field path 'panel.wdth'. Did you mean 'panel.width'?"
    assert error.value.paths == ("panel.wdth",)
    with pytest.raises(BindingError) as error:
        Explorer(Estimate).snapshot({"panel.wdth": 2})
    assert str(error.value) == "bindings: Unknown field path 'panel.wdth'. Did you mean 'panel.width'?"
    with pytest.raises(BindingError) as error:
        Explorer(Panel).snapshot({"unrelated": 2})
    assert str(error.value) == "bindings: Unknown field path 'unrelated'."


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


def test_quickstart_supports_forward_inverse_and_free_exploration():
    from examples.gui_quickstart import ReturnModel

    client = create_app(ReturnModel, inputs={"capital": 100_000}).test_client()
    initial = client.post("/api/explore", json={}).json
    assert rows(initial)["expected_return"]["kind"] == "free"
    assert initial["status"] == "conditional"
    bindings = initial["scenarios"][0]["bindings"]
    forward = client.post("/api/explore", json={"bindings": bindings}).json
    assert rows(forward)["gain"]["value"] == "8000"
    changed = client.post("/api/explore", json={"bindings": {**bindings, "capital": "200000"}}).json
    assert rows(changed)["gain"]["value"] == "16000"
    released = client.post("/api/explore", json={"bindings": {"expected_return": None}}).json
    assert rows(released)["expected_return"]["kind"] == "free"
    inverse = client.post("/api/explore", json={"bindings": {"gain": "8000"}}).json
    assert rows(inverse)["expected_return"]["display_value"] == "8%"
    assert rows(inverse)["expected_return"]["kind"] == "derived"


def test_percent_math_groups_the_whole_expression():
    class Rate(Modello):
        weight = InstanceDummy("weight", real=True)
        result = weight / 20 + Rational(3, 100)

    data = rows(Explorer(Rate, metadata={"result": FieldInfo(number_format="percent")}).snapshot())
    markup = data["result"]["value_mathml"]
    assert "<mo>(</mo>" in markup and "<mo>)</mo>" in markup
    ElementTree.fromstring(markup)


def test_numeric_formatting_retains_exact_values_and_marks_rounding():
    from modello._gui.formatting import display_number

    assert display_number(Rational(6500)) == ("6,500", False)
    assert display_number(Rational(13, 2), percent=True) == ("6.5%", False)
    assert display_number(sqrt(793) / 2, percent=True) == ("14.08%", True)
    assert display_number(Rational(1, 10000)) == ("0.0001", False)
    assert display_number(Symbol("x")) == ("", False)
    assert display_number(Rational(10) ** 1000)[1]


def test_portfolio_demo_starts_open_and_can_solve_forward_or_inverse():
    from examples.gui import DEMO_INPUTS, Portfolio

    explorer = Explorer(Portfolio, inputs=DEMO_INPUTS)
    initial = explorer.snapshot()
    assert initial["hidden_inputs"] == 2
    assert rows(initial)["allocation.equity"]["kind"] == "free"
    assert rows(initial)["expected_return"]["value_state"] == "symbolic"
    assert not any(field["kind"] == "fixed" for field in initial["fields"])
    split = explorer.snapshot({"allocation.equity": Rational(7, 10)})
    assert split["status"] == "checked"
    assert rows(split)["expected_return"]["value"] == "13/200"
    assert rows(split)["expected_gain"]["display_value"] == "6,500"
    assert rows(split)["volatility"]["display_value"] == "14.08%"
    assert rows(split)["volatility"]["approximate"]
    assert rows(split)["expected_return"]["kind"] == "derived"
    inverse = explorer.snapshot({"expected_return": Rational(6, 100)})
    assert inverse["status"] == "checked"
    assert rows(inverse)["expected_return"]["kind"] == "input"
    assert rows(inverse)["allocation.equity"]["display_value"] == "60%"
    assert rows(inverse)["allocation.bonds"]["display_value"] == "40%"
    with pytest.raises(BindingError):
        explorer.snapshot({"allocation.equity": Rational(6, 5)})
    assert rows(explorer.snapshot())["allocation.equity"]["kind"] == "free"


def test_invalid_portfolio_edit_identifies_both_input_and_derived_domain():
    from examples.gui import DEMO_INPUTS, Portfolio

    client = create_app(Portfolio, inputs=DEMO_INPUTS).test_client()
    response = client.post("/api/explore", json={"bindings": {"allocation.equity": "120%"}})
    assert response.status_code == 422
    assert set(response.json["paths"]) == {"allocation.equity", "allocation.bonds"}


def test_portfolio_missing_return_story_drops_and_reinfers_an_original_input():
    from examples.gui import DEMO_INPUTS, Portfolio

    client = create_app(Portfolio, inputs=DEMO_INPUTS).test_client()
    initial = client.post("/api/explore", json={}).json
    scenario = next(item for item in initial["scenarios"] if item["title"] == "Find a missing return")
    response = client.post("/api/explore", json={"bindings": scenario["bindings"]})
    assert response.status_code == 200
    assert response.json["status"] == "checked"
    field = rows(response.json)["market.equity.expected_return"]
    assert field["input_state"] == "dropped"
    assert field["kind"] == "derived"
    assert field["initial_input"] == "8%"
    assert field["input"] == ""
    assert field["value"] == "51/700"
    assert field["display_value"] == "7.29%"
    assert response.json["hidden_inputs"] == 2
    # Restore the assumption and drop the target together to return to forward calculation.
    restored = client.post("/api/explore", json={"bindings": {"allocation.equity": "70%"}}).json
    assert rows(restored)["market.equity.expected_return"]["input_state"] == "initial"
    assert rows(restored)["expected_return"]["display_value"] == "6.5%"


def test_portfolio_assumption_comparison_and_partial_unspecification():
    from examples.gui import DEMO_INPUTS, Portfolio

    explorer = Explorer(Portfolio, inputs=DEMO_INPUTS)
    compared = rows(
        explorer.snapshot({"allocation.equity": Rational(7, 10), "market.bonds.expected_return": Rational(1, 25)})
    )
    assert compared["expected_return"]["display_value"] == "6.8%"
    assert compared["market.bonds.expected_return"]["input_state"] == "overridden"
    restored = rows(explorer.snapshot({"allocation.equity": Rational(7, 10)}))
    assert restored["expected_return"]["display_value"] == "6.5%"
    assert restored["market.bonds.expected_return"]["input_state"] == "initial"
    unspecified = explorer.snapshot({"allocation.equity": Rational(7, 10), "market.equity.expected_return": None})
    assert unspecified["status"] == "conditional"
    assert rows(unspecified)["market.equity.expected_return"]["kind"] == "free"
    assert rows(unspecified)["market.equity.expected_return"]["input_state"] == "dropped"
    assert rows(unspecified)["expected_return"]["value_state"] == "symbolic"

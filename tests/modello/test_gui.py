"""GUI exploration and browser requests preserve observations and edit boundaries."""

import json
import subprocess
import sys
from xml.etree import ElementTree

import pytest
from sympy import Rational, Sum, Symbol

from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model, vector
from modello.diagnostics import ConstraintKind
from modello.errors import ModelDefinitionError, ModelInputError
from modello.gui import BindingError, Explorer, FieldInfo, ModelInfo, Scenario, create_app, gui
from tests.modello._gui_models import Estimate, Panel, Window, mathml_text, rows

from ._models import Reading


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
    assert rows(explorer.snapshot())["area"]["value"] == "20"


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


def test_snapshot_keeps_inferred_component_observations(copied_estimate_problem):
    problem = copied_estimate_problem
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
    with pytest.raises(BindingError, match="Multiple solutions"):
        Explorer(Reading, inputs={"squared": 4}).snapshot()


def test_external_parameters_stay_conditional():
    class External(Modello):
        x = InstanceDummy("x", real=True)
        value = x * Symbol("parameter")

    data = Explorer(External, inputs={"x": 2}).snapshot()
    assert data["status"] == "conditional"
    assert "external[parameter]" in rows(data)["value"]["value"]


def test_app_updates_are_isolated_and_invalid_edits_preserve_previous_model():
    client = create_app(Estimate, inputs={"panel.width": 4, "panel.height": 5, "price": 10}).test_client()
    for path, status in (
        ("/", 200),
        ("/assets/explorer.js", 200),
        ("/assets/explorer.css", 200),
        ("/assets/inputs.py", 404),
    ):
        assert client.get(path).status_code == status
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
    "options, status",
    [
        pytest.param({"json": []}, 400, id="array-body"),
        pytest.param({"json": {"x": 1}}, 400, id="unknown-option"),
        pytest.param({"json": {"raw": "yes"}}, 400, id="non-boolean-raw"),
        pytest.param({"json": {"bindings": []}}, 400, id="non-mapping-bindings"),
        pytest.param({"json": {"bindings": {"width": 2}}}, 422, id="non-text-binding"),
        pytest.param({"json": {"bindings": {"width": "__import__('os')"}}}, 422, id="invalid-arithmetic"),
        pytest.param({"data": "not json"}, 400, id="invalid-json"),
    ],
)
def test_app_rejects_malformed_requests(options, status):
    client = create_app(Panel).test_client()
    assert client.post("/api/explore", **options).status_code == status


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


def test_browser_cannot_edit_a_copied_inferred_observation(copied_estimate_problem):
    problem = copied_estimate_problem
    client = create_app(problem).test_client()
    response = client.post("/api/explore", json={"bindings": {"panel.area": "20"}})
    assert response.status_code == 422
    assert response.json["paths"] == ["panel.area"]


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
    assert data["components"][0] == {
        "path": "panel",
        "parent": "",
        "title": "Geometry",
        "description": "A nested panel.",
        "collapsed": False,
    }
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


def test_field_provenance_distinguishes_calculations_from_recorded_observations(copied_estimate_problem):
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
    copied = rows(Explorer(copied_estimate_problem).snapshot())
    assert copied["panel.area"]["provenance"] == "Recorded component result"
    assert not copied["panel.area"]["editable"]


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


def test_vector_inputs_partial_inverse_solve_and_per_entry_release():
    explorer = Explorer(Window, inputs={"samples": [1, None, 3]})
    initial = rows(explorer.snapshot())
    assert initial["samples[1]"]["kind"] == "free"
    inverse = explorer.snapshot({"total": 6})
    assert inverse["status"] == "checked"
    assert rows(inverse)["samples[1]"]["value"] == "2"
    assert rows(inverse)["growth"]["value"] == "24"
    assert rows(inverse, "vectors")["changes"]["value"] == "[1, 1]"
    released = rows(explorer.snapshot({"samples[0]": None}))
    assert released["samples[0]"]["input_state"] == "dropped"
    assert released["samples[2]"]["kind"] == "input"
    assert all(
        entry["kind"] == "free"
        for name, entry in rows(explorer.snapshot({"samples": None})).items()
        if name.startswith("samples[")
    )


def test_browser_can_edit_each_coordinate_and_rejects_invalid_edits():
    client = create_app(Window, inputs={"samples": [1, 2, 3]}).test_client()
    response = client.post("/api/explore", json={"bindings": {"samples[1]": "5/2"}})
    assert response.status_code == 200
    assert rows(response.json)["total"]["value"] == "13/2"
    assert rows(response.json)["samples[1]"]["input_state"] == "overridden"
    invalid = client.post("/api/explore", json={"bindings": {"samples[1]": "[1,2]"}})
    assert invalid.status_code == 422
    assert invalid.json["paths"] == ["samples[1]"]
    assert rows(client.post("/api/explore", json={}).json)["total"]["value"] == "6"


def test_coordinate_definitions_link_to_the_complete_formula_through_a_vector_alias():
    class Parent(Modello):
        window = component(Window)
        alias = window.changes

    data = Explorer(Parent).snapshot(raw=True)
    coordinate = rows(data)["alias[0]"]
    own = next(
        constraint
        for constraint in coordinate["constraints"]
        if constraint["kind"] == "definition" and constraint["path"] in coordinate["aliases"]
    )
    assert own["path"] == "window.changes[0]"
    assert own["equation_path"] == "window.changes"
    assert own["equation_path"] in rows(data, "equations")


def test_nested_vector_copy_can_be_reopened_without_turning_outputs_into_inputs():
    class Pair(Modello):
        window = component(Window)
        doubled = window.total * 2

    original = Pair("source", window=Window("source-window", samples=[1, 2, 3]))
    data = Explorer(original).snapshot({"window.samples[1]": 4})
    assert rows(data)["doubled"]["value"] == "16"
    assert original.window.samples == Vector(1, 2, 3)
    assert rows(data, "vectors")["window.samples"]["shape"] == [3]


def test_hidden_vector_entries_are_omitted_from_summaries_and_editors():
    explorer = Explorer(Window, inputs={"samples": [1, 1234567, 3]}, metadata={"samples[1]": FieldInfo(hidden=True)})
    data = explorer.snapshot()
    assert "samples[1]" not in rows(data)
    summary = rows(data, "vectors")["samples"]
    assert "1234567" not in json.dumps(summary)
    assert summary["entries"] == ["samples[0]", "samples[2]"]
    assert "samples[1]" in summary["value"]
    hidden = Explorer(Window, metadata={"samples": FieldInfo(hidden=True)}).snapshot(raw=True)
    assert "samples" not in rows(hidden, "vectors")
    assert not any(path.startswith("samples[") for path in rows(hidden))


def test_vector_metadata_escapes_literal_labels_and_inherits_percent_formatting():
    data = Explorer(
        Window,
        inputs={"samples": [Rational(1, 10), 0, 0]},
        metadata={"samples": FieldInfo(symbol='<x>"&', number_format="percent")},
    ).snapshot()
    summary = rows(data, "vectors")["samples"]
    assert '<x>"&' in mathml_text(summary["symbol_mathml"])
    assert ElementTree.fromstring(summary["symbol_mathml"]).find(".//x") is None
    assert rows(data)["samples[0]"]["display_value"] == "10%"
    assert mathml_text(summary["value_mathml"]) == "[1000]"


def test_vector_scenarios_are_flattened_to_safe_coordinate_edits():
    @ModelInfo(
        initial_field="samples",
        highlights=("samples", "total"),
        scenarios=(
            Scenario(
                title="A complete window",
                description="Observe all entries.",
                bindings={"samples": [1, 2, 3]},
                select="samples",
            ),
        ),
    )
    class Guided(Window):
        pass

    client = create_app(Guided).test_client()
    raw = client.post("/api/explore", json={"raw": True}).json
    scenario = raw["scenarios"][0]
    assert scenario["bindings"] == {"samples[0]": "1", "samples[1]": "2", "samples[2]": "3"}
    assert scenario["select"] == "samples"
    response = client.post("/api/explore", json={"bindings": scenario["bindings"]})
    assert response.status_code == 200
    assert rows(response.json)["total"]["value"] == "6"


def test_component_paths_are_not_valid_quantity_highlights():
    @ModelInfo(highlights=("window",))
    class Invalid(Modello):
        window = component(Window)

    with pytest.raises(ModelInputError, match="highlights"):
        Explorer(Invalid)


def test_whole_vector_edits_normalize_to_independent_coordinate_proposals():
    client = create_app(Window, inputs={"samples": [1, None, 3]}).test_client()
    initial = client.post("/api/explore", json={}).json
    summary = rows(initial, "vectors")["samples"]
    assert summary["kind"] == "mixed"
    assert summary["input"] == summary["initial_input"] == "[1, ?, 3]"
    assert summary["input_state"] == "initial"
    assert summary["editable"]
    response = client.post("/api/explore", json={"bindings": {"samples": "[1, 5/2, 3]"}})
    assert response.status_code == 200
    assert response.json["bindings"] == {"samples[0]": "1", "samples[1]": "5/2", "samples[2]": "3"}
    assert rows(response.json)["total"]["value"] == "13/2"
    next_bindings = {**response.json["bindings"], "samples[1]": "2"}
    assert rows(client.post("/api/explore", json={"bindings": next_bindings}).json)["total"]["value"] == "6"
    dropped = client.post("/api/explore", json={"bindings": {"samples": None}}).json
    assert all(value is None for value in dropped["bindings"].values())
    assert rows(dropped, "vectors")["samples"]["input_state"] == "dropped"


def test_bulk_edits_cannot_change_hidden_or_recorded_coordinates():
    client = create_app(
        Window, inputs={"samples": [1, 2, 3]}, metadata={"samples[1]": FieldInfo(hidden=True)}
    ).test_client()
    response = client.post("/api/explore", json={"bindings": {"samples": "[4, 5, 6]"}})
    assert response.status_code == 422
    assert response.json["paths"] == ["samples"]
    assert not rows(client.post("/api/explore", json={}).json, "vectors")["samples"]["editable"]

    class Parent(Modello):
        window = component(Window)

    problem = inspect_model(Parent, "recorded", window=Window("source", samples=[1, None, 3], total=6))
    recorded = create_app(problem).test_client()
    data = recorded.post("/api/explore", json={}).json
    assert rows(data)["window.samples[1]"]["kind"] == "fixed"
    assert not rows(data, "vectors")["window.samples"]["editable"]
    response = recorded.post("/api/explore", json={"bindings": {"window.samples": "[1,2,3]"}})
    assert response.status_code == 422


def test_vector_domain_validation_covers_each_entry_and_retains_applied_inputs():
    class Positive(Modello):
        samples = Vector("x", size=2, positive=True)
        total = samples.sum()

    client = create_app(Positive, inputs={"samples": [1, 2]}).test_client()
    response = client.post("/api/explore", json={"bindings": {"samples": "[1, -2]"}})
    assert response.status_code == 422
    assert "samples[1]" in response.json["paths"]
    assert rows(client.post("/api/explore", json={}).json)["total"]["value"] == "3"


def test_explicit_coordinate_metadata_replaces_inherited_vector_metadata():
    data = Explorer(
        Window, metadata={"samples": FieldInfo(unit="USD", number_format="percent"), "samples[0]": FieldInfo()}
    ).snapshot(raw=True)
    assert rows(data)["samples[0]"]["unit"] == ""
    assert rows(data)["samples[0]"]["number_format"] == "number"
    assert rows(data)["samples[0]"]["title"] == "samples[0]"
    assert rows(data)["samples[1]"]["unit"] == "USD"


def test_hidden_vector_symbols_remain_navigable_without_observation_values():
    data = Explorer(
        Window, inputs={"samples": [1, 1234567, 3]}, metadata={"samples": FieldInfo(hidden=True)}
    ).snapshot()
    symbol = rows(data, "symbols")["samples"]
    assert symbol["value_state"] == "hidden"
    assert "1234567" not in json.dumps(symbol)
    assert not symbol["entries"] and not symbol["editable"]
    assert 'data-path="samples"' in symbol["symbol_mathml"]


def test_empty_vectors_keep_their_own_labels_and_show_an_empty_definition():
    class Empty(Modello):
        first = Vector("a", size=0)
        second = vector(1, (Index("i"), 0, -1))

    data = Explorer(Empty).snapshot(raw=True)
    vectors = rows(data, "vectors")
    assert mathml_text(vectors["first"]["symbol_mathml"]) == "first→"
    assert mathml_text(vectors["second"]["symbol_mathml"]) == "second→"
    assert "∅" not in mathml_text(data["equations"][0]["mathml"])
    assert mathml_text(data["equations"][0]["mathml"]) == "second→=[]"


def test_derived_vector_dependencies_link_to_free_inputs_in_other_vectors():
    from examples.gui_timeseries import DEMOS

    model, inputs = DEMOS["inventory"]
    data = Explorer(model, inputs=inputs).snapshot()
    dependencies = rows(data, "vectors")["quantities"]["depends_on"]
    assert [item["path"] for item in dependencies] == ["fills[2]"]


def test_a_large_out_of_range_prefix_is_rejected_without_losing_the_initial_case():
    class Prefix(Modello):
        samples = Vector("x", size=3, real=True)
        count = InstanceDummy("count", integer=True, nonnegative=True)
        i = Index("i")
        total = Sum(samples[i], (i, 0, count - 1))

    client = create_app(Prefix, inputs={"samples": [1, 2, 3], "count": 3}).test_client()
    rejected = client.post("/api/explore", json={"bindings": {"count": "1e100"}})
    assert rejected.status_code == 422
    restored = client.post("/api/explore", json={"bindings": {}})
    assert restored.status_code == 200
    assert rows(restored.json)["total"]["value"] == "6"


def test_accepted_arithmetic_text_survives_repeated_browser_edits():
    client = create_app(Window).test_client()
    expression = "(1/3)**100"
    response = client.post("/api/explore", json={"bindings": {"samples": f"[{expression}, 2, 3]"}})
    assert response.status_code == 200
    assert response.json["bindings"]["samples[0]"] == expression
    assert rows(response.json)["samples[0]"]["input"] == expression
    seed = rows(response.json, "vectors")["samples"]["input"]
    assert expression in seed
    assert client.post("/api/explore", json={"bindings": response.json["bindings"]}).status_code == 200
    assert client.post("/api/explore", json={"bindings": {"samples": seed}}).status_code == 200


def test_scalar_binding_normalization_preserves_the_accepted_expression():
    class Scalar(Modello):
        value = gui(InstanceDummy("value"))

    client = create_app(Scalar).test_client()
    expression = "2**100 * 2**100"
    response = client.post("/api/explore", json={"bindings": {"value": expression}})
    assert response.status_code == 200
    assert response.json["bindings"]["value"] == expression
    assert rows(response.json)["value"]["input"] == expression
    assert client.post("/api/explore", json={"bindings": response.json["bindings"]}).status_code == 200


def test_large_bulk_edit_can_be_followed_by_an_individual_coordinate_edit():
    class Large(Modello):
        samples = Vector("x", size=1024, real=True)

    client = create_app(Large).test_client()
    bulk = "[" + ",".join(["1"] * 1024) + "]"
    response = client.post("/api/explore", json={"raw": True, "bindings": {"samples": bulk}})
    assert response.status_code == 200
    proposal = {**response.json["bindings"], "samples[0]": "2"}
    assert len(json.dumps(proposal)) > 16_384
    response = client.post("/api/explore", json={"raw": True, "bindings": proposal})
    assert response.status_code == 200
    assert rows(response.json)["samples[0]"]["input"] == "2"


def test_oversized_requests_return_a_json_validation_message():
    app = create_app(Window)
    response = app.test_client().post(
        "/api/explore", data=" " * app.config["MAX_CONTENT_LENGTH"] + "{}", content_type="application/json"
    )
    assert response.status_code == 413
    assert response.is_json
    assert "too large" in response.json["error"]


def test_reopening_an_instance_rejects_a_changed_vector_shape():
    original = Window("source", samples=[1, 2, 3])
    original.samples = Vector(1, 2)
    with pytest.raises(ModelInputError, match="samples"):
        Explorer(original)

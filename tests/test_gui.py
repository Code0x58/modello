"""Exploration preserves paths, input provenance, domains, and browser boundaries."""

import json
import subprocess
import sys
from xml.etree import ElementTree

import pytest
from sympy import Rational, Symbol, pi, sqrt

from modello import InstanceDummy, Modello, component, inspect_model
from modello._gui.inputs import parse_scalar
from modello.diagnostics import ConstraintKind
from modello.errors import ModelInputError
from modello.gui import BindingError, Explorer, FieldInfo, ModelInfo, Scenario, create_app


class Panel(Modello):
    """A rectangular panel."""

    width = InstanceDummy("width", positive=True)
    height = InstanceDummy("height", positive=True)
    area = width * height
    __gui__ = ModelInfo(fields={"width": FieldInfo(description="Width in metres.", symbol="w")})


class Estimate(Modello):
    panel = component(Panel)
    price = InstanceDummy("price", positive=True)
    total = panel.area * price
    __gui__ = ModelInfo(fields={"price": FieldInfo(symbol="p", hidden=True)})


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
    class Described(Modello):
        panel = component(Panel)
        __gui__ = ModelInfo(
            title="Panel estimate",
            description="A model description.",
            info="A model note.",
            fields={"panel": FieldInfo(title="Geometry"), "panel.height": FieldInfo(description="Height in metres.")},
        )

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


def test_model_info_rejects_ambiguous_declarations():
    class WrongType(Modello):
        __gui__ = {}

    class EmptyPath(Modello):
        __gui__ = ModelInfo(fields={"": FieldInfo(title="Title")})

    with pytest.raises(ModelInputError, match="ModelInfo declaration"):
        Explorer(WrongType)
    with pytest.raises(ModelInputError, match="empty field path"):
        Explorer(EmptyPath)


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
        empty = component(Empty)
        __gui__ = ModelInfo(fields={"empty": FieldInfo(title="Placeholder")})

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
    class Rate(Modello):
        value = InstanceDummy("value", real=True)
        doubled = 2 * value
        __gui__ = ModelInfo(
            initial_field="value",
            highlights=("doubled",),
            fields={"value": FieldInfo(number_format="percent"), "doubled": FieldInfo(number_format="percent")},
            scenarios=(Scenario("Set 3%", "Apply a known rate.", {"value": Rational(3, 100)}, select="doubled"),),
        )

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
        {"scenarios": (Scenario("Hidden", "Cannot change background data.", {"price": 2}),)},
        {"scenarios": (Scenario("Unknown selection", "Bad field.", select="unknown"),)},
        {"highlights": ("price",)},
    ],
)
def test_model_navigation_validates_paths_and_hidden_policy(options):
    class Invalid(Estimate):
        __gui__ = ModelInfo(fields={"price": FieldInfo(hidden=True)}, **options)

    with pytest.raises(ModelInputError):
        Explorer(Invalid)


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

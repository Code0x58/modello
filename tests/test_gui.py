"""Exploration preserves paths, input provenance, domains, and browser boundaries."""

import json
import subprocess
import sys
from xml.etree import ElementTree

import pytest
from sympy import Rational, Symbol, pi, sqrt

from modello import InstanceDummy, Modello, component, inspect_model
from modello.diagnostics import ConstraintKind
from modello.errors import ModelInputError
from modello.gui import BindingError, Explorer, FieldInfo, _parse_scalar, create_app


class Panel(Modello):
    """A rectangular panel."""

    width = InstanceDummy("width", positive=True)
    height = InstanceDummy("height", positive=True)
    area = width * height
    __gui__ = {"width": FieldInfo(description="Width in metres.", symbol="w")}


class Estimate(Modello):
    panel = component(Panel)
    price = InstanceDummy("price", positive=True)
    total = panel.area * price
    __gui__ = {"price": FieldInfo(symbol="p", hidden=True)}


def rows(data):
    return {row["path"]: row for row in data["fields"]}


def test_raw_never_solves(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("Raw rendering must not solve")

    monkeypatch.setattr("modello.gui.solve", unexpected)
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
    assert rows(bound)["total"]["kind"] == "fixed"
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
        ("3/2", Rational(3, 2)),
        ("sqrt(2)", sqrt(2)),
        ("pi", pi),
        ("2**3 - 1", 7),
        ("-2 + +3", 1),
    ],
)
def test_exact_arithmetic(text, value):
    assert _parse_scalar(text) == value


@pytest.mark.parametrize(
    "text",
    [
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
        _parse_scalar(text)


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

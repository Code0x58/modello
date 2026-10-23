"""Vector navigation and mathematical notation survive the full explorer flow."""

import json
from xml.etree import ElementTree as ET

import pytest
from sympy import Derivative, Function, IndexedBase, Integral, Limit, Product, Rational, Sum, Symbol

from modello import Index, InstanceDummy, Modello, Vector, component, inspect_model, vector
from modello._gui.inputs import parse_vector
from modello._gui.mathml import EquationPrinter
from modello.errors import ModelInputError
from modello.gui import Explorer, FieldInfo, ModelInfo, Scenario, create_app, gui

NS = {"m": "http://www.w3.org/1998/Math/MathML"}


class Window(Modello):
    samples = gui(Vector("x", size=3, real=True), title="Observations", symbol="x", unit="m")
    i = Index("i")
    total = samples.sum(i)
    growth = Product(1 + samples[i], (i, 0, 2))
    changes = gui(vector(samples[i] - samples[i - 1], (i, 1, 2)), symbol="Δx")


def by_path(snapshot, key="fields"):
    return {entry["path"]: entry for entry in snapshot[key]}


def text(markup):
    return "".join(ET.fromstring(markup).itertext())


def test_raw_vector_formulas_and_symbols_are_valid_navigable_mathml():
    data = Explorer(Window).snapshot(raw=True)
    equations = by_path(data, "equations")
    vectors = by_path(data, "vectors")
    assert set(equations) == {"total", "growth", "changes"}
    assert "∑" in text(equations["total"]["mathml"])
    assert "∏" in text(equations["growth"]["mathml"])
    assert " for i∈{1,2}" in text(equations["changes"]["mathml"])
    assert ET.fromstring(vectors["samples"]["symbol_mathml"]).find("m:mover", NS) is not None
    assert vectors["samples"]["shape"] == [3]
    assert vectors["samples"]["entries"] == ["samples[0]", "samples[1]", "samples[2]"]
    assert by_path(data)["samples[1]"]["parent"] == "samples"
    assert by_path(data)["samples[1]"]["unit"] == "m"
    for item in data["fields"] + data["vectors"]:
        for key in ("symbol_mathml", "value_mathml", "exact_mathml"):
            root = ET.fromstring(item[key])
            assert root.tag.endswith("math")
    selected = ET.fromstring(equations["total"]["mathml"]).find(".//m:msub", NS)
    assert selected.get("data-path") == "samples"


def test_vector_inputs_partial_inverse_solve_and_per_entry_release():
    explorer = Explorer(Window, inputs={"samples": [1, None, 3]})
    initial = by_path(explorer.snapshot())
    assert initial["samples[1]"]["kind"] == "free"
    inverse = explorer.snapshot({"total": 6})
    assert inverse["status"] == "checked"
    assert by_path(inverse)["samples[1]"]["value"] == "2"
    assert by_path(inverse)["growth"]["value"] == "24"
    assert by_path(inverse, "vectors")["changes"]["value"] == "[1, 1]"
    released = by_path(explorer.snapshot({"samples[0]": None}))
    assert released["samples[0]"]["input_state"] == "dropped"
    assert released["samples[2]"]["kind"] == "input"
    assert all(
        entry["kind"] == "free"
        for name, entry in by_path(explorer.snapshot({"samples": None})).items()
        if name.startswith("samples[")
    )


def test_browser_can_edit_each_coordinate_and_rejects_invalid_edits():
    client = create_app(Window, inputs={"samples": [1, 2, 3]}).test_client()
    response = client.post("/api/explore", json={"bindings": {"samples[1]": "5/2"}})
    assert response.status_code == 200
    assert by_path(response.json)["total"]["value"] == "13/2"
    assert by_path(response.json)["samples[1]"]["input_state"] == "overridden"
    invalid = client.post("/api/explore", json={"bindings": {"samples[1]": "[1,2]"}})
    assert invalid.status_code == 422
    assert invalid.json["paths"] == ["samples[1]"]
    assert by_path(client.post("/api/explore", json={}).json)["total"]["value"] == "6"


def test_nested_vector_copy_can_be_reopened_without_turning_outputs_into_inputs():
    class Pair(Modello):
        window = component(Window)
        doubled = window.total * 2

    original = Pair("source", window=Window("source-window", samples=[1, 2, 3]))
    data = Explorer(original).snapshot({"window.samples[1]": 4})
    assert by_path(data)["doubled"]["value"] == "16"
    assert original.window.samples == Vector(1, 2, 3)
    assert by_path(data, "vectors")["window.samples"]["shape"] == [3]


def test_hidden_vector_entries_are_omitted_from_summaries_and_editors():
    explorer = Explorer(Window, inputs={"samples": [1, 1234567, 3]}, metadata={"samples[1]": FieldInfo(hidden=True)})
    data = explorer.snapshot()
    assert "samples[1]" not in by_path(data)
    summary = by_path(data, "vectors")["samples"]
    assert "1234567" not in json.dumps(summary)
    assert summary["entries"] == ["samples[0]", "samples[2]"]
    assert "samples[1]" in summary["value"]
    hidden = Explorer(Window, metadata={"samples": FieldInfo(hidden=True)}).snapshot(raw=True)
    assert "samples" not in by_path(hidden, "vectors")
    assert not any(path.startswith("samples[") for path in by_path(hidden))


def test_vector_metadata_escapes_literal_labels_and_inherits_percent_formatting():
    data = Explorer(
        Window,
        inputs={"samples": [Rational(1, 10), 0, 0]},
        metadata={"samples": FieldInfo(symbol='<x>"&', number_format="percent")},
    ).snapshot()
    summary = by_path(data, "vectors")["samples"]
    assert '<x>"&' in text(summary["symbol_mathml"])
    assert ET.fromstring(summary["symbol_mathml"]).find(".//x") is None
    assert by_path(data)["samples[0]"]["display_value"] == "10%"
    assert text(summary["value_mathml"]) == "[1000]"


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
    assert by_path(response.json)["total"]["value"] == "6"


def test_component_paths_are_not_valid_quantity_highlights():
    @ModelInfo(highlights=("window",))
    class Invalid(Modello):
        window = component(Window)

    with pytest.raises(ModelInputError, match="highlights"):
        Explorer(Invalid)


def test_multiple_sum_and_product_limits_are_all_rendered():
    x, y = Symbol("x"), Symbol("y")
    printer = EquationPrinter({})
    for operator, glyph in ((Sum, "∑"), (Product, "∏")):
        root = ET.fromstring(printer.expression(operator(x + y, (x, 0, 2), (y, 1, 3))))
        limits = root.findall(".//m:munderover", NS)
        assert len(limits) == 2
        assert ["".join(limit[1].itertext()) for limit in limits] == ["y=1", "x=0"]
        assert ["".join(limit[0].itertext()) for limit in limits] == [glyph, glyph]


def test_integral_differentials_keep_the_order_of_integration():
    x, y = Symbol("x"), Symbol("y")
    root = ET.fromstring(EquationPrinter({}).expression(Integral(x + y, (x, 0, y), (y, 0, 1))))
    assert [child.text for child in root[0][-4:]] == ["d", "x", "d", "y"]
    assert len(root.findall(".//m:msubsup", NS)) == 2


def test_derivative_powers_belong_on_the_variables_in_the_denominator():
    x = Symbol("x")
    root = ET.fromstring(EquationPrinter({}).expression(Derivative(Function("f")(x), (x, 2))))
    denominator = root.find("m:mfrac/m:mrow[2]", NS)
    assert denominator[0].text == "d"
    assert denominator[1].tag.endswith("msup")
    assert denominator[1][0].text == "x"
    assert denominator[1][1].text == "2"


def test_one_sided_limit_renders_its_direction():
    x = Symbol("x")
    markup = EquationPrinter({}).expression(Limit(1 / x, x, 0, dir="-"))
    assert "limx→0-" in text(markup)


def test_whole_vector_edits_normalize_to_independent_coordinate_proposals():
    client = create_app(Window, inputs={"samples": [1, None, 3]}).test_client()
    initial = client.post("/api/explore", json={}).json
    summary = by_path(initial, "vectors")["samples"]
    assert summary["kind"] == "mixed"
    assert summary["input"] == summary["initial_input"] == "[1, ?, 3]"
    assert summary["input_state"] == "initial"
    assert summary["editable"]
    response = client.post("/api/explore", json={"bindings": {"samples": "[1, 5/2, 3]"}})
    assert response.status_code == 200
    assert response.json["bindings"] == {"samples[0]": "1", "samples[1]": "5/2", "samples[2]": "3"}
    assert by_path(response.json)["total"]["value"] == "13/2"
    next_bindings = {**response.json["bindings"], "samples[1]": "2"}
    assert by_path(client.post("/api/explore", json={"bindings": next_bindings}).json)["total"]["value"] == "6"
    dropped = client.post("/api/explore", json={"bindings": {"samples": None}}).json
    assert all(value is None for value in dropped["bindings"].values())
    assert by_path(dropped, "vectors")["samples"]["input_state"] == "dropped"


def test_bulk_edits_cannot_change_hidden_or_recorded_coordinates():
    client = create_app(
        Window, inputs={"samples": [1, 2, 3]}, metadata={"samples[1]": FieldInfo(hidden=True)}
    ).test_client()
    response = client.post("/api/explore", json={"bindings": {"samples": "[4, 5, 6]"}})
    assert response.status_code == 422
    assert response.json["paths"] == ["samples"]
    assert not by_path(client.post("/api/explore", json={}).json, "vectors")["samples"]["editable"]

    class Parent(Modello):
        window = component(Window)

    problem = inspect_model(Parent, "recorded", window=Window("source", samples=[1, None, 3], total=6))
    recorded = create_app(problem).test_client()
    data = recorded.post("/api/explore", json={}).json
    assert by_path(data)["window.samples[1]"]["kind"] == "fixed"
    assert not by_path(data, "vectors")["window.samples"]["editable"]
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
    assert by_path(client.post("/api/explore", json={}).json)["total"]["value"] == "3"


def test_explicit_coordinate_metadata_replaces_inherited_vector_metadata():
    data = Explorer(
        Window, metadata={"samples": FieldInfo(unit="USD", number_format="percent"), "samples[0]": FieldInfo()}
    ).snapshot(raw=True)
    assert by_path(data)["samples[0]"]["unit"] == ""
    assert by_path(data)["samples[0]"]["number_format"] == "number"
    assert by_path(data)["samples[0]"]["title"] == "samples[0]"
    assert by_path(data)["samples[1]"]["unit"] == "USD"


@pytest.mark.parametrize(
    "raw",
    [
        "[1,2]",
        "[1,2,3,4]",
        "[1,,3]",
        "[x,2,3]",
        "[1,True,3]",
        "[1, __import__('os'), 3]",
        "1,2,3",
        "[1, (2**100)**100, 3]",
    ],
)
def test_vector_parser_reuses_scalar_limits_and_rejects_code(raw):
    with pytest.raises(ModelInputError):
        parse_vector(raw, 3)


def test_vector_parser_supports_exact_percentages_and_explicit_missing_entries():
    assert parse_vector("[10%, ?, 3/2]", 3) == (Rational(1, 10), None, Rational(3, 2))
    assert parse_vector("[]", 0) == ()


def test_hidden_vector_symbols_remain_navigable_without_observation_values():
    data = Explorer(
        Window, inputs={"samples": [1, 1234567, 3]}, metadata={"samples": FieldInfo(hidden=True)}
    ).snapshot()
    symbol = by_path(data, "symbols")["samples"]
    assert symbol["value_state"] == "hidden"
    assert "1234567" not in json.dumps(symbol)
    assert not symbol["entries"] and not symbol["editable"]
    assert 'data-path="samples"' in symbol["symbol_mathml"]


def test_empty_vectors_keep_their_own_labels_and_show_an_empty_definition():
    class Empty(Modello):
        first = Vector("a", size=0)
        second = vector(1, (Index("i"), 0, -1))

    data = Explorer(Empty).snapshot(raw=True)
    vectors = by_path(data, "vectors")
    assert text(vectors["first"]["symbol_mathml"]) == "first→"
    assert text(vectors["second"]["symbol_mathml"]) == "second→"
    assert "∅" not in text(data["equations"][0]["mathml"])
    assert text(data["equations"][0]["mathml"]) == "second→=[]"


def test_native_indexed_vectors_and_multivariate_calculus_are_well_formed():
    x, y = Symbol("x"), Symbol("y")
    array = IndexedBase("a", shape=(3,))
    printer = EquationPrinter({})
    assert ET.fromstring(printer.expression(array)).find("m:mover", NS) is not None
    assert ET.fromstring(printer.expression(array[x])).find("m:msub/m:mi", NS).text == "a"
    derivative = ET.fromstring(printer.expression(Derivative(Function("f")(x, y), x, y)))
    assert "∂" in "".join(derivative.itertext())
    for expression in (Integral(x, x), Integral(x, (x, y)), Limit(1 / x, x, 0, dir="+")):
        ET.fromstring(printer.expression(expression))


def test_temporary_vector_formulas_render_as_their_selected_scalar_expression():
    class Scaled(Modello):
        samples = Vector("x", size=3, real=True)
        total = (2 * samples).sum()

    data = Explorer(Scaled).snapshot(raw=True)
    markup = data["equations"][0]["mathml"]
    assert "∑" in text(markup)
    assert " for " not in text(markup)
    assert "mtable" not in markup
    assert "samples" in text(markup)


def test_derived_vector_dependencies_link_to_free_inputs_in_other_vectors():
    from examples.gui_timeseries import DEMOS

    model, inputs = DEMOS["inventory"]
    data = Explorer(model, inputs=inputs).snapshot()
    dependencies = by_path(data, "vectors")["quantities"]["depends_on"]
    assert [item["path"] for item in dependencies] == ["fills[2]"]


def test_multiplication_uses_a_visible_dot_between_literal_quantity_names():
    a, b = Symbol("opening"), Symbol("growth")
    rendered = EquationPrinter({}).expression(a * b)
    assert "·" in text(rendered)


@pytest.mark.parametrize("demo", ["inventory", "returns", "rolling", "exposure", "trend"])
def test_timeseries_guided_scenarios_solve_and_retain_valid_formula_markup(demo):
    from examples.gui_timeseries import DEMOS

    model, inputs = DEMOS[demo]
    explorer = Explorer(model, inputs=inputs)
    initial = explorer.snapshot()
    assert initial["fields"]
    for scenario in explorer._layout.model_info.scenarios:
        data = explorer.snapshot(scenario.bindings)
        for row in [*data["fields"], *data["vectors"]]:
            for constraint in row["constraints"]:
                ET.fromstring(constraint["mathml"])
        if scenario.title == "Fit the stationary line":
            fields = by_path(data)
            assert fields["slope"]["value"] == fields["intercept"]["value"] == "9/10"
            assert "∑" in text(fields["squared_error"]["constraints"][0]["mathml"])
            assert "d" in text(fields["slope_gradient"]["constraints"][0]["mathml"])


def test_accepted_arithmetic_text_survives_repeated_browser_edits():
    client = create_app(Window).test_client()
    expression = "(1/3)**100"
    response = client.post("/api/explore", json={"bindings": {"samples": f"[{expression}, 2, 3]"}})
    assert response.status_code == 200
    assert response.json["bindings"]["samples[0]"] == expression
    assert by_path(response.json)["samples[0]"]["input"] == expression
    seed = by_path(response.json, "vectors")["samples"]["input"]
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
    assert by_path(response.json)["value"]["input"] == expression
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
    assert by_path(response.json)["samples[0]"]["input"] == "2"


def test_vector_text_budget_scales_with_its_trusted_shape():
    literal = "12345678901234567890123456789012"
    text = "[" + ", ".join([literal] * 600) + "]"
    assert len(text) > 16_000
    values = parse_vector(text, 600)
    assert len(values) == 600
    assert values[0] == values[-1] == int(literal)


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

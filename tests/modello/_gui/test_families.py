"""User journeys from open questions to alternatives, bounds, and checked sections."""

import json

import pytest
from sympy import Eq, Rational, S, Symbol

from examples.gui import DEMO_INPUTS, Portfolio
from examples.solution_families import ImpactStudy
from modello import InstanceDummy, Modello, inspect_model
from modello.errors import ModelInputError
from modello.gui import BindingError, Explorer, FieldInfo, ModelInfo, PlotInfo, Scenario, create_app


class Response(Modello):
    x = InstanceDummy("x", real=True)
    y = InstanceDummy("y", real=True)
    total = x + y


def rows(data):
    return {field["path"]: field for field in data["fields"]}


def test_risk_question_compare_preview_narrow_and_remove_bound():
    client = create_app(Portfolio, inputs=DEMO_INPUTS).test_client()
    question = {"bindings": {"volatility": "5%"}}
    result = client.post("/api/explore", json=question).json
    assert result["status"] == "alternatives"
    family = result["exploration"]
    assert family["case_count"] == 2 and family["coordinate_count"] == 1
    assert family["selection"] is None
    assert {case["values"]["allocation.equity"]["value"] for case in family["cases"]} == {"0", "2/17"}
    assert "expected_return" in family["suggestions"]
    preview = client.post("/api/explore", json={**question, "case": 1}).json
    assert preview["status"] == "checked"
    assert preview["exploration"]["case_count"] == 2
    assert rows(preview)["allocation.equity"]["input_state"] == "unspecified"
    assert rows(preview)["allocation.equity"]["possible_values"]["kind"] == "finite"
    narrowed = client.post("/api/explore", json={**question, "bounds": {"expected_return": ["3.5%", None]}}).json
    assert narrowed["status"] == "checked"
    assert narrowed["exploration"]["case_count"] == 1
    assert rows(narrowed)["allocation.equity"]["value"] == "2/17"
    assert client.post("/api/explore", json=question).json["exploration"]["case_count"] == 2


def test_coordinate_count_reduces_observations_and_deduplicates_aliases():
    assert Explorer(Response).snapshot()["exploration"]["kind"] == "region"
    assert Explorer(Response, inputs={"y": 3}).snapshot()["exploration"]["kind"] == "curve"
    closed = Explorer(Response, inputs={"x": 2, "y": 3}).snapshot()
    assert closed["exploration"]["coordinate_count"] == 0
    assert closed["status"] == "checked"

    class Alias(Modello):
        x = InstanceDummy("x", real=True)
        other = x
        squared = x**2

    assert Explorer(Alias).snapshot()["exploration"]["coordinate_count"] == 1
    inverse = Explorer(Alias, inputs={"squared": 4}).snapshot()
    assert inverse["exploration"]["kind"] == "cases"
    # A coordinate can have two discrete values; it is not a dimension certificate.
    assert inverse["exploration"]["coordinate_count"] == 1


def test_open_filtered_family_keeps_filters_and_section_context_when_editing():
    class Square(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    problem = inspect_model(Square, "positive", squared=4)
    family = problem.solutions().where(problem.symbols.x > 0)
    explorer = Explorer(family)
    assert rows(explorer.snapshot())["x"]["value"] == "2"
    changed = explorer.snapshot({"squared": 9})
    assert rows(changed)["x"]["value"] == "3"
    assert changed["exploration"]["conditions"]
    assert family.one().x == 2
    assert explorer.snapshot(bounds={"x": (None, -1)})["status"] == "empty"


def test_calibration_ranges_predictions_and_point_selection_retain_whole_family():
    explorer = Explorer(ImpactStudy.calibrated())
    data = explorer.snapshot()
    assert rows(data)["impact"]["possible_values"]["exact"] == "Interval(1/4, 3/4)"
    assert rows(data)["planned.cost"]["possible_values"]["text"] == "[1,026, 1,076]"
    selected = explorer.snapshot(point={"impact": Rational(1, 2)})
    assert rows(selected)["planned.cost"]["value"] == "1051"
    # A shared, unspecified component quantity is inferred in this preview, not recorded.
    assert rows(selected)["impact"]["kind"] == "derived"
    assert selected["exploration"]["coordinate_count"] == 1
    assert selected["exploration"]["selection"]["coordinates"] == {"impact": "1/2"}
    json.dumps(selected)
    assert explorer.snapshot()["status"] == "conditional"


def test_replacing_a_target_does_not_accumulate_incompatible_observations():
    explorer = Explorer(Portfolio, inputs={**DEMO_INPUTS, "expected_return": Rational(3, 50)})
    assert rows(explorer.snapshot())["allocation.equity"]["value"] == "3/5"
    assert rows(explorer.snapshot({"expected_return": Rational(7, 100)}))["allocation.equity"]["value"] == "4/5"
    assert explorer.snapshot({"expected_return": Rational(9, 100)})["status"] == "empty"


def test_point_preview_is_a_checked_witness_and_never_an_observation():
    explorer = Explorer(Response)
    checked = explorer.snapshot(point={"x": 2, "y": 3})
    assert checked["status"] == "checked" and rows(checked)["total"]["value"] == "5"
    assert checked["exploration"]["coordinate_count"] == 2
    assert all(field["kind"] != "input" for field in checked["fields"])
    assert explorer.snapshot()["status"] == "conditional"
    with pytest.raises(ModelInputError, match="every remaining coordinate"):
        explorer.snapshot(point={"x": 2})


def test_one_coordinate_samples_filters_domains_and_exact_clickthrough():
    problem = inspect_model(Response, "negative", y=3)
    family = problem.solutions().where(problem.symbols.x < 0)
    client = create_app(family).test_client()
    response = client.post(
        "/api/plot",
        json={
            "controls": ["x"],
            "output": "total",
            "windows": {"x": ["-1", "1"]},
        },
    )
    assert response.status_code == 200
    data = response.json
    assert len(data["points"]) == 20 and data["rejected"] == 21
    assert data["unknown"] == 0 and not data["section"]
    assert all(point["axes"][0] < 0 for point in data["points"])
    selected = client.post("/api/explore", json={"point": data["points"][7]["point"]}).json
    assert selected["status"] == "checked"
    assert selected["exploration"]["conditions"]
    assert rows(client.post("/api/explore", json={}).json)["x"]["kind"] == "free"


def test_two_coordinate_chart_retains_coupled_constraints():
    problem = inspect_model(Response, "coupled")
    family = problem.solutions().where(Eq(problem.symbols.x * problem.symbols.y, 1))
    explorer = Explorer(family)
    summary = explorer.snapshot()["exploration"]
    assert summary["coordinate_count"] == 2 and summary["kind"] == "region"
    chart = explorer.plot(controls=["x", "y"], output="total", windows={"x": (-1, 1), "y": (-1, 1)})
    assert chart["samples"] == 441 and chart["grid_size"] == 21
    assert len(chart["points"]) == 2
    assert chart["rejected"] == 439
    assert all(point["axes"][0] * point["axes"][1] == 1 for point in chart["points"])


def test_many_coordinates_require_a_checked_reference_for_a_section():
    class Volume(Modello):
        x = InstanceDummy("x", real=True)
        y = InstanceDummy("y", real=True)
        z = InstanceDummy("z", real=True)
        total = x + y + z

    explorer = Explorer(Volume)
    assert explorer.snapshot()["exploration"]["kind"] == "section"
    options = {"controls": ["x"], "output": "total", "windows": {"x": (0, 2)}}
    with pytest.raises(ModelInputError, match="reference"):
        explorer.plot(**options)
    chart = explorer.plot(**options, point={"x": 1, "y": 2, "z": 3})
    assert chart["section"] and len(chart["points"]) == 41
    assert chart["fixed"]["y"]["value"] == "2" and chart["fixed"]["z"]["value"] == "3"
    assert chart["points"][0]["axes"] == [0, 5]
    assert chart["points"][-1]["point"] == {"x": "2", "y": "2", "z": "3"}
    assert explorer.snapshot(point=chart["points"][-1]["point"])["status"] == "checked"


def test_undecided_samples_and_zero_coordinates_do_not_claim_infeasibility():
    class Conditional(Modello):
        x = InstanceDummy("x", real=True)
        output = x * Symbol("external")

    explorer = Explorer(Conditional)
    chart = explorer.plot(controls=["x"], output="output", windows={"x": (1, 2)})
    assert not chart["points"] and chart["unknown"] == 41
    assert chart["rejected"] == 0
    snapshot = explorer.snapshot({"x": 2})
    assert snapshot["exploration"]["coordinate_count"] == 0
    assert snapshot["status"] == "conditional"


def test_large_finite_domain_is_not_eagerly_enumerated(monkeypatch):
    from modello import SolutionFamily

    class Discrete(Modello):
        x = InstanceDummy("x", integer=True)

    problem = inspect_model(Discrete, "large")
    family = problem.solutions().where(problem.symbols.x >= 0, problem.symbols.x <= 100)
    monkeypatch.setattr(SolutionFamily, "cases", lambda self: pytest.fail("Large sets must not be materialised"))
    result = Explorer(family).snapshot()
    assert result["exploration"]["kind"] == "finite"
    assert result["exploration"]["candidate_count"] == 101
    assert result["exploration"]["case_count"] is None
    assert rows(result)["x"]["possible_values"]["kind"] == "finite"


def test_exact_range_aliases_and_invalid_edits_are_atomic():
    class Alias(Modello):
        x = InstanceDummy("x", real=True)
        other = x

    client = create_app(Alias).test_client()
    data = client.post("/api/explore", json={"bounds": {"other": ["0.25", "3/4"]}}).json
    assert rows(data)["x"]["bounds"] == ["1/4", "3/4"]
    assert rows(data)["other"]["possible_values"]["text"] == "[0.25, 0.75]"
    assert client.post("/api/explore", json={"bounds": {"x": ["2", "1"]}}).status_code == 422
    assert client.post("/api/explore", json={"bounds": {"x": ["0", "1"], "other": ["0", "2"]}}).status_code == 422
    assert rows(client.post("/api/explore", json={}).json)["x"]["bounds"] == ["", ""]


@pytest.mark.parametrize(
    "payload",
    [
        {"bounds": []},
        {"bounds": {"x": ["1"]}},
        {"bounds": {"x": ["__import__('os')", None]}},
        {"case": True},
        {"case": -1},
        {"case": 999},
        {"point": []},
        {"point": {"x": None}},
    ],
)
def test_family_request_validation(payload):
    response = create_app(Response).test_client().post("/api/explore", json=payload)
    assert response.status_code in (400, 422)


def test_hidden_values_and_recorded_observations_remain_uneditable():
    from tests.modello._gui_models import Estimate, Panel

    problem = inspect_model(Estimate, "record", panel=Panel("panel", width=2, height=3), price=1234567)
    client = create_app(problem).test_client()
    raw = client.post("/api/explore", json={"raw": True}).json
    assert "1234567" not in json.dumps(raw)
    assert client.post("/api/explore", json={"bounds": {"price": ["0", "10"]}}).status_code == 422
    # Filtering a recorded result intersects it; it cannot release or replace the observation.
    infeasible = client.post("/api/explore", json={"bounds": {"panel.area": ["0", "5"]}}).json
    assert infeasible["status"] == "empty"
    assert rows(client.post("/api/explore", json={}).json)["panel.area"]["value"] == "6"
    hidden = create_app(Response, metadata={"x": FieldInfo(hidden=True)}).test_client()
    assert hidden.post("/api/explore", json={"point": {"x": "1", "y": "2"}}).status_code == 422


def test_raw_family_view_never_reduces_and_scenarios_include_bounds(monkeypatch):
    family = ImpactStudy.calibrated()
    explorer = Explorer(family)
    monkeypatch.setattr(Explorer, "_family_view", lambda *args: pytest.fail("Raw view must not reduce"))
    assert explorer.snapshot(raw=True)["exploration"] is None

    @ModelInfo(scenarios=(Scenario(title="Bound", description="A bounded question", bounds={"x": (0, 1)}),))
    class Guided(Response):
        pass

    scenario = Explorer(Guided).snapshot(raw=True)["scenarios"][0]
    assert scenario["bounds"] == {"x": ["0", "1"]}
    with pytest.raises(ModelInputError, match="bounds"):
        Explorer(Guided, metadata={"x": FieldInfo(hidden=True)})


def test_unsupported_family_scope_keeps_existing_numeric_preview():
    data = Explorer(Response, inputs={"x": 0.5, "y": 1.5}).snapshot()
    assert data["exploration"]["kind"] == "unsupported"
    with pytest.raises(BindingError):
        Explorer(Response).snapshot({"x": "not a scalar"})


def test_plot_presets_use_display_units_and_raw_view_does_not_solve(monkeypatch):
    from modello import SolutionFamily

    plot = PlotInfo(controls=("x",), output="squared", windows={"x": (Rational(1, 4), Rational(3, 4))})

    @ModelInfo(plot=plot, scenarios=(Scenario(title="Curve", description="A curved response", plot=plot),))
    class Percent(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    monkeypatch.setattr(SolutionFamily, "_prepare", lambda *args: pytest.fail("Raw presets must not solve"))
    data = Explorer(Percent, metadata={"x": FieldInfo(number_format="percent")}).snapshot(raw=True)
    assert data["plot"] == data["scenarios"][0]["plot"]
    assert data["plot"]["windows"] == {"x": ["25%", "75%"]}


@pytest.mark.parametrize(
    ("plot", "message"),
    [
        (PlotInfo(controls=(), output="total", windows={}), "one or two distinct"),
        (PlotInfo(controls=("x", "x"), output="total", windows={"x": (0, 1)}), "one or two distinct"),
        (PlotInfo(controls=("typo",), output="total", windows={"typo": (0, 1)}), "Unknown field path"),
        (PlotInfo(controls=("x",), output="total", windows={}), "window endpoints"),
        (PlotInfo(controls=("x",), output="total", windows={"x": (1, 0)}), "increasing finite real"),
        (PlotInfo(controls=("x",), output="total", windows={"x": (0, S.Infinity)}), "increasing finite real"),
    ],
)
def test_invalid_plot_presets_fail_at_authoring(plot, message):
    @ModelInfo(plot=plot)
    class Invalid(Response):
        pass

    with pytest.raises(ModelInputError, match=message):
        Explorer(Invalid)


def test_plot_presets_respect_hidden_quantities_and_aliases():
    @ModelInfo(plot=PlotInfo(controls=("x",), output="total", windows={"x": (0, 1)}))
    class Guided(Response):
        pass

    with pytest.raises(ModelInputError, match="visible"):
        Explorer(Guided, metadata={"x": FieldInfo(hidden=True)})

    @ModelInfo(plot=PlotInfo(controls=("x", "alias"), output="squared", windows={"x": (0, 1), "alias": (0, 1)}))
    class Alias(Modello):
        x = InstanceDummy("x", real=True)
        alias = x
        squared = x**2

    with pytest.raises(ModelInputError, match="rather than aliases"):
        Explorer(Alias)


def test_highlighted_alias_is_used_for_alternatives_and_comparison():
    @ModelInfo(highlights=("other",))
    class Alias(Modello):
        x = InstanceDummy("x", real=True)
        other = x
        squared = x**2

    result = Explorer(Alias, inputs={"squared": 4}).snapshot()
    assert result["exploration"]["coordinate_count"] == 1
    assert {case["values"]["other"]["value"] for case in result["exploration"]["cases"]} == {"-2", "2"}
    assert "x" not in result["exploration"]["cases"][0]["values"]


def test_disconnected_values_and_display_windows_remain_exact():
    class Square(Modello):
        x = InstanceDummy("x", real=True)
        squared = x**2

    snapshot = Explorer(Square).snapshot(bounds={"squared": (4, 9)})
    possible = rows(snapshot)["x"]["possible_values"]
    assert possible["kind"] == "union" and possible["text"] == "[-3, -2] ∪ [2, 3]"
    portfolio = Explorer(Portfolio, inputs=DEMO_INPUTS).snapshot()
    assert portfolio["exploration"]["coordinates"][0]["window"] == ["0%", "100%"]
    with pytest.raises(ModelInputError, match="distinct finite"):
        Explorer(Response, inputs={"y": 0}).plot(
            controls=["x"],
            output="total",
            windows={"x": (1, 1 + Rational(1, 10**100))},
        )

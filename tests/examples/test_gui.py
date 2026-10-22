"""Guided portfolio and projectile questions reuse their original model equations."""

from sympy import Rational

from modello.gui import Explorer, create_app
from tests.modello._gui_models import rows


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
    assert explorer.snapshot({"allocation.equity": Rational(6, 5)})["status"] == "empty"
    assert rows(explorer.snapshot())["allocation.equity"]["kind"] == "free"


def test_invalid_portfolio_edit_identifies_both_input_and_derived_domain():
    from examples.gui import DEMO_INPUTS, Portfolio

    client = create_app(Portfolio, inputs=DEMO_INPUTS).test_client()
    response = client.post("/api/explore", json={"bindings": {"allocation.equity": "120%"}})
    assert response.status_code == 200
    assert response.json["status"] == "empty"
    assert "allocation.bonds" in response.json["exploration"]["violations"]


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


def test_exact_response_insight_and_three_coordinate_demo():
    from examples.gui_families import demo_family

    calibration = Explorer(demo_family("calibration")).snapshot()
    assert {
        (relation["source"], relation["output"], relation["ratio"])
        for relation in calibration["exploration"]["relations"]
    } == {("recorded.cost", "planned.cost", "25")}
    reachability = Explorer(demo_family("reachability"))
    assert reachability.snapshot()["exploration"]["coordinate_count"] == 3
    chart = reachability.plot(
        controls=["x", "y"],
        output="height",
        windows={"x": (-40, 40), "y": (-40, 40)},
        point={"x": 0, "y": 0, "target_height": 0},
    )
    assert chart["section"] and len(chart["points"]) == 317
    assert chart["rejected"] == 124 and chart["unknown"] == 0


def test_projectile_guided_curve_area_surface_and_elevated_band():
    from examples.gui_families import demo_family

    client = create_app(demo_family("projectile")).test_client()
    initial = client.post("/api/explore", json={}).json
    assert initial["exploration"]["coordinate_count"] == 1
    curve = client.post("/api/plot", json={key: initial["plot"][key] for key in ("controls", "output", "windows")}).json
    assert curve["points"][0]["axes"] == [-40, 0]
    assert curve["points"][20]["axes"] == [0, 20]
    assert curve["points"][-1]["axes"] == [40, 0]
    assert all(abs(point["axes"][1] - (20 - point["axes"][0] ** 2 / 80)) < 1e-12 for point in curve["points"])
    for index in (1, 2, 3):
        scenario = initial["scenarios"][index]
        options = {key: scenario[key] for key in ("bindings", "bounds")}
        preview = client.post("/api/explore", json=options)
        assert preview.status_code == 200
        assert preview.json["exploration"]["coordinate_count"] == 2
        plot = scenario["plot"]
        response = client.post(
            "/api/plot", json={**options, **{key: plot[key] for key in ("controls", "output", "windows")}}
        )
        assert response.status_code == 200
        chart = response.json
        assert chart["unknown"] == 0 and chart["rejected"] > 0
        assert not chart["section"]
        if index == 2:
            assert plot["view"] == "surface"
            assert len(chart["points"]) == 317
            assert next(point for point in chart["points"] if point["index"] == 220)["axes"] == [0, 0, 20]
            assert all(abs(z - (20 - (x * x + y * y) / 80)) < 1e-12 for x, y, z in (p["axes"] for p in chart["points"]))
        else:
            assert all(z >= 0 for _, _, z in (p["axes"] for p in chart["points"]))
            assert all(abs(z - (20 - x * x / 80 - y)) < 1e-12 for x, y, z in (p["axes"] for p in chart["points"]))
            if index == 3:
                assert all(8 <= point["axes"][1] <= 12 for point in chart["points"])
        selected = client.post("/api/explore", json={**options, "point": chart["points"][0]["point"]}).json
        assert selected["status"] == "checked"
        assert selected["exploration"]["coordinate_count"] == 2
    scenario = initial["scenarios"][4]
    choices = client.post("/api/explore", json={"bindings": scenario["bindings"]}).json
    assert choices["exploration"]["case_count"] == 2
    assert {candidate["values"]["x"]["value"] for candidate in choices["exploration"]["cases"]} == {
        "-20*sqrt(2)",
        "20*sqrt(2)",
    }
    # Reset returns to the original curved boundary, including its bound observations.
    assert client.post("/api/explore", json={}).json["exploration"]["coordinate_count"] == 1

"""The portfolio explorer supports forward, inverse, and revised assumptions."""

import pytest
from sympy import Rational

from modello.gui import BindingError, Explorer, create_app
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

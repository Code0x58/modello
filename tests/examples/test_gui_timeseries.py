"""Guided time-series scenarios retain their formulas and inverse solves."""

from xml.etree import ElementTree

import pytest

from modello.gui import Explorer
from tests.modello._gui_models import mathml_text, rows


@pytest.mark.parametrize(
    "demo", ["inventory", "returns", "rolling", "exposure", "trend", "relative", "diffusion", "reversion"]
)
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
                ElementTree.fromstring(constraint["mathml"])
        if scenario.title == "Fit the stationary line":
            fields = rows(data)
            assert fields["slope"]["value"] == fields["intercept"]["value"] == "9/10"
            assert "∑" in mathml_text(fields["squared_error"]["constraints"][0]["mathml"])
            assert "d" in mathml_text(fields["slope_gradient"]["constraints"][0]["mathml"])

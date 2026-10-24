"""Exercise numerical contracts through both public symbolic solve entrypoints."""

import pytest

from modello import inspect_model


@pytest.fixture(params=("constructor", "family"))
def solve_model(request):
    def solve(model, **observations):
        if request.param == "constructor":
            return model("scenario", **observations)
        return inspect_model(model, "scenario", **observations).solutions().one()

    return solve

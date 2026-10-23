"""Fresh bound problems and controlled solver outcomes for core tests."""

import pytest
from sympy import ConditionSet, Eq

import modello.diagnostics as diagnostics
from modello import ModelProblem, inspect_model

from ._models import Check, PositiveScalar, Reading, Sum


@pytest.fixture(
    params=[(NotImplementedError, "unsupported solver"), (ValueError, "solver value error")],
    ids=["unsupported", "value-error"],
)
def backend_error(request: pytest.FixtureRequest) -> Exception:
    constructor, message = request.param
    return constructor(message)


@pytest.fixture
def copied_estimate_problem():
    from ._gui_models import Estimate, Panel

    return inspect_model(Estimate, "snapshot", panel=Panel("source", width=4, height=5), price=10)


@pytest.fixture
def reading_problem() -> ModelProblem[Reading]:
    return inspect_model(Reading, "reading")


@pytest.fixture
def check_problem() -> ModelProblem[Check]:
    return inspect_model(Check, "check")


@pytest.fixture
def conflicting_sum() -> ModelProblem[Sum]:
    return inspect_model(Sum, "conflict", x=1, y=2, total=4)


@pytest.fixture
def positive_problem() -> ModelProblem[PositiveScalar]:
    return inspect_model(PositiveScalar, "positive")


@pytest.fixture
def inconclusive_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics, "solve", lambda *args, **kwargs: [])
    monkeypatch.setattr(diagnostics, "solveset", lambda f, x, domain: ConditionSet(x, Eq(f, 0), domain))

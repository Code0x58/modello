"""Validation inputs shared with the accounting example and documentation."""

import pytest

from examples.accounting_breaks import Ledger, accounting_rows
from examples.portfolio.models import HoldingState
from modello import Index, InstanceDummy, Modello, inspect_model, model_recurrence, model_vector
from modello.contrib.validation import WindowValidator
from modello.diagnostics import Candidate, FieldValue


@pytest.fixture
def validator():
    return WindowValidator(Ledger, "rows")


@pytest.fixture
def rows():
    return list(accounting_rows())


@pytest.fixture
def cash_rows(request: pytest.FixtureRequest, rows):
    return [dict(rows[0], cash=cash, nav=cash, cash_change=movement) for cash, movement in request.param]


@pytest.fixture
def make_validator():
    """Prepare one test-specific law using an existing example row schema."""

    def make(updates, *, row_model=HoldingState, start=1):
        class History(Modello):
            N = InstanceDummy("N", integer=True, nonnegative=True)
            rows = model_vector(row_model, size=N)
            n = Index("n")
            law = model_recurrence(rows[n], start=start, **updates(rows, n))

        return WindowValidator(History, "rows")

    return make


@pytest.fixture
def check_assignment():
    """Check supplied field values with the core API, without solving for gaps."""

    def check(model, values, **inputs):
        problem = inspect_model(model, "comparison", **inputs)
        candidate = Candidate(
            tuple(
                FieldValue(problem.field(*((path,) if isinstance(path, str) else path)), value)
                for path, value in values.items()
            )
        )
        return problem.check(candidate)

    return check

"""Validation inputs shared with the accounting example and documentation."""

import pytest

from examples.accounting_breaks import Ledger, accounting_rows
from modello.contrib.validation import WindowValidator


@pytest.fixture
def validator():
    return WindowValidator(Ledger, "rows")


@pytest.fixture
def rows():
    return list(accounting_rows())

"""The generic ledger example identifies a locally isolated accounting break."""

from copy import deepcopy

from examples.accounting_breaks import EVENTS, Ledger, accounting_rows, find_breaks
from modello.contrib.validation import WindowValidator


def test_accounting_break_and_input_preservation():
    rows = accounting_rows()
    original = deepcopy(rows)
    problems = list(find_breaks(rows, labels=EVENTS))
    assert [(p.position, p.index, p.field, p.rule, p.observed, p.expected, p.residual) for p in problems] == [
        (2, "sale", "cash", "posting", 75, 83, -8)
    ]
    assert rows == original
    validator = WindowValidator(Ledger, "rows")
    results = list(validator.iter_results(rows, labels=EVENTS))
    assert [result.valid for result in results] == [None, True, False, True]
    assert results[-1].complete
    assert validator.window_size == 2 and validator.first_test_position == 1
    assert next(find_breaks(rows, labels=EVENTS)).position == 2


def test_correcting_the_break_and_subsequent_balances_validates_the_ledger():
    rows = accounting_rows()
    for position in (2, 3):
        for field in ("cash", "nav"):
            rows[position][field] += 8
    assert list(find_breaks(rows)) == []
    audit = WindowValidator(Ledger, "rows").audit(rows)
    assert audit.first_invalid is None
    assert (audit.assumed_rows, audit.checked_rows, audit.unknown_rows) == (1, 3, 0)

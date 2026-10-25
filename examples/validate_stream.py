"""Validate ordinary Python rows without pandas or another frame library.

Run ``uv run python -m examples.validate_stream``. External stream processors
can instead call ``validator.check_window(window, position=n)`` directly.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

from modello import Index, InstanceDummy, Modello, model_recurrence, model_vector
from modello.contrib.validation import ValidationSummary, WindowResult, WindowValidator


class CashRow(Modello):
    cash = InstanceDummy("cash", real=True)
    debit = InstanceDummy("debit", real=True)


class CashLedger(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    rows = model_vector(CashRow, size=N)
    n = Index("n")
    posting = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)


def scan_batches() -> Iterator[WindowResult]:
    """One row iterator preserves history and absolute positions across batches."""
    batches = (((100, 0), (90, 10)), ((85, 2),), ((83, 2),))
    validator = WindowValidator(CashLedger, "rows")
    return validator.iter_results(chain.from_iterable(batches))


if __name__ == "__main__":
    summary = ValidationSummary.from_results(scan_batches())
    print(f"Rows: {summary.row_count}; fully checked: {summary.checked_rows}; assumed: {summary.assumed_rows}")
    if summary.first_invalid is not None:
        problem = summary.first_invalid
        print(f"Break at position {problem.position}: {problem.field} residual {problem.residual}")

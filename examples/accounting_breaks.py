"""Find a cash roll-forward break in ordinary nested account observations.

Run ``uv run python -m examples.accounting_breaks``. Amounts are integer minor
currency units; quantities count shares. No frame library is needed.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator, Mapping

from modello import Index, InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.contrib.validation import CheckResult, WindowValidator


class Holding(Modello):
    quantity = InstanceDummy("quantity", integer=True)
    price = InstanceDummy("price", positive=True)
    value = quantity * price


class AccountRow(Modello):
    holding = component(Holding)
    cash = InstanceDummy("cash", real=True)
    cash_change = InstanceDummy("cash_change", real=True)
    quantity_change = InstanceDummy("quantity_change", integer=True)
    nav = cash + holding.value


class Ledger(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    rows = model_vector(AccountRow, size=N)
    n = Index("n")
    posting = model_recurrence(
        rows[n],
        cash=rows[n - 1].cash + rows[n].cash_change,
        **{"holding.quantity": rows[n - 1].holding.quantity + rows[n].quantity_change},
    )


EVENTS = ("opening", "purchase", "sale", "repurchase")


def accounting_rows() -> tuple[dict[str, int], ...]:
    """Include an eight-cent cash break at the sale, then post locally correctly."""
    fields = ("holding.quantity", "holding.price", "holding.value", "cash", "cash_change", "quantity_change", "nav")
    values = (
        (0, 10_000, 0, 100_000, 0, 0, 100_000),
        (2, 10_000, 20_000, 79_999, -20_001, 2, 99_999),
        (1, 11_000, 11_000, 90_990, 10_999, -1, 101_990),
        (2, 10_500, 21_000, 80_489, -10_501, 1, 101_489),
    )
    return tuple(dict(zip(fields, row, strict=True)) for row in values)


def find_breaks(
    rows: Iterable[Mapping[str, object]], *, labels: Iterable[Hashable] | None = None
) -> Iterator[CheckResult]:
    """An example of caller-owned error selection over generic validator results."""
    validator = WindowValidator(Ledger, "rows")
    return (
        check
        for result in validator.iter_results(rows, labels=labels)
        for check in result.checks
        if check.verdict == "violated"
    )


if __name__ == "__main__":
    for problem in find_breaks(accounting_rows(), labels=EVENTS):
        print(
            f"{problem.index}: {problem.field} observed {problem.observed}, "
            f"expected {problem.expected}, residual {problem.residual}"
        )

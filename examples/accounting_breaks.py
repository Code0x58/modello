"""Find a cash shortfall: ``uv run python -m examples.accounting_breaks``."""

from __future__ import annotations

from modello import Index, InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.contrib.validation import WindowValidator


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
    """Return ledger observations with an incorrect sale balance."""
    fields = ("holding.quantity", "holding.price", "holding.value", "cash", "cash_change", "quantity_change", "nav")
    values = (
        (0, 10, 0, 40, 0, 0, 40),
        (2, 10, 20, 20, -20, 2, 40),
        (1, 10, 10, 29, 10, -1, 39),
        (2, 10, 20, 19, -10, 1, 39),
    )
    return tuple(dict(zip(fields, row, strict=True)) for row in values)


if __name__ == "__main__":
    results = WindowValidator(Ledger, "rows").iter_results(accounting_rows(), labels=EVENTS)
    for problem in (check for result in results for check in result.checks if check.invalid):
        print(
            f"{problem.index}: {problem.field} observed {problem.observed}, "
            f"expected {problem.expected}, residual {problem.residual}"
        )

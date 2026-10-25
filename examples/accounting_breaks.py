"""Find a cash roll-forward break in observed, nested account rows.

Run ``uv run --extra pandas python -m examples.accounting_breaks``.
Amounts are integer minor currency units; quantities count shares.
"""

from __future__ import annotations

import pandas as pd

from modello import Index, InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.contrib.pandas import FrameValidation, FrameValidator


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


def accounting_frame() -> pd.DataFrame:
    """Include one eight-cent cash break at the sale, then post locally correctly."""
    return pd.DataFrame(
        [
            [0, 10_000, 0, 100_000, 0, 0, 100_000],
            [2, 10_000, 20_000, 79_999, -20_001, 2, 99_999],
            [1, 11_000, 11_000, 90_990, 10_999, -1, 101_990],
            [2, 10_500, 21_000, 80_489, -10_501, 1, 101_489],
        ],
        columns=pd.MultiIndex.from_tuples(
            [
                ("holding", "quantity"),
                ("holding", "price"),
                ("holding", "value"),
                ("cash", ""),
                ("cash_change", ""),
                ("quantity_change", ""),
                ("nav", ""),
            ]
        ),
        index=pd.Index(["opening", "purchase", "sale", "repurchase"], name="event"),
    )


def find_breaks(frame: pd.DataFrame) -> FrameValidation:
    """Check the same declared law against each pair of observed account rows."""
    return FrameValidator(Ledger, "rows").validate(frame)


if __name__ == "__main__":
    report = find_breaks(accounting_frame())
    print(report.breaks[["position", "index", "field", "observed", "expected", "residual"]].to_string(index=False))

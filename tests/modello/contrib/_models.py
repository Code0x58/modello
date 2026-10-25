"""Cash schema and recurrence shared by validation tests."""

from modello import Index, InstanceDummy, Modello, model_recurrence, model_vector


class CashRow(Modello):
    cash = InstanceDummy("cash", real=True)
    debit = InstanceDummy("debit", real=True)


class CashLedger(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    rows = model_vector(CashRow, size=N)
    n = Index("n")
    posting = model_recurrence(rows[n], cash=rows[n - 1].cash - rows[n].debit)

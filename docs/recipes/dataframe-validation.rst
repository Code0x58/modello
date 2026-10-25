Validate pandas and Polars DataFrames
=====================================

Select the validator's fields and pass a row iterator to ``iter_results``. It
maintains the rolling history and yields one result per row. These recipes use
the ledger from :doc:`../contrib/validation`, which explains the accounting
example, result records, and validation coverage.

.. doctest:: pandas, polars

   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> from modello.contrib.validation import WindowValidator
   >>> validator = WindowValidator(Ledger, "rows")

Validate pandas rows
--------------------

Use ``loc`` to select column labels in ``validator.fields`` order, then pass
their values and the frame's index labels to the validator:

.. doctest:: pandas

   >>> import pandas as pd
   >>> frame = pd.DataFrame(accounting_rows(), index=EVENTS, dtype=object)
   >>> selected = frame.loc[:, list(validator.fields)]
   >>> selected = selected.astype(object).where(selected.notna(), None)
   >>> rows = selected.itertuples(index=False, name=None)
   >>> results = list(validator.iter_results(rows, labels=frame.index))
   >>> frame["valid"] = pd.array([result.valid for result in results], dtype="boolean")
   >>> frame["valid"].tolist()
   [<NA>, True, False, True]
   >>> [result.index for result in results if result.invalid]
   ['sale']

``True`` means the row passes, ``False`` identifies a proven violation, and
``pd.NA`` means validity is assumed or undecidable. The output keeps the frame's
index, including duplicate labels and gaps; validation follows row order.

The ``astype(object).where(...)`` conversion replaces pandas' missing-value
sentinels with ``None``. Constructing the input with ``dtype=object`` preserves
exact integer and ``Decimal`` values; converting an existing float column
cannot recover precision already lost.

Call ``iter_results`` once over the rows: ``apply(axis=1)`` would start a separate
iterator for each row rather than maintain history across them.

MultiIndex columns
~~~~~~~~~~~~~~~~~~

For an existing frame named ``accounts`` with nested columns such as
``("holding", "quantity")`` and ``("cash", "")``, build the selection from
``validator.field_paths``. This convention pads shorter paths with empty strings:

.. testcode:: pandas
   :hide:

   accounts = selected.copy()
   column_depth = max(map(len, validator.field_paths))
   accounts.columns = pd.MultiIndex.from_tuples(
       [path + ("",) * (column_depth - len(path)) for path in validator.field_paths])

.. doctest:: pandas

   >>> keys = [path + ("",) * (accounts.columns.nlevels - len(path)) for path in validator.field_paths]
   >>> rows = accounts.loc[:, keys].itertuples(index=False, name=None)
   >>> [result.valid for result in validator.iter_results(rows, labels=accounts.index)]
   [None, True, False, True]

Map the paths to your source labels when names or padding differ, keeping the
selected columns in ``validator.field_paths`` order. Apply the same missing-value
conversion when needed.

Validate Polars rows
--------------------

Select the same fields and supply Polars' row iterator. Polars nulls already
arrive as ``None``:

.. doctest:: polars

   >>> import polars as pl
   >>> frame = pl.DataFrame(accounting_rows()).with_columns(pl.Series("event", EVENTS))
   >>> rows = frame.select(list(validator.fields)).iter_rows()
   >>> results = list(validator.iter_results(rows, labels=frame["event"]))
   >>> frame = frame.with_columns(
   ...     pl.Series("valid", [result.valid for result in results], dtype=pl.Boolean))
   >>> frame["valid"].to_list()
   [None, True, False, True]

Both recipes perform validation in Python/SymPy. For batched or unbounded input,
see the streaming guidance in :doc:`../contrib/validation`.

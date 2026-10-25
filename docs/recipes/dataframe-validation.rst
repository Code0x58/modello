Validate pandas and Polars DataFrames
=====================================

Use the ledger from :doc:`../contrib/validation` to add a nullable validity
column. Select ``validator.fields`` in order and pass the rows to
``iter_results``, which maintains the rolling history.

.. doctest:: pandas, polars

   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> from modello.contrib.validation import WindowValidator
   >>> validator = WindowValidator(Ledger, "rows")

Validate pandas rows
--------------------

``loc`` selects field names; ``itertuples`` supplies their stored values:

.. doctest:: pandas

   >>> import pandas as pd
   >>> frame = pd.DataFrame(accounting_rows(), index=EVENTS, dtype=object)
   >>> rows = frame.loc[:, list(validator.fields)].itertuples(index=False, name=None)
   >>> frame["valid"] = pd.array(
   ...     [result.valid for result in validator.iter_results(rows)], dtype="boolean")
   >>> frame["valid"].tolist()
   [<NA>, True, False, True]

``True`` means the row passes, ``False`` identifies a proven violation, and
``pd.NA`` means validity is assumed or undecidable. The output keeps the frame's
index, including duplicate labels and gaps; validation follows row order.

For nullable input columns, first use
``frame = frame.astype(object).where(frame.notna(), None)`` to replace pandas'
missing-value sentinels with ``None``. ``dtype=object`` preserves exact integer
and ``Decimal`` inputs; it cannot recover precision already lost to floats.

MultiIndex columns
~~~~~~~~~~~~~~~~~~

For an existing ``accounts`` frame with columns such as
``("holding", "quantity")`` and ``("cash", "")``, select
``validator.field_paths``, padding shorter paths with empty strings:

.. testcode:: pandas
   :hide:

   accounts = frame.loc[:, list(validator.fields)].copy()
   column_depth = max(map(len, validator.field_paths))
   accounts.columns = pd.MultiIndex.from_tuples(
       [path + ("",) * (column_depth - len(path)) for path in validator.field_paths])

.. doctest:: pandas

   >>> keys = [path + ("",) * (accounts.columns.nlevels - len(path)) for path in validator.field_paths]
   >>> rows = accounts.loc[:, keys].itertuples(index=False, name=None)

Pass ``rows`` to ``iter_results`` as above. Adapt the keys when your column names
or padding differ, preserving field order.

.. testcode:: pandas
   :hide:

   assert [result.valid for result in validator.iter_results(rows)] == [None, True, False, True]

Validate Polars rows
--------------------

Polars supplies tuples directly, with nulls represented by ``None``:

.. doctest:: polars

   >>> import polars as pl
   >>> frame = pl.DataFrame(accounting_rows()).with_columns(pl.Series("event", EVENTS))
   >>> rows = frame.select(list(validator.fields)).iter_rows()
   >>> frame = frame.with_columns(
   ...     pl.Series("valid", [result.valid for result in validator.iter_results(rows)], dtype=pl.Boolean))
   >>> frame["valid"].to_list()
   [None, True, False, True]

Both recipes check rows in Python/SymPy. To inspect failures or process a stream,
see :doc:`../contrib/validation`.

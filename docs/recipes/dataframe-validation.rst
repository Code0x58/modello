Validate pandas and Polars DataFrames
=====================================

Use the ledger from :doc:`../contrib/validation` to validate a pandas frame or
stream a Polars lazy query. Select ``validator.fields`` in order and pass the
rows to ``iter_results``, which maintains the rolling history.

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

Stream a Polars lazy query
--------------------------

Scan a Parquet ledger at ``ledger_path``. `collect_batches
<https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.collect_batches.html>`_
streams ordered batches without collecting the whole result. Flatten their
rows into one iterator so validation retains history across batches:

.. testsetup:: polars

   from pathlib import Path
   from tempfile import TemporaryDirectory
   import polars as pl
   from examples.accounting_breaks import accounting_rows

   directory = TemporaryDirectory()
   ledger_path = Path(directory.name) / "ledger.parquet"
   pl.DataFrame(accounting_rows()).write_parquet(ledger_path)

.. doctest:: polars

   >>> import polars as pl
   >>> from itertools import chain
   >>> query = pl.scan_parquet(ledger_path).select(list(validator.fields))
   >>> batches = query.collect_batches(engine="streaming", maintain_order=True)
   >>> rows = chain.from_iterable(batch.iter_rows() for batch in batches)
   >>> results = validator.iter_results(rows)
   >>> [result.valid for result in results]
   [None, True, False, True]

Polars supplies nulls as ``None``. Consume ``results`` incrementally for large
inputs; the list above only displays this small example. Keep one scan in
event order for each independent history.
Batch size is independent of ``validator.window_size``. Polars currently marks
``collect_batches`` as unstable.

.. testcode:: polars
   :hide:

   for chunk_size in range(1, len(EVENTS) + 1):
       batches = query.collect_batches(chunk_size=chunk_size, engine="streaming", maintain_order=True)
       rows = chain.from_iterable(batch.iter_rows() for batch in batches)
       assert [result.valid for result in validator.iter_results(rows)] == [None, True, False, True]

.. testcleanup:: polars

   directory.cleanup()

Both recipes check rows in Python/SymPy. See :doc:`../contrib/validation` for
inspecting failures and summarising coverage.

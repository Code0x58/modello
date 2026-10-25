Validate pandas and Polars DataFrames
=====================================

Check recorded account balances against a recurrence, attach a validity column,
and locate accounting breaks. These recipes use the
:doc:`window validator <../contrib/validation>` with pandas and Polars.

Install Modello alongside the DataFrame library you use: ``pip install modello
pandas`` or ``pip install modello polars``.

The accounting example in :doc:`../contrib/validation` checks
:math:`C_n=C_{n-1}+\Delta C_n`, where :math:`C_n` is the recorded cash balance
and :math:`\Delta C_n` is the cash movement at row :math:`n`. Cash values and
residuals use integer cents. ``True`` means the row passes, ``False``
identifies a break, and ``None`` means validity is assumed or undecidable.

Validate pandas rows
--------------------

Select the model's fields in ``validator.fields`` order, then iterate over
`DataFrame windows <https://pandas.pydata.org/docs/user_guide/window.html>`_.
Each window contains all the columns needed by the law. Object columns retain
exact integer and ``Decimal`` inputs; ``itertuples`` passes their stored values
to the validator.

.. doctest:: pandas

   >>> import pandas as pd
   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> from modello.contrib.validation import WindowValidator
   >>> validator = WindowValidator(Ledger, "rows")
   >>> frame = pd.DataFrame(accounting_rows(), index=EVENTS, dtype=object)
   >>> projected = frame.loc[:, list(validator.fields)]
   >>> def pandas_windows(projected, validator):
   ...     windows = projected.rolling(validator.window_size, min_periods=1)
   ...     for position, window in enumerate(windows):
   ...         rows = tuple(
   ...             tuple(None if value is pd.NA or value is pd.NaT else value for value in row)
   ...             for row in window.itertuples(index=False, name=None)
   ...         )
   ...         yield validator(rows, position=position, index=window.index[-1])
   >>> results = list(pandas_windows(projected, validator))
   >>> mask = pd.Series(
   ...     [result.valid for result in results], index=frame.index, dtype="boolean", name="valid")
   >>> mask.tolist()
   [<NA>, True, False, True]
   >>> mask.index.equals(frame.index)
   True

The mask has one entry per input row and preserves the frame's index. The
opening balance is assumed, so its entry is ``pd.NA``. The sale is eight cents
short (USD 0.75 against USD 0.83), so that row is ``False``. The repurchase
passes using the preceding recorded balance.

Include the short initial windows: a law of order :math:`p` first tests position
:math:`p`, using :math:`p+1` observations. Pass the absolute row position to the
validator. The index label is retained for reporting; it does not determine
the row's mathematical position. See :doc:`../contrib/validation` for laws with
different start positions.

Handle missing values and index labels
--------------------------------------

The validator accepts ``None`` and numeric NaN as missing observations. The
helper also converts pandas' ``pd.NA`` and ``pd.NaT`` sentinels to ``None``.
A missing balance can leave both its own row and the next transition undecidable:

.. doctest:: pandas

   >>> missing = projected.copy()
   >>> missing["cash"] = missing["cash"].mask(missing.index == "purchase", pd.NA)
   >>> [result.valid for result in pandas_windows(missing, validator)]
   [None, None, None, True]

Duplicate labels and gaps in the index are preserved. Windows follow the
frame's row order:

.. doctest:: pandas

   >>> events_per_label, label_spacing = 2, 10
   >>> labels = pd.Index(
   ...     (position // events_per_label * label_spacing for position in range(len(projected))), name="event")
   >>> gappy = projected.set_axis(labels)
   >>> labelled = list(pandas_windows(gappy, validator))
   >>> [result.index for result in labelled] == gappy.index.tolist()
   True

Find the first or all errors
----------------------------

Filter the results for violated checks. Use ``next`` to stop at the first break,
or collect the iterator to inspect every break:

.. doctest:: pandas

   >>> def errors(results):
   ...     return (check for result in results for check in result.checks if check.verdict == "violated")
   >>> first = next(errors(pandas_windows(projected, validator)), None)
   >>> if first is not None:
   ...     print((first.position, first.index, first.field, first.expected, first.residual))
   (2, 'sale', 'cash', 83, -8)
   >>> all_errors = list(errors(pandas_windows(projected, validator)))
   >>> [(check.position, check.residual) for check in all_errors]
   [(2, -8)]

An empty error iterator means no violation was proven. Initial assumptions and
missing data can still prevent complete validation. Inspect per-row coverage,
or audit the full sequence before deciding whether to accept the data:

.. doctest:: pandas

   >>> from modello.contrib.validation import ValidationSummary
   >>> audit = ValidationSummary.from_results(pandas_windows(projected, validator))
   >>> audit.row_count, audit.assumed_rows, audit.checked_rows, audit.violated_rows
   (4, 1, 3, 1)

Select nested fields from MultiIndex columns
--------------------------------------------

For a model with nested components, build MultiIndex labels from
``validator.field_paths``, padding shorter paths with empty strings. Pass the
selected rows to the same window helper:

.. doctest:: pandas

   >>> column_depth = max(map(len, validator.field_paths))
   >>> keys = [path + ("",) * (column_depth - len(path)) for path in validator.field_paths]
   >>> accounts = projected.copy()
   >>> accounts.columns = pd.MultiIndex.from_tuples(keys)
   >>> selected = accounts.loc[:, keys]
   >>> account_results = list(pandas_windows(selected, validator))
   >>> [result.valid for result in account_results]
   [None, True, False, True]
   >>> [(check.index, check.field, check.residual) for check in errors(account_results)]
   [('sale', 'cash', -8)]

For an existing MultiIndex frame, use its ``columns.nlevels`` as
``column_depth``. Map field paths to source labels when names differ, keeping
the selected columns in ``validator.fields`` order.

Validate Polars rows
--------------------

Select the model's fields, then supply Polars' `row iterator
<https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.iter_rows.html>`_.
``iter_results`` maintains the rolling window and returns one result per input
row. Attach the validity values as a nullable boolean column:

.. doctest:: polars

   >>> import polars as pl
   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> from modello.contrib.validation import WindowValidator
   >>> validator = WindowValidator(Ledger, "rows")
   >>> polars_frame = pl.DataFrame(accounting_rows()).with_columns(pl.Series("event", EVENTS))
   >>> rows = polars_frame.select(list(validator.fields)).iter_rows()
   >>> polars_results = list(validator.iter_results(rows, labels=polars_frame["event"]))
   >>> [result.valid for result in polars_results]
   [None, True, False, True]
   >>> annotated = polars_frame.with_columns(
   ...     pl.Series("valid", [result.valid for result in polars_results], dtype=pl.Boolean))
   >>> annotated["valid"].to_list()
   [None, True, False, True]
   >>> breaks = (
   ...     check for result in polars_results for check in result.checks if check.verdict == "violated")
   >>> [(check.index, check.residual) for check in breaks]
   [('sale', -8)]
   >>> nullable = polars_frame.with_columns(
   ...     pl.when(pl.col("event") == "purchase").then(None).otherwise(pl.col("cash")).alias("cash"))
   >>> nullable_rows = nullable.select(list(validator.fields)).iter_rows()
   >>> [result.valid for result in validator.iter_results(nullable_rows)]
   [None, None, None, True]

Polars nulls arrive as ``None``, so they need no additional conversion. Checks
execute in Python/SymPy. For batched or unbounded sources, keep one row iterator
across batches, or retain the overlapping history and absolute positions when
supplying windows directly. See :doc:`../contrib/validation` for streaming and
partitioning examples.

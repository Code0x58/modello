Caller-owned DataFrame recipes
==============================

The :doc:`window validator <validation>` has no frame-library dependency.
Projection, null conversion, windowing, and error selection belong to the
caller. These are executable examples, not runtime adapters shipped by Modello.

Install the repository's documentation dependencies with Python 3.12 or newer:

.. code-block:: console

   $ uv sync --locked --dev --group docs
   $ uv run --locked --group docs sphinx-build --fail-on-warning --keep-going --builder=doctest docs docs/_build/doctest

The ``docs`` dependency group includes pandas and Polars. Documentation CI runs
these examples with Sphinx's doctest builder before building the HTML guide.
The ordinary library test environment does not require either frame library.

pandas: project, iterate windows, call
--------------------------------------

pandas' rolling ``apply`` expects a scalar numeric result per column. For a law
that relates several fields, iterate the `DataFrame windows
<https://pandas.pydata.org/docs/user_guide/window.html>`_ and call the validator
once per row window. This also preserves exact integers and decimal cells.

.. doctest::

   >>> import pandas as pd
   >>> from examples.validate_stream import CashLedger
   >>> from modello.contrib.validation import WindowValidator
   >>> validator = WindowValidator(CashLedger, "rows")
   >>> frame = pd.DataFrame({"cash": [100, 90, 85, 83], "debit": [0, 10, 2, 2]},
   ...                      index=["opening", "purchase", "sale", "fee"], dtype=object)
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
   >>> mask = pd.Series([result.valid for result in results], index=frame.index, dtype="boolean", name="valid")
   >>> mask.tolist()
   [<NA>, True, False, True]
   >>> mask.index.equals(frame.index)
   True

Emit partial prefix windows: an order-p law first tests position p, using p+1
observations. A framework that discards short windows can hide assumed history
and, with different law starts, discard earlier evidence. Pass absolute row
positions, not the current window's length. Keep labels as metadata.

Null conversion remains local to the recipe. The kernel accepts ``None`` and
numeric NaN; the helper converts pandas' ``NA`` and ``NaT`` sentinels:

.. doctest::

   >>> missing = projected.copy()
   >>> missing.loc["purchase", "cash"] = pd.NA
   >>> [result.valid for result in pandas_windows(missing, validator)]
   [None, None, None, True]
   >>> gappy = projected.set_axis(pd.Index([0, 3, 3, 20], name="event"))
   >>> labelled = list(pandas_windows(gappy, validator))
   >>> [result.index for result in labelled] == gappy.index.tolist()
   True

Choose first or all errors
--------------------------

The validator returns evidence; the caller chooses its terminal operation.
``next`` can stop a lazy pipeline at the first proven violation. A comprehension
can collect every violation. Unknown and assumed checks remain distinct.

.. doctest::

   >>> def errors(results):
   ...     return (check for result in results for check in result.checks if check.verdict == "violated")
   >>> first = next(errors(pandas_windows(projected, validator)), None)
   >>> first.position, first.index, first.field, first.expected, first.residual
   (2, 'sale', 'cash', 88, -3)
   >>> all_errors = list(errors(pandas_windows(projected, validator)))
   >>> [(check.position, check.residual) for check in all_errors]
   [(2, -3)]

An empty error iterator means no violation was proved. For strict acceptance,
also inspect per-row coverage or build a finite audit over all results:

.. doctest::

   >>> from modello.contrib.validation import ValidationSummary
   >>> audit = ValidationSummary.from_results(pandas_windows(projected, validator))
   >>> audit.row_count, audit.assumed_rows, audit.checked_rows, audit.violated_rows
   (4, 1, 3, 1)

MultiIndex columns are a projection choice
------------------------------------------

The nested accounting model still uses ordinary canonical field paths. Map
those paths to your frame's column labels before applying the same helper:

.. doctest::

   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> accounts = pd.DataFrame(accounting_rows(), index=EVENTS)
   >>> accounts.columns = pd.MultiIndex.from_tuples(
   ...     tuple(name.split(".")) if "." in name else (name, "") for name in accounts.columns)
   >>> account_validator = WindowValidator(Ledger, "rows")
   >>> keys = [path + ("",) * (accounts.columns.nlevels - len(path)) for path in account_validator.field_paths]
   >>> selected = accounts.loc[:, keys]
   >>> account_results = list(pandas_windows(selected, account_validator))
   >>> [result.valid for result in account_results]
   [None, True, False, True]
   >>> [(check.index, check.field, check.residual) for check in errors(account_results)]
   [('sale', 'cash', -8)]

The row tuples must follow ``validator.fields`` order. Custom source names,
column depths, and grouping are choices in the caller's projection, rather
than a Modello-specific column-binding API.

Polars: select and supply Python rows
-------------------------------------

Polars can supply the same projected tuples through its `row iterator
<https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.iter_rows.html>`_.
Here the generic bounded iterator owns the count window. A stream framework
that already owns windows can call ``validator(window, position=n)`` directly.

.. doctest::

   >>> import polars as pl
   >>> polars_frame = pl.DataFrame({"event": [0, 3, 3, 20], "debit": [0, 10, 2, 2], "cash": [100, 90, 85, 83]})
   >>> rows = polars_frame.select(list(validator.fields)).iter_rows()
   >>> polars_results = list(validator.iter_results(rows, labels=polars_frame["event"]))
   >>> [result.valid for result in polars_results]
   [None, True, False, True]
   >>> annotated = polars_frame.with_columns(pl.Series("valid", [result.valid for result in polars_results], dtype=pl.Boolean))
   >>> annotated["valid"].to_list()
   [None, True, False, True]
   >>> [(check.index, check.residual) for check in errors(polars_results)]
   [(3, -3)]
   >>> nullable = pl.DataFrame({"cash": [100, 90, None, 86, 84], "debit": [0, 10, 2, 2, 2]})
   >>> [result.valid for result in validator.iter_results(nullable.select(list(validator.fields)).iter_rows())]
   [None, True, None, None, True]

Polars nulls already arrive as ``None``. Checks execute in Python/SymPy; the
frame supplies values without a pandas conversion. For batched or unbounded
sources, keep one row iterator across batches or retain overlap and absolute
positions in the framework, as described in the :doc:`streaming contract
<validation>`.

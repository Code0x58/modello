Observed recurrence windows
===========================

The :doc:`account walkthrough <../model-vectors>` calculates balances from
trades. When a statement already contains those balances, use
``WindowValidator.iter_results`` to check each posting against its recorded
predecessor.

The ledger in ``examples.accounting_breaks`` starts with 40, buys two shares at
10 each, then sells one for 10. The sale should leave 30 in cash, but the
statement records 29. Its cash and share balances obey

.. math::

   \widehat C_n = C_{n-1} + \Delta C_n, \qquad
   \widehat Q_n = Q_{n-1} + \Delta Q_n.

The cash residual is :math:`e_n=C_n-\widehat C_n`: here :math:`29-30=-1`.

.. list-table:: Cash columns from the ledger
   :header-rows: 1

   * - Event
     - Cash
     - Cash movement
     - Valid
     - Why
   * - opening
     - 40.00
     - 0.00
     - Assumed
     - Opening balance is assumed.
   * - purchase
     - 20.00
     - -20.00
     - Yes
     -
   * - sale
     - 29.00
     - 10.00
     - No
     - Expected 30.00; shortfall 1.00.
   * - repurchase
     - 19.00
     - -10.00
     - Yes
     - Uses the recorded sale balance.

Find an accounting break
------------------------

``iter_results`` maintains the rolling history and yields one result per row.
Collect the results to inspect the full sequence:

.. doctest::

   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> from modello.contrib.validation import WindowValidator
   >>> rows = accounting_rows()
   >>> validator = WindowValidator(Ledger, "rows")
   >>> results = list(validator.iter_results(rows, labels=EVENTS))
   >>> [result.valid for result in results]
   [None, True, False, True]
   >>> for result in results:
   ...     if result.first_invalid is not None:
   ...         problem = result.first_invalid
   ...         print(problem.index, problem.field, problem.expected, problem.residual)
   ...         break
   sale cash 30 -1

``valid`` is ``True`` when all current-row checks pass, ``False`` for a proven
violation, and ``None`` for assumed history or insufficient evidence.
Checks before a law's declared start are assumed; an order-:math:`p` law
normally starts at position :math:`p`, using the preceding :math:`p` rows.

Each result retains individual checks, including their rule, observations,
expected value, residual, and verdict. ``complete`` means all checks for that
row ran and were decided, including violations. ``first_invalid is None``
alone does not establish coverage. Later rows use recorded predecessors, so
the repurchase can pass even though the sale failed.

Audit coverage
---------------

For a finite job, ``audit`` scans all rows and summarises their coverage:

.. doctest::

   >>> audit = validator.audit(rows, labels=EVENTS)
   >>> audit.row_count, audit.checked_rows, audit.assumed_rows, audit.unknown_rows, audit.violated_rows
   (4, 3, 1, 0, 1)
   >>> audit.valid
   False

``checked_rows`` counts fully decided rows, ``assumed_rows`` counts wholly
assumed history, and ``unknown_rows`` counts other rows whose validity is
undecidable. ``violated_rows`` counts proven breaks, including rows with
incomplete checks, so it can overlap with ``checked_rows``.

For strict acceptance, require ``audit.valid is True``. Even with no breaks,
assumed or undecidable rows keep that verdict at ``None``. A policy that trusts
initial history can require no violated or unknown rows and at least one fully
checked transition. Empty and history-only inputs provide no such transition.

``ValidationSummary.from_results(results)`` reuses collected results. It requires
contiguous positions from zero; supply ``start_position`` explicitly to audit
another interval. Its ``[start_position, stop_position)`` names that coverage.

Input and numerical values
---------------------------

Supply flat mappings keyed by ``validator.fields``, or row tuples in that order.
Use ``None`` for framework-specific nulls; numeric NaN also means missing.
Parse numeric text before validation. The :doc:`DataFrame recipes
<../recipes/dataframe-validation>` show column selection, nested MultiIndex
fields, nullable validity columns, and error selection in pandas and Polars.

Validation covers the selected row schema's domains, definitions, aliases,
and fixed causal laws. Put per-event inputs in that schema; other vectors and
parent aggregate or boundary constraints are outside this scope. Missing
values and parameters are never inferred or filled.

Integers, ``Decimal``, ``Fraction``, and SymPy numbers retain exact arithmetic.
Equation tolerances use ``abs(observed-expected) <= atol + rtol*abs(expected)``;
domains remain strict. Independent constants supplied through ``parameters``
must satisfy their field assumptions. Results retain SymPy values for exact
evidence.

Streams and manual windows
---------------------------

For a stream, iterate over ``iter_results`` directly without collecting a list.
Each call starts fresh at position zero, so chain batches into one call to keep
history across them. On Python 3.12+:

.. testcode::

   from itertools import batched, chain

   batches = batched(rows, validator.window_size)
   print([result.valid for result in validator.iter_results(chain.from_iterable(batches))])

.. testoutput::

   [None, True, False, True]

Batch length is a source choice; it need not match ``window_size``. For Python
3.10–3.11, the `Python batching recipe
<https://docs.python.org/3/library/itertools.html#itertools.batched>`_ uses
:func:`itertools.islice`.

If a DataFrame or stream processor already supplies row windows, call the
validator directly. The equivalent manual loop is:

.. doctest::

   >>> from collections import deque
   >>> history = deque(maxlen=validator.window_size)
   >>> manual_results = []
   >>> for position, (label, row) in enumerate(zip(EVENTS, rows, strict=True)):
   ...     history.append(tuple(row[field] for field in validator.fields))
   ...     manual_results.append(validator(tuple(history), position=position, index=label))
   >>> manual_results == list(validator.iter_results(rows, labels=EVENTS))
   True

Each call checks the final row of an oldest-first window containing exactly
``min(position+1, validator.window_size)`` rows. Include short initial windows;
laws with earlier starts can already find breaks there. ``position`` is an
absolute nonnegative Python integer; labels are hashable metadata and do not insert missing
observations between gaps. Keep separate histories for independent accounts,
and restore overlapping history and positions across batches or restarts.


API
---

.. currentmodule:: modello.contrib.validation

.. autoclass:: WindowValidator(model, rows, *, parameters=None, atol=0, rtol=0)
   :members: fields, field_paths, order, window_size, first_test_position, fully_testable_position, check_window, iter_results, audit
   :special-members: __call__

.. autoclass:: WindowResult
   :members: valid, complete, assumed, first_invalid

.. autoclass:: CheckResult

.. autoclass:: ValidationSummary
   :members: valid, complete, from_results, stop_position

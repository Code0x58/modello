Observed recurrence windows (contrib)
=====================================

``modello.contrib.validation`` is independent of pandas, Polars, and stream
frameworks. It prepares a reusable function over projected rows, with explicit
coverage alongside each result. The base ``modello`` installation is sufficient.
The :doc:`pandas adapter <pandas>` uses this same window function.

A law of order :math:`p` normally tests

.. math::

   x_n = f(n, x_{n-1}, \ldots, x_{n-p}, u_n), \qquad n \ge p.

Its input window ends at the observation being tested. Positions are zero based:
**the first test is at position p, not p+1**. The first ``p`` observations supply
assumed history; the window contains ``p+1`` rows.

.. list-table:: History and first test
   :header-rows: 1

   * - Order
     - Assumed positions
     - First test position
     - Window width
   * - 0
     - none (with ``start=0``)
     - 0
     - 1
   * - 1
     - 0
     - 1
     - 2
   * - 2
     - 0, 1
     - 2
     - 3

Declare once, supply windows anywhere
-------------------------------------

The ordinary model-vector API supplies the schema and source law:

.. literalinclude:: ../../examples/validate_stream.py
   :start-at: class CashRow
   :end-before: def scan_batches
   :language: python

For this cash process, :math:`\widehat C_n=C_{n-1}-D_n` and the residual is
:math:`e_n=C_n-\widehat C_n`.

.. code-block:: pycon

   >>> from examples.validate_stream import CashLedger
   >>> from modello.contrib.validation import WindowValidator
   >>> validator = WindowValidator(CashLedger, "rows")
   >>> validator.fields
   ('cash', 'debit')
   >>> validator.order, validator.window_size, validator.first_test_position
   (1, 2, 1)
   >>> result = validator.check_window(((90, 10), (85, 2)), position=2, index="sale")
   >>> result.valid, result.complete
   (False, True)
   >>> problem = result.first_invalid
   >>> problem.position, problem.index, problem.field, problem.expected, problem.residual
   (2, 'sale', 'cash', 88, -3)

``check_window`` checks **only the final row**. A complete result means all of
that row's declared checks ran and reached a decided outcome. It does not mean
that preceding rows in the supplied window were independently validated.
Those are observations used to evaluate the current equations.

The function is stateless: it stores no incoming history between calls. Calls
may be interleaved across accounts or dispatched to workers when each receives
the correct window and position. Prepared validators can be pickled when their
model classes live in an importable module.

Input contract
--------------

* Project columns in exactly ``validator.fields`` order and pass row tuples,
  or supply flat mappings keyed by those field names. Extra mapping keys are
  ignored. ``field_paths`` exposes the corresponding nested attribute paths for
  adapters; flat names include ``holding.quantity`` or ``quantities[0]``.
* A window is oldest first, ending at the row being tested. Its length must be
  exactly ``min(position+1, window_size)``. A short window is valid only at the
  true beginning of a stream. Missing overlap later is a configuration error.
* ``position`` is a required absolute, nonnegative Python integer. It is the
  row coordinate used by the law, independent of the current window's length,
  batch number, and index label. A law containing ``n`` needs this coordinate
  even when the window already has its full width.
* Labels are optional hashable metadata and never determine the coordinate.
  Preserve account grouping, event order, deduplication, and the intended
  sampling grid in the source adapter. Count windows contain observed rows;
  a gap between timestamp labels does not invent missing observations.
* Use ``None`` for framework-specific null sentinels. Python float NaN, decimal
  NaN, and SymPy NaN also yield missing observations. The pandas adapter converts
  its own null sentinels to ``None``. Numeric text must be parsed by the adapter;
  it is never evaluated as Python or SymPy source.

Missing mapping fields, wrong row/window sizes, and invalid positions raise
``ModelInputError``. Malformed scalar cells instead produce domain violations
when their checks apply. Schema binding is required even for assumed rows;
initial scalar values themselves remain assumptions.

Every check is visible
----------------------

.. code-block:: pycon

   >>> opening = validator.check_window(((100, 0),), position=0)
   >>> opening.valid, opening.complete, opening.assumed
   (None, False, True)
   >>> [(check.field, check.kind, check.verdict, check.start) for check in opening.checks]
   [('cash', 'domain', 'assumed', 1), ('debit', 'domain', 'assumed', 1), ('cash', 'recurrence', 'assumed', 1)]

Each ``CheckResult`` retains its field, rule, kind, first applicable ``start``
position, observed and expected values, residual, verdict, and reason. Verdicts
are ``assumed``, ``satisfied``, ``violated``, or ``unknown``. Assumed checks have
not run; unknown checks ran without enough information to decide.

``WindowResult.valid`` is ``False`` if any check proves a violation, ``True``
only when every check passes, and ``None`` otherwise. ``complete`` requires every
check to have run and be decided; it can be true for an invalid row. A proved
violation still gives ``False`` when some other checks are unknown. The
``assumed`` property means that every check at this position is initial history.
``first_invalid is None`` alone never establishes coverage.

Multiple laws keep their own starts. ``first_test_position`` is the earliest
start; ``fully_testable_position`` is when every law has started, before checking
its actual data. The declared start may be later than the largest lag. A shorter
law can find a break before another law starts, so external frameworks should
emit partial prefix windows too. Keeping only full-width windows can discard
that evidence. ``iter_results`` handles those prefixes automatically.

Streams, batches, and Polars
----------------------------

``iter_results`` takes any ordered iterable of rows. It emits one ``WindowResult``
per input row, including initial assumptions, using a bounded deque:

.. code-block:: pycon

   >>> from itertools import chain
   >>> batches = (((100, 0), (90, 10)), ((85, 2),), ((83, 2),))
   >>> results = list(validator.iter_results(chain.from_iterable(batches)))
   >>> [result.valid for result in results]
   [None, True, False, True]

Each call starts position zero and fresh history. To preserve continuity across
batches, chain their iterators into one call. For a framework that owns its own
count window, keep the last ``p`` rows and the absolute position across batches:

.. code-block:: python

   from collections import deque

   history = deque(maxlen=validator.window_size)
   for position, row in enumerate(source_rows):
       history.append(row)
       result = validator.check_window(tuple(history), position=position)
       consume(result)  # the service's own output or policy

Maintain separate histories and position counters for independent partitions.
A restart must restore the overlap and absolute position or deliberately begin
a new partition; restarting enumeration inside each batch changes the law.
For continuous streams consume individual results, rather than waiting for a
finite audit to finish.

Polars can provide the same tuples without a pandas conversion:

.. code-block:: python

   rows = polars_frame.select(list(validator.fields)).iter_rows()
   for result in validator.iter_results(rows):
       consume(result)

``select`` preserves the requested projection and ``iter_rows`` supplies Python
row tuples; see the `Polars selection API
<https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.select.html>`_
and `row iterator
<https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.iter_rows.html>`_.
Checks execute in Python/SymPy; Polars supplies the data. No Polars dependency
or native execution engine is required by the validator.

Coverage before acceptance
--------------------------

For a finite job, ``audit`` scans the complete input and retains counts plus the
first violation. It never short-circuits, because later rows affect coverage:

.. code-block:: pycon

   >>> audit = validator.audit(((100, 0), (90, 10), (88, 2)))
   >>> audit.row_count, audit.checked_rows, audit.assumed_rows, audit.unknown_rows, audit.violated_rows
   (3, 2, 1, 0, 0)
   >>> audit.first_invalid, audit.valid, audit.complete
   (None, None, False)

These transitions passed, but the initial observation was assumed. The strict
whole-input verdict stays ``None``. Empty and history-only inputs also cannot
claim complete success. ``checked_rows`` counts fully decided rows, including
fully checked invalid rows; ``assumed_rows`` counts wholly assumed initial rows;
``unknown_rows`` counts other rows without a decided validity. ``violated_rows``
counts rows with at least one proved violation, even if other checks are missing.

A service can deliberately accept a trusted prefix while still requiring at
least one fully checked transition and rejecting undecidable later rows:

.. code-block:: python

   audit = validator.audit(source_rows)
   if audit.violated_rows:
       reject(audit.first_invalid)
   elif audit.unknown_rows or audit.checked_rows == 0:
       handle_insufficient_evidence(audit)
   else:
       accept_with_trusted_history(assumed_rows=audit.assumed_rows)

The policy functions belong to the consuming service. For acceptance with no
assumptions at all, require ``audit.valid is True`` instead.

``ValidationSummary.from_results`` aggregates an existing result iterator in
constant working memory. By default it requires every position from zero, in
order, including assumed prefixes. Missing, duplicate, or reordered positions raise
``ModelInputError``; otherwise a framework that drops initial results could
misrepresent partial output as a complete audit. For a periodic stream horizon,
explicitly name another ``start_position``:

.. code-block:: pycon

   >>> from modello.contrib.validation import ValidationSummary
   >>> horizon = ValidationSummary.from_results(results[2:], start_position=2)
   >>> horizon.start_position, horizon.stop_position, horizon.valid
   (2, 4, False)

The reported interval is ``[start_position, stop_position)``; its verdict covers
only those final-row results. Preceding history is outside the interval. An
unbounded stream has no final whole-input summary until the consumer defines a
finite horizon.

Scope and numerical values
--------------------------

The selected row model's scalar domains, definitions, aliases, and relative laws
are checked. Unrelated parent equations, aggregates, terminal/boundary constraints,
and other vectors are outside this adapter's validation scope. Keep per-row
inputs in the selected row schema. Missing values and parameters are not fitted
or filled. A passing row is evidence for these declared checks, conditional on
its supplied predecessors; it is not an independent guarantee about unmodelled
business rules or the preceding history.

Exact integers, ``Decimal``, ``Fraction``, and SymPy numbers preserve exact
arithmetic. Floating inputs retain their precision. Equation tolerances use
``abs(observed-expected) <= atol + rtol*abs(expected)``; domains remain strict.
Independent parameters and resolved constant definitions must satisfy their
field assumptions. Results retain SymPy values for exact evidence; a service
chooses its own representation at a JSON or other serialisation boundary.


API
---

.. currentmodule:: modello.contrib.validation

.. autoclass:: WindowValidator(model, rows, *, parameters=None, atol=0, rtol=0)
   :members: fields, field_paths, order, window_size, first_test_position, fully_testable_position, check_window, iter_results, audit

.. autoclass:: WindowResult
   :members: valid, complete, assumed, first_invalid

.. autoclass:: CheckResult

.. autoclass:: ValidationSummary
   :members: valid, complete, from_results, stop_position

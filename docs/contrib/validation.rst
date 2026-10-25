Observed recurrence windows
===========================

The :doc:`account walkthrough <../model-vectors>` calculates balances from
trades. When a statement already contains those balances, check that each
posting agrees with the preceding balance and the recorded cash movement.
``WindowValidator`` reports the accounting breaks while keeping the supplied
observations intact.

The nested ledger in ``examples.accounting_breaks`` follows a purchase, sale,
and repurchase. Amounts are integer cents. Its cash and share balances obey

.. math::

   \widehat C_n = C_{n-1} + \Delta C_n, \qquad
   \widehat Q_n = Q_{n-1} + \Delta Q_n.

For cash, the residual is :math:`e_n=C_n-\widehat C_n`. The sale below records
90,990 cents against an expected :math:`79{,}999+10{,}999=90{,}998`: an eight-cent
break.

.. list-table:: Cash columns from the ledger
   :header-rows: 1

   * - Event
     - Cash
     - Cash movement
     - Valid
     - Why
   * - opening
     - 100,000
     - 0
     - Unknown
     - Opening balance is assumed.
   * - purchase
     - 79,999
     - -20,001
     - Yes
     -
   * - sale
     - 90,990
     - 10,999
     - No
     - Expected 90,998; eight-cent shortfall.
   * - repurchase
     - 80,489
     - -10,501
     - Yes
     - Uses the recorded sale balance.

Find an accounting break
------------------------

Reuse the ledger's row schema and recurrence laws:

.. doctest::

   >>> from collections import deque
   >>> from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
   >>> from modello.contrib.validation import WindowValidator
   >>> rows = accounting_rows()
   >>> validator = WindowValidator(Ledger, "rows")
   >>> validator.order, validator.window_size
   (1, 2)
   >>> history = deque(maxlen=validator.window_size)
   >>> for position, (label, row) in enumerate(zip(EVENTS, rows, strict=True)):
   ...     history.append(tuple(row[field] for field in validator.fields))
   ...     result = validator(tuple(history), position=position, index=label)
   ...     if result.first_invalid is not None:
   ...         problem = result.first_invalid
   ...         print(problem.index, problem.field, problem.expected, problem.residual)
   ...         break
   sale cash 90998 -8

The callable checks the final row of an oldest-first window; ``check_window``
is the equivalent named method. Each transition uses recorded predecessors,
so a later row can pass even after an earlier break. The ledger also checks
share movements, holding values, and net asset values from its row schema.

For a law of order :math:`p`, the window contains :math:`p+1` observations.
The first test is normally at zero-based position :math:`p`; the first
:math:`p` rows supply assumed history. Each law retains its declared start:
``first_test_position`` is the earliest start and ``fully_testable_position``
is when all laws have started.

Read the results
-----------------

Scan an ordered iterable to obtain one result per observation:

.. doctest::

   >>> results = list(validator.iter_results(rows, labels=EVENTS))
   >>> [result.valid for result in results]
   [None, True, False, True]

``valid`` is ``False`` for a proved violation, ``True`` when all current-row
checks pass, and ``None`` for assumed history or insufficient evidence.
``complete`` means every current-row check ran and was decided, including
violations. It describes that row, not the validation of its predecessors.

Each ``CheckResult`` records the field, rule, first applicable ``start``,
observed and expected values, residual, and reason. Its verdict is
``assumed``, ``satisfied``, ``violated``, or ``unknown``; ``assumed`` means the
check has not started, while ``unknown`` means it ran without a decided outcome.
``first_invalid is None`` alone does not establish complete validation.
The :doc:`DataFrame recipes <../recipes/dataframe-validation>` show nullable
validity columns and first/all-error selection with pandas and Polars.

Supply windows or streams
--------------------------

* Supply flat mappings keyed by ``validator.fields``, as above, or row tuples
  in that exact order. ``field_paths`` exposes nested paths for selecting
  source columns. Extra mapping keys are ignored.
* A window must contain exactly ``min(position+1, validator.window_size)`` rows.
  Include short initial windows: laws with earlier starts can already find
  breaks there. ``position`` is a required absolute nonnegative Python integer.
* Labels are hashable reporting metadata. Windows follow row order; gaps in
  timestamp labels do not insert observations. Keep independent accounts in
  separate partitions with their own positions and history.
* Use ``None`` for framework-specific nulls. Numeric NaN also means missing;
  parse numeric text before validation. Wrong row/window sizes, missing fields,
  and invalid positions raise ``ModelInputError``. Malformed scalar cells
  produce domain violations when their checks apply.

``iter_results`` retains at most ``window_size`` rows. Each call starts fresh
at position zero; chain batches into one call to preserve continuity. On Python
3.12+, :func:`itertools.batched` splits a source iterable into batches:

.. testcode::

   from itertools import batched, chain

   batch_size = validator.window_size
   batches = batched(rows, batch_size)
   batch_results = validator.iter_results(chain.from_iterable(batches))
   print([result.valid for result in batch_results])

.. testoutput::

   [None, True, False, True]

Batch size is a source setting; it need not equal ``window_size``. Python
3.10–3.11 can use the equivalent :func:`itertools.islice` recipe in the
`Python batching documentation <https://docs.python.org/3/library/itertools.html#itertools.batched>`_.

If your stream processor supplies windows, call ``validator(window, position=n)``
directly. Keep the overlapping :math:`p` rows and absolute position across
batches or restarts. The validator stores no history between these calls.

Audit coverage
---------------

``audit`` scans a finite input completely and reports coverage plus the first
violation. Audit through the first position where all laws apply; this example
has a passing transition but still retains its assumed prefix:

.. doctest::

   >>> prefix = rows[:validator.fully_testable_position + 1]
   >>> audit = validator.audit(prefix)
   >>> audit.row_count, audit.checked_rows, audit.assumed_rows, audit.unknown_rows
   (2, 1, 1, 0)
   >>> audit.first_invalid, audit.valid, audit.complete
   (None, None, False)

``checked_rows`` counts fully decided rows, ``assumed_rows`` counts wholly
assumed history, and ``violated_rows`` counts rows with a proved break.
``unknown_rows`` counts remaining rows whose validity is undecidable.
For strict acceptance require ``audit.valid is True``. A policy that trusts
initial history can instead require no violated or unknown rows and at least
one fully checked transition. Empty and history-only inputs provide no such
transition.

``ValidationSummary.from_results(results)`` audits an existing result iterator.
It requires every position from zero in order, rejecting gaps, duplicates,
and omitted prefixes. To audit another contiguous interval, supply an explicit
``start_position``; ``[start_position, stop_position)`` then names its coverage.

Scope and numerical values
---------------------------

Validation covers the selected row schema's domains, definitions, aliases,
and fixed causal laws. Put per-row inputs in that schema; other vectors and
parent aggregate or boundary constraints are outside this scope. Missing
values and parameters are never inferred or filled.

Integers, ``Decimal``, ``Fraction``, and SymPy numbers retain exact arithmetic.
Equation tolerances use ``abs(observed-expected) <= atol + rtol*abs(expected)``;
domains remain strict. Supply independent constants through ``parameters``;
they and their resolved definitions must satisfy their field assumptions.
Results retain SymPy values for exact evidence.


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

Validate recorded rows
======================

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

The cash residual is :math:`e_n=C_n-\widehat C_n`, here :math:`29-30=-1`.

.. list-table:: Cash columns from the ledger
   :header-rows: 1

   * - Event
     - Cash
     - Cash movement
     - Valid
     - Why
   * - opening
     - 40
     - 0
     - Assumed
     - Opening balance is assumed.
   * - purchase
     - 20
     - -20
     - Yes
     -
   * - sale
     - 29
     - 10
     - No
     - Expected 30; shortfall 1.
   * - repurchase
     - 19
     - -10
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
   >>> [result.index for result in results if result.invalid]
   ['sale']

Use ``invalid`` for proven failures; ``not result.valid`` also includes assumed
or undecidable rows, whose ``valid`` is ``None``. Checks before a law's start are
assumed. An order-:math:`p` law normally starts at position :math:`p`, using :math:`p+1`
rows. ``first_test_position`` and ``fully_testable_position`` expose the earliest
and latest declared starts. Later checks use recorded predecessors, so the
repurchase passes after the sale fails.

What a result contains
----------------------

A ``ValidationResult`` contains the zero-based ``position``, the supplied
``index`` label (defaulting to the position), and a tuple of ``CheckResult``
objects in ``checks``. Inspect the sale's failing check:

.. doctest::

   >>> sale = next(result for result in results if result.invalid)
   >>> sale.position, sale.index
   (2, 'sale')
   >>> problem = next(check for check in sale.checks if check.invalid)
   >>> problem.field, problem.rule, problem.kind
   ('cash', 'posting', 'recurrence')
   >>> problem.observed, problem.expected, problem.residual
   (29, 30, -1)
   >>> problem.reason
   'observed differs from expected'

``checks`` includes passing, failing, assumed, and undecidable checks.
``complete`` means every check reached a decision, including failures:

.. csv-table:: Row outcomes
   :header: Outcome, ``valid``, ``invalid``, ``assumed``, ``complete``

   All checks pass, ``True``, ``False``, ``False``, ``True``
   A check fails, ``False``, ``True``, ``False``, Varies
   Only assumed history, ``None``, ``False``, ``True``, ``False``
   Undecided with no failure, ``None``, ``False``, ``False``, ``False``

Each ``CheckResult`` identifies its ``field``, named ``rule``, and ``kind``
(``domain``, ``definition``, ``alias``, or ``recurrence``), along with the row's
position and label. ``valid``, ``invalid``, and ``assumed`` describe the individual
check; ``start`` is its first applicable position.

Equation residuals are observed minus expected. Assumed checks have no expected
value or residual; unresolved dependencies can also leave them unknown.
Domain checks report a numeric set as ``expected`` and no residual, while
checking all field assumptions. ``observed`` captures accepted values as SymPy
scalars when read; malformed cells retain their supplied value for inspection.

Audit coverage
---------------

For a finite job, ``audit`` scans all rows and summarises their coverage:

.. doctest::

   >>> audit = validator.audit(rows, labels=EVENTS)
   >>> audit.row_count, audit.valid_rows, audit.invalid_rows, audit.assumed_rows, audit.unknown_rows
   (4, 2, 1, 1, 0)
   >>> audit.valid, audit.complete
   (False, False)
   >>> audit.first_invalid == sale
   True

The four row counts partition the input and sum to ``row_count``.
``first_invalid`` is a ``ValidationResult`` or ``None`` if no failure was proven;
the summary stores no other row results. ``complete`` requires every check in a
nonempty interval to be decided, so the assumed opening keeps it false here.

``audit.valid`` is ``False`` for a proven failure, ``True`` only for a nonempty
interval of passing rows, and otherwise ``None``. A policy that trusts initial
history can instead require no invalid or unknown rows and at least one valid
row; empty and history-only inputs do not qualify.

To reuse collected results, call ``ValidationSummary.from_results(results)``.
Positions must be contiguous from zero, or from an explicit ``start_position``
for a later interval. ``stop_position`` is the exclusive end.

Input and numerical values
---------------------------

Supply flat mappings keyed by ``validator.fields``, or row tuples in that order.
Convert framework-specific nulls to ``None``; numeric NaN also means missing.
Parse numeric text before validation. See the :doc:`DataFrame recipes
<../recipes/dataframe-validation>` for pandas, Polars, and MultiIndex columns.

Validation checks the row schema's domains, definitions, aliases, and fixed
causal laws. Put per-event inputs in that schema: other vectors and parent
aggregate or boundary constraints are outside its scope. Missing values and
parameters are never inferred or filled.

Calculus and row-local series or process queries use the core model's operator
semantics. Ordinary equations use recorded field values.

Integers, ``Decimal``, ``Fraction``, and exact SymPy numbers retain exact arithmetic.
Equation tolerances use ``abs(observed-expected) <= atol + rtol*abs(expected)``;
domains remain strict. Scalar ``parameters`` must be independent model fields
with valid domains and consistent aliases; omit the row-vector size, which
comes from the input. Resolvable scalar definitions, including unused constants,
are checked too. Conflicting or undecidable alias definitions are rejected.

Streams and manual windows
---------------------------

Consume ``iter_results`` directly for a stream. Flatten non-overlapping batches
into one row iterator before calling it: each call resets history and starts at
position zero. Batch sizes are independent of ``window_size``; independent
accounts need separate histories.

For externally managed windows or checkpoints, call the validator with the
required history and absolute position. For example, recheck the sale:

.. doctest::

   >>> position = sale.position
   >>> window = rows[max(0, position + 1 - validator.window_size):position + 1]
   >>> validator(window, position=position, index=sale.index) == sale
   True

Each call checks only the final row. Supply an oldest-first window of exactly
``min(position+1, validator.window_size)`` rows, including short initial windows.
``position`` is a nonnegative Python integer counting rows from the stream's
start; index labels and gaps do not change it.


API
---

.. currentmodule:: modello.contrib.validation

.. autoclass:: WindowValidator(model, rows, *, parameters=None, atol=0, rtol=0)
   :members: fields, field_paths, order, window_size, first_test_position, fully_testable_position, iter_results, audit
   :special-members: __call__

.. autoclass:: ValidationResult
   :members: valid, invalid, complete, assumed

.. autoclass:: CheckResult
   :members: invalid, assumed

.. autoclass:: ValidationSummary
   :members: valid, invalid, complete, row_count, from_results, stop_position

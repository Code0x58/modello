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
   >>> for result in results:
   ...     if result.invalid:
   ...         print(result.index)
   ...         break
   sale

``valid`` is ``True`` when all current-row checks pass, ``False`` for a proven
violation, and ``None`` for assumed history or insufficient evidence.
Checks before a law's declared start are assumed; an order-:math:`p` law
normally starts at position :math:`p`, using the preceding :math:`p` rows.

Use ``result.invalid`` to identify proven failures; ``not result.valid`` also
includes assumed or undecidable rows.
Later rows use recorded predecessors, so the repurchase can pass after the sale
failed.

What a result contains
----------------------

Each window call returns a ``ValidationResult``; ``iter_results`` yields one
for every input row. It contains the row's zero-based ``position``, its supplied
``index`` label, and a tuple of ``CheckResult`` objects in ``checks``.
For example, inspect the sale's failing check:

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
   >>> sale.valid, sale.invalid, sale.complete, sale.assumed
   (False, True, True, False)

Here ``complete`` is true because every check reached a decision, even though
one failed. ``checks`` also includes passing checks and checks that could not
run, so callers can inspect a failure without losing coverage information.

.. list-table:: Result flags
   :header-rows: 1

   * - Row outcome
     - ``valid``
     - ``invalid``
     - ``assumed``
     - ``complete``
   * - All checks pass
     - ``True``
     - ``False``
     - ``False``
     - ``True``
   * - A check fails
     - ``False``
     - ``True``
     - ``False``
     - Varies
   * - Assumed history
     - ``None``
     - ``False``
     - ``True``
     - ``False``
   * - Undecided, no failure
     - ``None``
     - ``False``
     - ``False``
     - ``False``

A ``CheckResult`` has the same ``valid``, ``invalid``, and ``assumed`` flags,
plus its row position and label. ``start`` names the first position where that
check applies. ``expected`` and ``residual`` are ``None`` for assumed checks;
an unresolved dependency can also leave ``expected`` unknown. Domain checks
have a numeric set as ``expected`` and no residual; all declared field
assumptions are checked, including restrictions not expressed by that set.

For accepted scalars, ``observed`` retains the converted SymPy value captured
when the row was read. Reusing a mutable source cell cannot change earlier
numeric evidence. Malformed cells retain their supplied value for inspection.

The core :class:`~modello.diagnostics.CandidateCheck` is a different record,
returned by :meth:`~modello.diagnostics.ModelProblem.check`: it contains an
assignment (``candidate``), equation ``residuals``, domain ``assumptions``, and
an overall ``verdict``. The window API adds row labels, rule names, and explicit
history markers to that checking machinery and returns the types shown above.

Audit coverage
---------------

For a finite job, ``audit`` scans all rows and summarises their coverage:

.. doctest::

   >>> audit = validator.audit(rows, labels=EVENTS)
   >>> audit.row_count, audit.valid_rows, audit.invalid_rows, audit.assumed_rows, audit.unknown_rows
   (4, 2, 1, 1, 0)
   >>> audit.valid
   False
   >>> audit.first_invalid == sale
   True

The valid, invalid, assumed, and unknown row counts partition the input;
``row_count`` is their sum. ``first_invalid`` is the first invalid row result,
or ``None`` if no row was proven invalid. An invalid row may still contain
undecidable checks; ``complete`` describes coverage of all checks.

For strict acceptance, require ``audit.valid``. Even with no breaks,
assumed or undecidable rows keep that outcome at ``None``. A policy that trusts
initial history can require no invalid or unknown rows and at least one valid
row. Empty and history-only inputs provide no such row.

``ValidationSummary`` retains the counts and first invalid ``ValidationResult``
without storing every row.

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

Named definitions inside derivatives and integrals use the core model's
operator semantics. Ordinary equations continue to use recorded field values.

Integers, ``Decimal``, ``Fraction``, and exact SymPy numbers retain exact arithmetic.
Equation tolerances use ``abs(observed-expected) <= atol + rtol*abs(expected)``;
domains remain strict. Independent constants supplied through ``parameters``
must satisfy their field assumptions, and parameter aliases must agree.
Resolvable scalar definitions are also checked as configuration, including
unused constants. Conflicting or undecidable alias definitions are rejected.
Results retain SymPy values for exact evidence.

Streams and manual windows
---------------------------

For a stream, iterate over ``iter_results`` without collecting a list; it
maintains the rolling history. For a source that delivers non-overlapping
batches of rows, flatten those batches into one row iterator and pass it to a
single ``iter_results`` call. A new call resets history and starts at position
zero. Delivery batch sizes are independent of ``window_size``.

If a DataFrame or stream processor already supplies overlapping windows, pass
each window directly to the validator. For example, recheck the sale:

.. doctest::

   >>> position = sale.position
   >>> window = rows[max(0, position + 1 - validator.window_size):position + 1]
   >>> validator(window, position=position, index=sale.index) == sale
   True

Each call checks the final row of an oldest-first window containing exactly
``min(position+1, validator.window_size)`` rows. Include short initial windows;
laws with earlier starts can already find breaks there. ``position`` is an
absolute nonnegative Python integer; gaps in labels do not insert rows.
Use separate histories for independent accounts. To resume from a checkpoint,
pass the required window and absolute position to the callable.


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

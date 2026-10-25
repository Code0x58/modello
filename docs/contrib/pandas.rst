Validating DataFrame histories (contrib)
========================================

Use ``modello.contrib.pandas`` to check existing observations against a row
model and its relative recurrence laws. Install the optional dependency:

.. code-block:: console

   $ pip install 'modello[pandas]'

Declare the row once, then reuse ordinary :doc:`model-vector laws
<../model-vectors>`. The frame supplies the finite window and its initial
history; validation checks observations without constructing or solving models.

A cash roll-forward
-------------------

.. code-block:: pycon

   >>> import pandas as pd
   >>> from modello import Index, InstanceDummy, Modello, model_recurrence, model_vector
   >>> from modello.contrib.pandas import FrameValidator
   >>> class CashRow(Modello):
   ...     cash = InstanceDummy("cash", real=True)
   ...     debit = InstanceDummy("debit", real=True)
   >>> class CashLedger(Modello):
   ...     N = InstanceDummy("N", integer=True, nonnegative=True)
   ...     rows = model_vector(CashRow, size=N)
   ...     n = Index("n")
   ...     posting = model_recurrence(rows[n], cash=rows[n-1].cash - rows[n].debit)
   >>> frame = pd.DataFrame({"cash": [100, 90, 85, 83], "debit": [0, 10, 2, 2]},
   ...                      index=["opening", "purchase", "sale", "fee"])
   >>> validator = FrameValidator(CashLedger, "rows")
   >>> report = validator.validate(frame)
   >>> validator.window_size, report.valid
   (2, False)
   >>> report.breaks[["position", "index", "field", "observed", "expected", "residual"]].values.tolist()
   [[2, 'sale', 'cash', 85, 88, -3]]

The opening row seeds the history; the default ``start=1`` checks the first
transition at row one. Here the debit belongs to the current row. A source
whose debit describes the following transition would instead use
``rows[n-1].debit``. The equation makes that convention explicit.

At the sale, the observed balance is three units below the expected balance.
The fee row then correctly follows the **observed** sale balance of 85.
Expected results never replace observations, so one break does not generate a
different projected history. A bad observation can still affect every equation
whose declared lag window includes it; the report does not assign a root cause.

Every row, including initial history, also receives the row model's scalar
definition and domain checks. For example, a modelled ``nav = cash + value``
is checked on the opening row too. No missing quantity is inferred or filled.

Nested accounts and MultiIndex columns
--------------------------------------

Nested models map directly to column paths. ``("holding", "quantity")``
observes ``row.holding.quantity``; a root scalar in that two-level index uses
``("cash", "")``. Trailing empty or missing levels pad shallower paths.
Flat dotted labels such as ``"holding.quantity"`` work too. Finite vector
fields use positions, for example ``("quantities", 0)`` or ``"quantities[0]"``.

The runnable accounting example retains holding valuation, NAV, and two
roll-forward equations in one schema:

.. literalinclude:: ../../examples/accounting_breaks.py
   :start-at: class Holding
   :end-before: def accounting_frame
   :language: python

.. code-block:: pycon

   >>> from examples.accounting_breaks import accounting_frame, find_breaks
   >>> accounts = accounting_frame()
   >>> accounts.columns.nlevels
   2
   >>> find_breaks(accounts).breaks[["index", "field", "observed", "expected", "residual"]].values.tolist()
   [['sale', 'cash', 90990, 90998, -8]]

Amounts in this example are integer minor currency units. The sale has an
eight-cent cash break; its holding, valuation, and NAV equations still agree
with their observed inputs. The subsequent purchase is locally correct.
Run it with ``uv run --extra pandas python -m examples.accounting_breaks``.

All scalar row fields, including derived fields and aliases, require columns.
Extra metadata columns are ignored. Column labels must be unique, and two
labels must not ambiguously name the same model field. Supply explicit mappings
when source labels differ:

.. code-block:: pycon

   >>> source = frame.rename(columns={"cash": "Cash Balance", "debit": "Debit Amount"})
   >>> mapped = validator.validate(source, columns={"cash": "Cash Balance", "debit": "Debit Amount"})
   >>> mapped.breaks["residual"].tolist()
   [-3]

Rows and index labels
---------------------

Validation follows positional frame order. Sort the source into its intended
event order first, and validate each independent account separately, for
example inside an ordinary pandas ``groupby`` loop. Index labels are retained
for reporting; duplicate, MultiIndex, and nonmonotonic labels are supported.
They do not declare groups, mathematical time, or a sampling interval.
The local recurrence coordinate ``n`` is the zero-based row position.

An order-``p`` law uses the first ``p`` rows as history and needs a window of
``p+1`` rows. Declare ``start=p`` to cover those lags. Multiple laws retain
their own start positions, while the shared window covers their largest lag.
An explicit later start postpones checks without enlarging the required window.
Rows before each start have no check for that law. A frame containing only
history has row checks but no transition checks; an empty frame has no checks.
``valid`` is true when all emitted checks pass, including the empty case.

Missing data, domains, and precision
------------------------------------

``report.checks`` contains satisfied, violated, and unknown checks.
``report.breaks`` selects violations; ``report.unknown`` selects undecided
checks. ``report.valid`` requires every check to be satisfied, so a report with
missing observations is not valid even if it has no proven break.

Missing cells (``None``, ``NaN``, ``pd.NA``, or ``NaT``) give unknown domain
checks and unknown equations wherever their values are required. Dependencies
in an inactive ``Piecewise`` branch do not prevent checking the active branch.
Missing **columns** raise ``ModelInputError`` because the frame cannot bind the
declared schema. Symbolic observations remain unknown numeric values. Invalid
scalar cells, nonfinite observations, and incompatible field assumptions give
domain violations. Undefined expressions, including division by zero, give
equation violations. Text cells are never evaluated as Python or SymPy input;
convert numeric source text before checking it.

The report records ``position``, ``index``, ``field``, ``rule``, ``observed``,
``expected``, ``residual``, ``verdict``, and ``reason``. An equation's residual is
observed minus expected. ``expected`` is absent for a missing dependency; domain
checks instead show a numeric search domain, with all declared assumptions
checked separately. Checks do not establish the validity of unobserved values
between rows.

Integer minor units, ``Decimal``, ``Fraction``, and exact SymPy numbers preserve
exact arithmetic. Decimal amounts convert to exact rationals. Floating values
retain their floating precision, and comparison is exact by default. Set
``atol`` and/or ``rtol`` explicitly for approximate inputs; an equation passes
when ``abs(observed - expected) <= atol + rtol * abs(expected)``. Tolerances
apply to equations, while field assumptions remain strict. The report retains
the residual even when tolerance accepts it.

Scalar model parameters can be fixed when preparing the validator:

.. code-block:: python

   validator = FrameValidator(InterestLedger, "rows", parameters={"retention": 0.99}, atol=1e-8)

Parameters must name independent scalar model fields and contain finite numeric
values compatible with their assumptions. Constant definitions in the parent
model, including formulas in supplied parameters, expand before validation.
The frame supplies the row vector's length; do not supply its size parameters.

Execution and scope
-------------------

The adapter reads the selected vector's row schema and source
``model_recurrence`` declarations. It replaces current and delayed field
references with observation slots once. Each row then needs only scalar
substitution and comparison. Existing data supplies every referenced quantity,
so coupled current-row equations need no solve or topological ordering.

``iter_checks(frame)`` yields ``FrameCheck`` records while retaining at most
``window_size`` rows. ``validate(frame)`` collects those records into a pandas
report, whose memory grows with the number of checks. Both methods read tuples
without coercing a mixed row to a common numeric dtype. The frame is unchanged.
Each call starts fresh history; separate calls on chunks do not carry state.

Supported laws use current rows and fixed past lags of the selected vector.
Nonlinear equations, current-row cycles, fixed-width indexed sums, nested
targets, row definitions, and multiple laws share the same execution path.
Future rows, absolute or terminal row references, variable lags, and references
to other model vectors raise ``ModelInputError`` when preparing the validator.
Keep per-row inputs in the selected row model, including nested components.

This is an adapter of a selected schema and its relative laws: the frame
replaces the original vector horizon. It does not validate unrelated parent
equations, aggregate or terminal constraints, or infer model parameters.
Use an ordinary joint model solve when missing values or boundary conditions
need inference. The scalar ``Series``/``Recurrence`` query API remains separate;
this extension adapts typed model-vector laws to observed nested rows.


API
---

.. currentmodule:: modello.contrib.pandas

.. autoclass:: FrameValidator(model, rows, *, parameters=None, atol=0, rtol=0)
   :members: validate, iter_checks, window_size

.. autoclass:: FrameValidation
   :members: checks, valid, breaks, unknown

.. autoclass:: FrameCheck

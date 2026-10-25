Validating DataFrame histories (contrib)
========================================

Use ``modello.contrib.pandas`` to check existing observations against a row
model and its relative recurrence laws. Install the optional dependency:

.. code-block:: console

   $ pip install 'modello[pandas]'

For services, Polars, or a framework that supplies its own rolling windows,
use the :doc:`pandas-independent validator <validation>`. This module only binds
columns, normalises pandas nulls, and feeds that same window function.

An order-:math:`p` process relates the current observation to earlier rows and
current inputs:

.. math::

   x_n = f(n, x_{n-1}, \ldots, x_{n-p}, u_n), \qquad n \ge p.

The first :math:`p` rows supply assumed history. A window of :math:`p+1` rows
includes that history **and the observation being tested**. Validation uses
observed predecessors, rather than generating a replacement path or solving
for missing data. Declare the row once and reuse ordinary
:doc:`model-vector laws <../model-vectors>`.

A cash roll-forward
-------------------

Let :math:`C_n` be the observed cash balance and :math:`D_n` the debit posted
on row :math:`n`. The expected balance and accounting residual are

.. math::

   \widehat C_n = C_{n-1} - D_n, \qquad
   e_n = C_n - \widehat C_n, \qquad n = 1, \ldots, m-1.

This frame contains one break:

.. list-table:: Input frame
   :header-rows: 1

   * - index
     - cash
     - debit
   * - opening
     - 100
     - 0
   * - purchase
     - 90
     - 10
   * - sale
     - 85
     - 2
   * - fee
     - 83
     - 2

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
   >>> validator.validate(frame)
   opening      <NA>
   purchase     True
   sale        False
   fee          True
   Name: valid, dtype: boolean

The opening balance is assumed, so its indicator is missing. At the sale,
:math:`\widehat C_2 = 90-2 = 88` while :math:`C_2=85`, giving
:math:`e_2=-3`. The fee row then correctly follows the **observed** sale balance:
:math:`85-2=83`. Expected values never replace observations. A bad observation
can affect every equation whose lag window includes it; the report does not
assign a root cause.

The default ``start=1`` tests the first transition at position one. Here the
debit belongs to the current row. A source whose debit describes the following
transition would instead use ``rows[n-1].debit``.

Find the first invalid point
----------------------------

``first_invalid(frame)`` returns the first proven violation as a ``FrameCheck``
and stops without reading later rows:

.. code-block:: pycon

   >>> problem = validator.first_invalid(frame)
   >>> problem.position, problem.index, problem.field
   (2, 'sale', 'cash')
   >>> problem.observed, problem.expected, problem.residual
   (85, 88, -3)
   >>> validator.first_invalid(frame.iloc[:1]) is None
   True

``None`` means **no violation was proved**. A valid observed history, a frame
containing only assumed initial rows, or missing dependencies can all produce
``None``. It does not mean that every row was tested successfully. The boolean
Series shows which rows could be validated; the detailed report explains
individual checks:

.. code-block:: pycon

   >>> checks = validator.report(frame)
   >>> breaks = checks.loc[checks["verdict"].eq("violated")]
   >>> breaks[["position", "index", "field", "observed", "expected", "residual"]].values.tolist()
   [[2, 'sale', 'cash', 85, 88, -3]]

Checks are ordered by row position: field domains first, then row definitions,
then recurrence equations in declaration order. This defines which problem is
returned if a row has several violations. Positions disambiguate duplicate
index labels.

For service acceptance decisions, use ``audit`` to expose coverage explicitly:

.. code-block:: pycon

   >>> good = frame.assign(cash=[100, 90, 88, 86])
   >>> coverage = validator.audit(good)
   >>> coverage.row_count, coverage.checked_rows, coverage.assumed_rows, coverage.unknown_rows
   (4, 3, 1, 0)
   >>> coverage.first_invalid, coverage.valid, coverage.complete
   (None, None, False)

The three transitions passed, but the opening row was assumed. The strict
whole-input verdict therefore remains ``None``. ``audit`` scans the entire
frame, including rows after a break; ``first_invalid`` is only a short-circuit
search. Empty or history-only frames cannot claim complete success. See the
:doc:`integration guide <validation>` for an explicit trusted-history policy.

Aligned indicators and assumed history
--------------------------------------

``validate(frame)`` returns a pandas nullable boolean Series named ``valid``,
with exactly the input index, order, and length. Missing indicators are
``pd.NA``, the missing value for pandas booleans; ``NaT`` is for temporal data.

.. list-table:: Row validity
   :header-rows: 1

   * - Indicator
     - Meaning
   * - ``True``
     - All recurrence laws have started and every check at this row passes.
   * - ``False``
     - At least one domain or equation check proves a violation.
   * - ``pd.NA``
     - Assumed initial history, a law that has not started, or an unresolved check.

A proven violation takes precedence over missing checks. Otherwise an
undecidable check prevents ``True``. pandas reductions such as ``all()`` skip
missing values by default, so an aggregate truth value is not evidence that
every indicator was known.

An order-``p`` law normally uses ``start=p``. The first ``p`` rows are assumed,
and the first tested row is position ``p``. A second-order law therefore needs
three observations for its first test. Shorter frames keep their entire index
and have only missing indicators; an empty frame gives an empty Series.
``first_invalid`` returns ``None`` in both cases.

Multiple laws retain their own starts. Domains and row-local definitions begin
at the earliest law's start; earlier records have an ``assumed`` verdict. If a later law has
not started, the combined indicator stays missing unless another check already
proves a violation. Setting a later start postpones checks without enlarging the
largest-lag window. Initial history is trusted, including its row definitions
and field domains; it is not independently audited.

Missing data can interrupt coverage and later allow it to resume:

.. code-block:: pycon

   >>> gappy = pd.DataFrame({"cash": [100, 90, None, 86, 84], "debit": [0, 10, 2, 2, 2]},
   ...                      index=[0, 3, 7, 10, 20])
   >>> validator.validate(gappy).tolist()
   [<NA>, True, <NA>, <NA>, True]
   >>> validator.first_invalid(gappy) is None
   True

At position two the balance is missing; the next transition needs that missing
predecessor. The final transition can again be checked from its observed pair.
No rows or labels are dropped.

Validation follows positional frame order. Sort into event order first and
validate independent accounts separately, for example in a pandas ``groupby``
loop. Gappy, duplicate, MultiIndex, and timezone-aware index labels are retained.
They do not imply groups, mathematical time, or a sampling interval: ``n`` is
the zero-based **row position**. A missing calendar date does not create a row.
For a daily process, reindex to the intended daily grid first so missing dates
become missing observations.

Nested accounts and MultiIndex columns
--------------------------------------

For holdings with quantity :math:`Q_n` and price :math:`P_n`, the example adds
valuation and NAV identities to cash and quantity roll-forwards:

.. math::

   V_n &= Q_n P_n, \\
   \mathrm{NAV}_n &= C_n + V_n, \\
   \widehat C_n &= C_{n-1} + \Delta C_n, \\
   \widehat Q_n &= Q_{n-1} + \Delta Q_n.

Nested fields map directly to column paths: ``("holding", "quantity")``
observes ``row.holding.quantity``. A root scalar in a two-level column index
uses ``("cash", "")``; trailing empty or missing levels pad shallower paths.
Flat dotted labels such as ``"holding.quantity"`` work too. Finite vector
fields use positions, for example ``("quantities", 0)`` or ``"quantities[0]"``.

.. literalinclude:: ../../examples/accounting_breaks.py
   :start-at: class Holding
   :end-before: def accounting_frame
   :language: python

.. code-block:: pycon

   >>> from examples.accounting_breaks import Ledger, accounting_frame, find_breaks
   >>> accounts = accounting_frame()
   >>> accounts.columns.nlevels
   2
   >>> FrameValidator(Ledger, "rows").validate(accounts).tolist()
   [<NA>, True, False, True]
   >>> find_breaks(accounts)[["index", "field", "observed", "expected", "residual"]].values.tolist()
   [['sale', 'cash', 90990, 90998, -8]]

Amounts are integer minor currency units. The sale has an eight-cent cash
break; its holding, valuation, and NAV equations agree with their observed
inputs. The subsequent purchase is locally correct. Run the example with
``uv run --extra pandas python -m examples.accounting_breaks``.

All scalar row fields, including derived fields and aliases, require columns.
Extra metadata columns are ignored. Labels must be unique and must not
ambiguously name the same field. Map different source labels explicitly:

.. code-block:: pycon

   >>> source = frame.rename(columns={"cash": "Cash Balance", "debit": "Debit Amount"})
   >>> validator.validate(source, columns={"cash": "Cash Balance", "debit": "Debit Amount"}).tolist()
   [<NA>, True, False, True]

Detailed checks and precision
-----------------------------

``report(frame)`` returns a DataFrame with ``position``, ``index``, ``field``,
``rule``, ``observed``, ``expected``, ``residual``, ``verdict``, and ``reason``.
It also includes ``kind`` and the first applicable ``start`` position for each
check. Verdicts are ``assumed``, ``satisfied``, ``violated``, or ``unknown``. An equation's residual
is observed minus expected. Missing dependencies have no expected value; domain
checks show a numeric search domain, with all declared assumptions checked
separately. Assumed checks remain visible with a reason and start metadata:

.. code-block:: pycon

   >>> checks.loc[checks["position"].eq(0), ["field", "kind", "verdict", "start"]].values.tolist()
   [['cash', 'domain', 'assumed', 1], ['debit', 'domain', 'assumed', 1], ['cash', 'recurrence', 'assumed', 1]]

The report does not establish what
happened between observed rows.

After the assumed prefix, missing cells (``None``, ``NaN``, ``pd.NA``, or
``NaT``) give unknown checks wherever their values are required. Dependencies
in an inactive ``Piecewise`` branch do not prevent checking the active branch.
Missing **columns** raise ``ModelInputError``, even for an empty or short frame.
Symbolic observations remain unknown numeric values. Invalid scalar cells,
nonfinite observations, and incompatible field assumptions give domain
violations. Undefined expressions, including division by zero, give equation
violations. Text cells are never evaluated as Python or SymPy input; convert
numeric source text before checking it.

Integer minor units, ``Decimal``, ``Fraction``, and exact SymPy numbers preserve
exact arithmetic. Decimal amounts convert to exact rationals. Floating values
retain their floating precision, and comparison is exact by default. Set
``atol`` and/or ``rtol`` for approximate inputs; an equation passes when

.. math::

   |x_n - \widehat x_n| \le \mathrm{atol} + \mathrm{rtol}\,|\widehat x_n|.

Tolerances apply to equations; field assumptions remain strict. The report
retains the residual even when tolerance accepts it. Independent scalar model
parameters can be fixed when preparing the validator:

.. code-block:: python

   validator = FrameValidator(InterestLedger, "rows", parameters={"retention": 0.99}, atol=1e-8)

Parameters must contain finite numeric values compatible with their assumptions.
Constant definitions in the parent model, including formulas in supplied
parameters, expand before checking. The frame supplies the row vector's length;
do not supply its size parameters.

Execution and scope
-------------------

``validator.window`` is the public
:class:`~modello.contrib.validation.WindowValidator` used by all adapter methods.
``FrameValidator.from_window(plan)`` wraps an existing prepared plan without
recompiling or changing its parameters and tolerances.
``iter_results(frame)`` projects columns in its required order, converts pandas
nulls to ``None``, and feeds its rolling iterator. The same ``check_window``
function is directly usable in another framework. Existing observations supply
the quantities, so coupled current-row equations need no solve or topological
ordering.

``iter_checks(frame)`` yields individual check records, including assumptions.
``iter_results`` retains at most ``window_size`` rows; ``first_invalid`` consumes
only as far as a proved violation. ``validate`` collects one nullable boolean per
row, ``report`` collects all checks, and ``audit`` retains coverage counts plus
the first violation. All methods read tuples without coercing mixed rows to a
common numeric dtype. Each call starts fresh history; separate calls on chunks
do not carry state. Use the independent iterator over chained batches or keep
the overlap and absolute position in the consuming framework.

Supported laws use current rows and fixed past lags of the selected vector.
Nonlinear equations, current-row cycles, fixed-width indexed sums, nested
targets, row definitions, and multiple laws share the same execution path.
Future rows, absolute or terminal references, variable lags, and references to
other vectors raise ``ModelInputError`` during preparation. Keep per-row inputs
in the selected row model, including nested components.

The frame replaces the original vector horizon. The adapter does not validate
unrelated parent equations or aggregate/terminal constraints, or infer model
parameters. Use a joint model solve for inference from missing values or
boundary conditions. Scalar ``Series``/``Recurrence`` queries remain separate;
this extension adapts typed model-vector laws to observed nested rows.


API
---

.. currentmodule:: modello.contrib.pandas

.. autoclass:: FrameValidator(model, rows, *, parameters=None, atol=0, rtol=0)
   :members: from_window, window, first_invalid, validate, report, audit, iter_results, iter_checks, window_size, first_test_position

``FrameCheck`` is an alias of
:class:`modello.contrib.validation.CheckResult` for pandas consumers.

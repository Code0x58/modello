8. Build attribution, utility, and report views
===============================================

The account already produces interval profit using the models from chapter 5.
Reporting adds named explanations of that profit, then presents those same
observations as tables and timelines. We will declare the attribution rule,
build its inputs, derive utility views, and only then assemble a full report.

Continue with the account and decision inputs from chapters 5 and 6. This setup
recreates them for an independent run of this page:

.. raw:: html

   <details><summary>Recreate the account and decision inputs</summary>

>>> from datetime import timedelta
>>> from pathlib import Path
>>> import examples.portfolio as portfolio_examples
>>> from examples.portfolio.explicit import Portfolio, Basket, EquityMembers
>>> from examples.portfolio.basket import equity
>>> from examples.portfolio.models import USDCash
>>> from examples.portfolio.account import Account
>>> from examples.portfolio.sources import CSVTransactionSource, timestamp
>>> times = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
>>> opening = Portfolio(
...     "opening",
...     basket=Basket(
...         "basket",
...         members=EquityMembers(
...             "equities",
...             aapl=equity("apple", quantity=10),
...             msft=equity("microsoft", quantity=10),
...         ),
...     ),
...     cash=USDCash("cash", amount=1000),
... )
>>> account = Account("research", times[0], opening)
>>> age = timedelta(days=4)
>>> market = opening.load_market()
>>> data_dir = Path(portfolio_examples.__file__).with_name("data")
>>> ledger = CSVTransactionSource(data_dir / "transactions.csv").load()
>>> from examples.compute.named_map import NamedMap
>>> from examples.portfolio.signals import MomentumSignals, TargetWeights
>>> path = account.replay(ledger, times, market=market, max_price_age=age)
>>> decision_at = timestamp("2026-01-07T09:00:00Z")
>>> arrival_at = timestamp("2026-01-07T09:05:00Z")
>>> decision = account.replay(
...     ledger, (times[0], decision_at), market=market, max_price_age=age
... ).snapshots[-1]
>>> prices = market.map(
...     lambda history: history.sample((*times[:2], decision_at), max_age=age)
... )
>>> signals = MomentumSignals(prices, lookback=2).load()
>>> targets = TargetWeights.from_signals(signals, decision_at, exposure=0.8)
>>> orders = decision.rebalance(targets, batch="rebalance-1", arrival_at=arrival_at)

.. raw:: html

   </details>

Declare the attribution convention
----------------------------------

A signal's P&L is not identified uniquely by a transaction log. Choose an
explicit convention: keep the opening book unchanged as a reference, and track
signed incremental shares from signal-linked fills in separate sleeves.

An order records which decision a fill belongs to. ``OrderAttribution`` adds
nonnegative signal-credit weights that sum to one:

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: OrderAttribution
   :language: python

Declare those weights alongside the decision, before using its execution
outcome. For our one-signal policy:

>>> from examples.portfolio.reporting import OrderAttribution
>>> attributions = orders.map(
...     lambda order: OrderAttribution(order, NamedMap({"momentum": 1.0}))
... )
>>> attributions["rebalance-1:aapl"].weights["momentum"]
1.0

A multi-signal decision could use ``NamedMap({"value": 0.4, "momentum": 0.6})``.
The reporting adapter checks the named histories were available at the decision
and that linked fills match the account, instrument, timing, side, and quantity.

Evaluate an executed sleeve with the existing model
---------------------------------------------------

Only executed shares enter the sleeve. Its shares persist into later intervals,
fees follow the declared credit weights, and instrument income follows the
signed shares held at the income event. Unlinked trades and account cash income
remain ``unattributed``; income after a position has closed has no current-share
allocation basis and also enters that row.

For example, Apple's momentum sleeve starts empty, acquires seven executed
shares for USD 735.60, receives USD 3.50 of the income, and pays USD 1 fees.
The same ``InstrumentPnL`` definition used by actual account positions gives:

>>> from sympy import Rational
>>> from examples.portfolio.pnl import InstrumentPnL
>>> sleeve = InstrumentPnL(
...     "momentum",
...     opening_quantity=0,
...     opening_price=100,
...     closing_price=107,
...     fill_quantity=7,
...     fill_notional=Rational("735.6"),
...     income=Rational("3.5"),
...     fees=1,
... )
>>> sleeve.net
159/10

Selling ten opening MSFT shares creates a negative signal sleeve against the
unchanged positive reference book. Their quantities add to the actual holding,
even after it reaches zero. The signal sleeve measures the incremental P&L of
selling rather than continuing to hold; funding and position constraints apply
to the combined actual account.

Supply interval columns to the reporting layer
----------------------------------------------

An instrument report keeps the actual position path, profit columns, per-source
profit columns, and per-source quantities:

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: InstrumentReport
   :language: python

``AccountReport`` collects the instrument views and account/source columns.
It retains the plans' symbolic expressions for inspection. Utility views are
optional at this stage; we will populate them below:

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: AccountReport
   :end-before:     def to_dict
   :language: python

The ledger adapter maintains ``(member, source)`` quantities. For each interval
it records opening quantities/prices, signed fill quantities/notionals, closing
prices, income, and fees. Those are precisely ``InstrumentPnL``'s inputs from
chapter 5. The compiled model supplies each source's profit columns.

To add instrument and account columns, use an axis-checked reduction:

.. literalinclude:: ../../../examples/portfolio/pnl.py
   :pyobject: Profit.sum
   :dedent: 4
   :language: python

Source totals feed ``AccountPnL`` together with opening NAV, closing NAV, and
external capital. A nonzero reconciliation residual indicates a mismatch.
The complete adapter adds validation and the explicit income-allocation cases:

.. raw:: html

   <details><summary>Inspect the complete ledger-to-sleeves adapter</summary>

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: AccountReport.from_account
   :dedent: 4
   :language: python

.. raw:: html

   </details>

Apply that layer with no utility definitions yet:

>>> report = path.report(
...     attributions, signals=NamedMap({"momentum": signals}), utilities=NamedMap({})
... )
>>> {
...     source: round(sum(columns["net"].values), 2)
...     for source, columns in report.sources.items()
... }
{'opening': 55.0, 'signal:momentum': 11.9, 'unattributed': 0.0}
>>> {
...     member: round(sum(item.profit["net"].values), 2)
...     for member, item in report.instruments.items()
... }
{'aapl': 90.9, 'msft': -24.0}
>>> tuple(round(value, 10) for value in report.account["residual"].values)
(0.0, 0.0, 0.0)

Opening holdings contribute USD 55.00 and momentum contributes USD 11.90,
reconciling to USD 66.90. Momentum's instrument contributions are USD 15.90 for
AAPL and USD -4.00 for MSFT. The USD 500 deposit contributes to neither profit
row. This is an additive decomposition under the chosen rule; interacting
signals may require another credit rule or counterfactual portfolios.

Retain symbolic attribution sensitivity
---------------------------------------

Because the report uses the same symbolic definitions, an allocation coefficient
can remain unknown or symbolic while the executed stream stays fixed:

>>> from sympy import Symbol, diff
>>> weight = Symbol("signal_weight", nonnegative=True)
>>> allocated = InstrumentPnL(
...     "allocated",
...     opening_quantity=0,
...     opening_price=100,
...     closing_price=107,
...     fill_quantity=7 * weight,
...     fill_notional=Rational("735.6") * weight,
...     income=Rational("3.5") * weight,
...     fees=weight,
... )
>>> allocated.net == Rational("15.9") * weight
True
>>> diff(allocated.net, weight)
159/10

This derivative changes the declared credit allocation, holding execution fixed.
Changing the trading signal could change orders and executions instead, requiring
another scenario. The symbolic identity does not establish a unique causal effect.

Derive a cash-utility view
--------------------------

A preference model supplies an increasing cash-wealth utility ``U(W)``. Define
its local absolute and relative Arrow–Pratt risk aversion from derivatives:

.. math::

   A(W)=-U''(W)/U'(W), \qquad R(W)=-W U''(W)/U'(W).

Start with an expression and its wealth symbol. ``CashUtility`` stores those
inputs and derives the view as follows:

.. literalinclude:: ../../../examples/portfolio/utility.py
   :pyobject: CashUtility
   :end-before:     def __post_init__
   :language: python

.. literalinclude:: ../../../examples/portfolio/utility.py
   :pyobject: CashUtility.expressions
   :dedent: 4
   :language: python

Its two family constructors define normalised CARA and CRRA utilities. The
coefficient helper converts a finite nonnegative coefficient to an exact SymPy
rational:

.. literalinclude:: ../../../examples/portfolio/utility.py
   :pyobject: CashUtility.cara
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/utility.py
   :pyobject: CashUtility.crra
   :dedent: 4
   :language: python

CARA has constant absolute risk aversion; its zero-coefficient limit is linear
utility. CRRA has constant relative risk aversion; its coefficient-one case is
logarithmic utility. Absolute aversion has units of inverse currency, while
relative aversion is dimensionless.

>>> from sympy import log
>>> from examples.portfolio.utility import CashUtility
>>> exponential = CashUtility.cara(0.001)
>>> power = CashUtility.crra(2)
>>> (
...     exponential.expressions["absolute_risk_aversion"],
...     power.expressions["relative_risk_aversion"],
... )
(1/1000, 2)
>>> wealth = Symbol("wealth", positive=True)
>>> custom = CashUtility("Log wealth", wealth, log(wealth))
>>> custom.expressions["relative_risk_aversion"]
1

The numerical plan takes wealth as its sole input and evaluates these derived
expressions using the same ``ScalarPlan`` introduced in chapter 3:

.. literalinclude:: ../../../examples/portfolio/utility.py
   :pyobject: CashUtility.plan
   :dedent: 4
   :language: python

``over`` checks positive wealth, finite outputs, and positive marginal utility.
Add the definitions to the report; it evaluates them on actual account NAV,
including deposits and withdrawals:

>>> utilities = NamedMap({"cara": exponential, "crra": power})
>>> report = path.report(
...     attributions, signals=NamedMap({"momentum": signals}), utilities=utilities
... )
>>> report.utilities["cara"].values["absolute_risk_aversion"].values
(0.001, 0.001, 0.001, 0.001)
>>> report.utilities["crra"].values["relative_risk_aversion"].values
(2.0, 2.0, 2.0, 2.0)

These coefficients are supplied preferences, not estimates from account returns.
Utility levels depend on the family's normalisation; positive affine changes
within a utility preserve both derivative ratios. Utility is a view of wealth,
not a performance index.

Project tables and timelines
----------------------------

The report is typed data. A simple instrument-level table is already a projection
of its named model outputs:

>>> instrument_rows = tuple(
...     (name, round(sum(item.profit["net"].values), 2))
...     for name, item in report.instruments.items()
... )
>>> instrument_rows
(('aapl', 90.9), ('msft', -24.0))
>>> profit_timeline = report.account["net"].cumulative_sum()
>>> tuple(round(value, 2) for value in profit_timeline.values)
(10.0, 41.4, 66.9)

The account table uses the same interval columns with ending timestamps;
instrument tables add quantities, weights, and source contributions:

>>> account_rows = tuple(
...     (
...         at.date().isoformat(),
...         round(report.account["net"].values[i], 2),
...         report.account["external_flows"].values[i],
...     )
...     for i, at in enumerate(report.account["net"].times)
... )
>>> for row in account_rows:
...     print(row)
('2026-01-06', 10.0, 0.0)
('2026-01-07', 31.4, 500.0)
('2026-01-08', 25.5, 0.0)

A chart can prepend the explicit opening zero to cumulative profit and plot it beside
NAV, capital flows, source P&L, signals, or utility observations. Rendering
introduces no additional financial calculation.

``to_dict`` projects these records for JSON. ``to_html(comparison, context=...)``
adds the named benchmark comparison built in chapter 7 and a data/methodology
caption. The shipped renderer produces account and instrument tables, signal
lineage, timeline charts, and utility curves; it runs offline.

With the reference indices from chapter 7, the completed view is now a final
projection rather than another calculation layer. Recreate the comparison here
when running this chapter independently:

>>> from examples.compute.series import Series
>>> from examples.portfolio.performance import PerformanceIndex
>>> from examples.portfolio.sources import CSVIndexSource
>>> benchmark = PerformanceIndex(
...     "Synthetic broad-market total-return index",
...     CSVIndexSource(data_dir / "benchmark.csv").load(),
... )
>>> cash_reference = PerformanceIndex(
...     "Zero-return USD cash", Series(times, (100,) * len(times))
... )
>>> comparison = PerformanceIndex(
...     "Account, net of booked fees and including income", path.unit_nav
... ).compare(
...     benchmark,
...     risk_free=cash_reference,
...     periods_per_year=252,
... )
>>> html = report.to_html(comparison, context="Synthetic observations; three daily returns")
>>> html.startswith("<!doctype html>")
True

Assemble the application after the layers
-----------------------------------------

We have now built the parts of the research run. ``ResearchScenario`` simply
declares their inputs: account and transaction provider, observation schedules,
decision/arrival/assessment horizons, reference indices, utilities, and policy
parameters. Its orchestration method performs the steps from these chapters:

.. literalinclude:: ../../../examples/portfolio/demo.py
   :pyobject: ResearchScenario
   :end-before:     def run
   :language: python

.. literalinclude:: ../../../examples/portfolio/demo.py
   :pyobject: ResearchScenario.run
   :dedent: 4
   :language: python

The class's ``example()`` constructor fills those fields with the same offline
fixtures and assumptions we have been using. This is a convenience for the
completed application:

.. raw:: html

   <details><summary>Inspect the completed scenario's fixture configuration</summary>

.. literalinclude:: ../../../examples/portfolio/demo.py
   :pyobject: ResearchScenario.example
   :dedent: 4
   :language: python

.. raw:: html

   </details>

>>> from examples.portfolio.demo import ResearchScenario
>>> result = ResearchScenario.example().run()
>>> finished = result.report()
>>> round(sum(finished.account["net"].values), 2)
66.9

Generate both views of the completed run:

.. code-block:: console

   $ python -m examples.portfolio.demo --html account.html --json account.json

`Open the generated interactive example <report.html>`_. It retains measurement
dates, three observed daily returns, the independent index comparison, and the
unavailable short-history CAGR/Sharpe values. The :doc:`implementation map <layers>`
locates the modules behind each layer and records the remaining scope boundaries.

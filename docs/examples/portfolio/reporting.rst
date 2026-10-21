5. Analyse performance and explain profit
=========================================

Start with the account's unit-NAV index, compare it with an independent
reference, then explain profit through signal-linked positions and utility.
The accounting and signal models are already defined; this chapter builds
their analytical and reporting views before assembling the complete scenario.

The setup below recreates the account and decision inputs from the preceding
chapters for an independent run.

.. raw:: html

   <details><summary>Recreate the account and decision inputs</summary>

>>> from datetime import timedelta
>>> from pathlib import Path
>>> import examples.portfolio as portfolio_examples
>>> from examples.portfolio.demo import Portfolio, Basket, EquityMembers
>>> from examples.portfolio.valuation import equity
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
>>> from examples.data import NamedMap
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

Define the observation and return statistics
--------------------------------------------

The account supplies four marks over three calendar days, giving three interval
returns. Compute total investment growth from its unit index:

>>> index = path.unit_nav
>>> len(index.simple_returns().values), (index.times[-1] - index.times[0]).days
(3, 3)
>>> round((index.values[-1] / index.values[0] - 1) * 100, 4)
2.4518

A result should retain that window alongside its statistics. Define a typed
record before deciding how to calculate each field:

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: Performance
   :language: python

``measure`` calculates simple interval returns, sample variance, drawdown, and
elapsed-time growth. Sharpe uses excess returns over an explicitly supplied
cash index. The example withholds annualized CAGR and Sharpe below one full
calendar year and returns no Sharpe for zero excess-return variance:

* Total return is ``final_level / opening_level - 1``.
* CAGR is ``growth ** (365.25 / elapsed_days) - 1`` when eligible.
* Annualized volatility is ``sqrt(sample_variance(returns) * periods_per_year)``.
* Annualized Sharpe is ``mean(excess) / sample_stddev(excess) * sqrt(periods_per_year)``.
* Maximum drawdown is the largest fall from a running index peak.

.. raw:: html

   <details><summary>Inspect the complete statistics calculation</summary>

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: measure
   :language: python

.. raw:: html

   </details>

The caller supplies a regular observation schedule and its annual frequency.
CAGR uses elapsed calendar time, ACT/365.25. Volatility scales sample return
variance by that frequency; a short sample's descriptive volatility is still
available, but does not establish a reliable risk estimate. Irregular
observations need an explicit resampling policy before this annual scaling.

Give each index a name and currency
-----------------------------------

An account, benchmark, and cash reference all have the same representation:

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: PerformanceIndex
   :end-before:     def __post_init__
   :language: python

The constructor requires positive levels and a name/currency. Its comparison
method checks exact axis and currency agreement, measures both paths, normalises
them to 100, and derives active returns and relative wealth:

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: PerformanceIndex.compare
   :dedent: 4
   :language: python

Load the independent synthetic benchmark from its ``at,level`` CSV. Define a
zero-return cash reference explicitly rather than silently choosing a Sharpe
baseline:

>>> from examples.data import Series
>>> from examples.portfolio.sources import CSVIndexSource
>>> from examples.portfolio.analytics import PerformanceIndex
>>> account_index = PerformanceIndex(
...     "Account, net of booked fees and including income", path.unit_nav
... )
>>> benchmark = PerformanceIndex(
...     "Synthetic broad-market total-return index",
...     CSVIndexSource(data_dir / "benchmark.csv").load(),
... )
>>> cash_reference = PerformanceIndex(
...     "Zero-return USD cash", Series(times, (100,) * len(times))
... )
>>> comparison = account_index.compare(
...     benchmark, risk_free=cash_reference, periods_per_year=252
... )
>>> (
...     round(comparison.portfolio.total_return * 100, 4),
...     round(comparison.benchmark.total_return * 100, 4),
... )
(2.4518, 3.0)
>>> round(comparison.return_difference * 100, 4), round(comparison.relative_return * 100, 4)
(-0.5482, -0.5322)
>>> comparison.portfolio.cagr, comparison.portfolio.sharpe_ratio
(None, None)

The first difference is in percentage points; relative return compares the two
wealth ratios. The account's USD 66.90 profit, its 2.4518% unitized return, and
its NAV growth answer related but different questions. The reference returned
3% over this same short synthetic window. The report must keep those facts
visible when presenting the toy portfolio.

Wrap valuation columns as portfolio paths
-----------------------------------------

For fixed holdings we can also reconcile each interval's return contribution.
Package the numerical columns already produced by chapter 2 into typed paths:

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: PositionPath
   :language: python

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: NamedPortfolioPath
   :end-before:     @classmethod
   :language: python

The ordinary ``ModelPlan`` evaluates the price observations into a ``ModelPath``.
The analytics layer projects that path's model fields and properties into these
records. It requires constant quantities and cash, and uses opening weights for
return contributions:

.. raw:: html

   <details><summary>Inspect the valuation-to-path adapter</summary>

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: NamedPortfolioPath.from_model
   :dedent: 4
   :language: python

.. raw:: html

   </details>

Use the same external histories and reporting times as the account, while
holding its opening quantities and cash fixed. ``EquityPrices[Time]`` pairs a
date or timestamp series with its instrument identity. ``NamedMarketData`` binds
that common axis to the portfolio's typed fields:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: NamedMarketData
   :language: python

>>> from examples.portfolio.models import EquityPrices
>>> from examples.portfolio.valuation import NamedMarketData
>>> from examples.portfolio.analytics import NamedPortfolioPath
>>> observations = NamedMarketData(times, market.map(
...     lambda history: EquityPrices(history.equity, history.sample(times, max_age=age))
... ))
>>> fixed = NamedPortfolioPath.from_model(opening.pricer().over(observations))
>>> fixed.nav.values
(2500.0, 2510.0, 2530.0, 2550.0)

For each interval, a position contributes its **opening** weight times its
interval price return. With zero-return cash, these contributions add to the
portfolio's return:

>>> observed = fixed.nav.simple_returns()
>>> summed = tuple(
...     sum(position.return_contribution.values[i] for position in fixed.positions.values())
...     for i in range(len(observed.values))
... )
>>> all(abs(a - b) < 1e-12 for a, b in zip(observed.values, summed))
True

This is return attribution for a fixed holding path. It is a different question
from attributing traded P&L to signals, which we will build next.

Optionally estimate fixed-weight portfolio risk
-----------------------------------------------

An asset covariance matrix has two holding-name axes. For annualized sample
covariance ``Sigma`` and reference weights ``w``, portfolio volatility is
``sqrt(w.T * Sigma * w)``. Euler contributions are
``w[i] * (Sigma * w)[i] / volatility`` and add to that volatility. The example
returns zero contributions when volatility is zero.

``analyse`` performs the time reduction, then evaluates risk at the supplied
reference weights. ``path.analytics`` supplies the latest observed weights:

.. raw:: html

   <details><summary>Inspect covariance and volatility contribution calculations</summary>

.. literalinclude:: ../../../examples/portfolio/analytics.py
   :pyobject: analyse
   :language: python

.. raw:: html

   </details>

>>> analytics = fixed.analytics(periods_per_year=252)
>>> round(analytics.performance.annualized_volatility * 100, 4)
3.6083
>>> round(analytics.latest_weights_risk.volatility * 100, 4)
3.4877
>>> round(sum(analytics.latest_weights_risk.contributions.values()) * 100, 4)
3.4877

Observed-path volatility uses drifting historical weights; reference-weight
risk holds the latest weights fixed against the sample covariance. These are
statistics of three synthetic daily returns, with covariance using ``n - 1``
degrees of freedom. Cash has zero return and risk in this example.

Return attribution for fixed shares and attribution of traded profit answer
different questions. Next connect the account's profit models to its signals.

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
outcome. The order's method constructs this record:

.. literalinclude:: ../../../examples/portfolio/signals.py
   :pyobject: OrderIntent.attribute
   :dedent: 4
   :language: python

For our one-signal policy, map the orders into attribution records. The projection
retains the decision's member bindings:

>>> attributions = orders.map(lambda order: order.attribute(NamedMap({"momentum": 1.0})))
>>> members = decision.portfolio.basket.members
>>> attributions.for_member(members.aapl).weights["momentum"]
1.0

A multi-signal decision could use ``NamedMap({"value": 0.4, "momentum": 0.6})``.
The reporting adapter accepts these records, including records from multiple
decisions. It derives its ledger index from each record's order ID, checks that
IDs are unique, and checks the named histories were available at the decision.
Linked fills must match the account, instrument, timing, side, and quantity.

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
>>> from examples.portfolio.models import Position, Quote
>>> from examples.portfolio.models import InstrumentPnL, TradingActivity
>>> sleeve = InstrumentPnL(
...     "momentum",
...     before=Position("opening", quantity=0, instrument=Quote("quote", price=100)),
...     after=Position("closing", instrument=Quote("quote", price=107)),
...     activity=TradingActivity(
...         "executed",
...         quantity=7,
...         notional=Rational("735.6"),
...         income=Rational("3.5"),
...         fees=1,
...     ),
... )
>>> sleeve.profit.net
159/10
>>> sleeve.after.quantity
7

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
prices, income, and fees in typed interval records. Those are precisely
``InstrumentPnL``'s inputs from chapter 3. ``Known`` bindings connect each column
to its actual model field, and the compiled view retains the full model schema.

Project the calculated profit fields into the labelled columns used by tables:

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: Profit.columns
   :dedent: 4
   :language: python

The labels belong to the report. Model evaluation and field selection use the
declared component and scalar references.

To add instrument and account columns, use an axis-checked reduction:

.. literalinclude:: ../../../examples/portfolio/models.py
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
...     before=sleeve.before,
...     after=Position("closing", instrument=sleeve.after.instrument),
...     activity=TradingActivity(
...         "allocated",
...         quantity=7 * weight,
...         notional=Rational("735.6") * weight,
...         income=Rational("3.5") * weight,
...         fees=weight,
...     ),
... )
>>> allocated.profit.net == Rational("15.9") * weight
True
>>> diff(allocated.profit.net, weight)
159/10

This derivative changes the declared credit allocation, holding execution fixed.
Changing the trading signal could change orders and executions instead, requiring
another scenario. The symbolic identity does not establish a unique causal effect.

Derive a cash-utility view
--------------------------

A preference model supplies an increasing cash-wealth utility ``U(W)``. Define
its local absolute and relative Arrow–Pratt risk aversion from derivatives:

.. math::

   \begin{aligned}
   A(W)&=-U''(W)/U'(W),\\
   R(W)&=-W U''(W)/U'(W).
   \end{aligned}

Start with an expression and its wealth symbol. ``CashUtility`` stores those
inputs and derives the view as follows:

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: CashUtility
   :end-before:     def __post_init__
   :language: python

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: CashUtility.expressions
   :dedent: 4
   :language: python

Its two family constructors define normalised CARA and CRRA utilities. The
coefficient helper converts a finite nonnegative coefficient to an exact SymPy
rational:

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: CashUtility.cara
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/reporting.py
   :pyobject: CashUtility.crra
   :dedent: 4
   :language: python

CARA has constant absolute risk aversion; its zero-coefficient limit is linear
utility. CRRA has constant relative risk aversion; its coefficient-one case is
logarithmic utility. Absolute aversion has units of inverse currency, while
relative aversion is dimensionless.

>>> from sympy import log
>>> from examples.portfolio.reporting import CashUtility
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
expressions using the same ``ScalarPlan`` introduced in chapter 2:

.. literalinclude:: ../../../examples/portfolio/reporting.py
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
adds the named benchmark comparison built in chapter 5 and a data/methodology
caption. The shipped renderer produces account and instrument tables, signal
lineage, timeline charts, and utility curves; it runs offline.

With the reference indices from chapter 5, the completed view is now a final
projection rather than another calculation layer. Recreate the comparison here
when running this chapter independently:

>>> from examples.data import Series
>>> from examples.portfolio.analytics import PerformanceIndex
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
unavailable short-history CAGR/Sharpe values. The :doc:`layer overview <index>`
locates the modules behind each layer and records the remaining scope boundaries.

The :doc:`next study <optimal_execution>` goes beyond scalar reuse: define an
execution cost functional, let SymPy derive continuous and discrete systems,
then reuse them for calibration, decision sensitivities, and liquidity reporting.

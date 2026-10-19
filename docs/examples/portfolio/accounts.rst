Accounts, signals, and execution research
=========================================

The :doc:`model views guide <modelling>` starts with the equation definitions
and explains how the same models support different knowns and unknowns.

This worked system combines the :doc:`named-member portfolio <named_members>`
with an external transaction log and time-aware market observations. It follows
one USD account from opening holdings through signals, intended orders, partial
executions, a deposit, and dividend income.

Install the examples and run the scenario:

.. code-block:: console

   $ pip install "modello[examples]"
   $ python -m examples.portfolio.demo
   $ python -m examples.portfolio.demo --json /tmp/account-report.json
   $ python -m examples.portfolio.demo --html /tmp/account-report.html

The :doc:`symbolic reporting walkthrough <reporting>` adds account and instrument
tables, signal P&L attribution, utility derivatives, and interactive timelines.

All observations are synthetic. The source files are independent:
``examples/portfolio/data/market.csv`` contains market trade prints, while
``transactions.csv`` contains broker/account postings, and ``benchmark.csv``
contains an illustrative broad-market total-return index. Change the adapters to
use another source; the computations consume typed observations and events.

Follow the layers
-----------------

The example has explicit boundaries:

.. mermaid::
   :caption: Sources, decisions, accounting, and execution analysis
   :alt: Published market observations feed signals and account valuation. The external ledger feeds account replay and execution analysis; intended orders do not create fills.

   flowchart TD
       market[Market source] --> prices[Published prices]
       prices --> grid[Sampled feature grid]
       grid --> signals[Momentum model]
       ledger[External transaction log] --> replay[Account replay with balance models]
       prices --> replay
       replay --> snapshot[Decision-time snapshot]
       signals --> targets[Target weights]
       snapshot --> targets
       targets --> orders[Rebalance model and order intentions]
       replay --> positions[Positions, cash, NAV]
       positions --> analytics[P&L models, returns, covariance]
       orders --> execution[Execution cost model and benchmarks]
       ledger -->|Reported fills| execution
       market -->|Independent prints| execution

``EquityProvider`` is an abstract ``Provider[PriceHistory]``. The transaction
adapter implements ``Provider[TransactionLog]``. ``SignalSource`` supplies named signal vectors; the example uses
trailing momentum. ``TargetWeights`` is the allocation boundary: the example
equally weights positive signals with 80% equity exposure and leaves the rest
in cash. An optimiser could consume forecasts, risk estimates, and constraints
and return the same type. Accounting and expression execution need no knowledge
of that policy's objective or solver.

An ``OrderIntent`` records the account, named position, equity identity, decision
time, arrival time, signed intended shares, and model execution price. Fills
come from the separate transaction log and join to intentions through
``order_id``. Manual fills without a model order still affect the account.

Bind instruments to providers
-----------------------------

The :doc:`basket definition <holdings>` binds each named holding to a configured
``EquityProvider``. For example, ``DummyEquityProvider(symbol="AAPL")`` identifies
and loads the supplied synthetic Apple prints. ``CSVEquityProvider`` accepts a
symbol, path, provider namespace, and quote currency for an external CSV.

The basket stores its provider map once. Its ``instruments`` map derives the
serializable quote identities, while ``positions[name]`` exposes the provider,
identity, quantity, value, and contextual weight together. Instrument metadata
is available while defining a symbolic model; loading is an explicit step:

.. code-block:: python

   market = opening.load_market()

The portfolio's ``load_market()`` method preserves holding names and shares
equivalent requests in that batch. Different provider types or request parameters remain distinct,
even with equal symbol text. A later call reloads the data. The
:doc:`provider guide <providers>` describes the generic return type, CSV schema,
and how another equity provider can supply the same accounting layer.

Observe time and availability
-----------------------------

``Series[T]`` represents a numeric vector on an ordered date or timestamp axis.
It supports pointwise projection, exact-axis combination, lag, trailing means,
returns, drawdown, and explicit backward sampling. Timestamp axes require an
offset and are normalized to UTC. Empty vectors represent empty event
projections and feature warmup periods.

``PriceHistory`` retains both the economic time of each print and the time it
became available. ``mark`` and ``sample`` select the latest published print and
enforce a caller-supplied age limit measured from its economic time. They never
select a later price to fill a missing earlier observation.

The signal example samples a deliberate feature grid: two prior closing
observations and the decision observation. Its lookback counts observations on
that grid, rather than arbitrary numbers of market updates. Lagged ratios then
produce momentum at the final timestamp of each window.

Transactions similarly carry execution time and reporting time. Account replay
uses postings known by the reporting horizon. A later report may restate past
holdings after a delayed posting arrives. Snapshots record ``known_at``;
``snapshot.rebalance`` requires a snapshot whose knowledge ends at the decision
time, preventing a restated historical snapshot from entering a past decision.

Declare and value an account
----------------------------

Use a typed member group to specify the opening quantities and cash:
the market source supplies the price variables.

.. code-block:: python

   # Portfolio is the declared basket-and-cash model from the holdings guide.
   opening = Portfolio(
       "opening",
       basket=Basket(
           "basket",
           members=EquityMembers(
               "equities",
               aapl=EquityPosition("apple", quantity=10),
               msft=EquityPosition("microsoft", quantity=10),
           ),
       ),
       cash=USDCash("cash", amount=1000),
   )
   account = Account("research", opened_at, opening)
   market = opening.load_market()
   path = account.replay(transaction_log, valuation_times, market=market,
                         max_price_age=timedelta(days=4))

The account's opening balances already include history through ``opened_at``.
Replay applies postings in ``(opened_at, mark]``, including events exactly at a
mark. Simultaneous events retain source order: a deposit can fund a subsequent
buy at that timestamp. Account names isolate postings from other accounts.

A fill changes shares by its signed quantity and cash by
``-quantity * price - fee``. Deposits and withdrawals change capital. Income
changes cash and investment return. Each mark produces named quantities,
prices, values, and weights, together with cash and total NAV.

>>> from examples.portfolio.demo import ResearchScenario
>>> result = ResearchScenario.example().run()
>>> path = result.account
>>> tuple(round(value, 2) for value in path.nav.values)
(2500.0, 2510.0, 3041.4, 3066.9)
>>> path.positions["aapl"].quantity.values
(10.0, 10.0, 17.0, 17.0)
>>> path.positions["msft"].quantity.values
(10.0, 10.0, 0.0, 0.0)

Reconcile P&L and remove external capital from returns
------------------------------------------------------

For an interval ending at mark :math:`P_1`, an equity's P&L is its initial
shares times the mark change, plus each signed fill's shares times
:math:`P_1 - P_{fill}`. Income is added and booked fees are deducted. Thus:

.. math::

   \Delta NAV = P\&L_{holding} + P\&L_{trading} + income - fees + external\ flows.

>>> tuple(round(value, 2) for value in path.pnl.total.values)
(10.0, 31.4, 25.5)
>>> path.pnl.external_flows.values
(0.0, 500.0, 0.0)
>>> tuple(round(value, 10) for value in path.reconciliation.values)
(0.0, 0.0, 0.0)

The performance index starts at 100. At each external cash flow, the engine
values the account immediately before the flow and issues or redeems units at
that unit price. Investment P&L changes unit NAV; external capital changes the
number of units. Internal valuation at flow times makes the result independent
of whether those times appear on the requested output grid.

This follows the cash-flow separation and geometric subperiod linking described
in the `GIPS calculation methodology
<https://www.gipsstandards.org/wp-content/uploads/2021/03/calculation_methodology_gs_2006.pdf>`_.
Dividend and interest income remain part of investment return.

>>> round(path.analytics(periods_per_year=252).performance.total_return * 100, 4)
2.4518

The account gained USD 566.90 in total value, consisting of USD 500 external
capital and USD 66.90 investment P&L. Dividing the NAV change by opening NAV
would include the deposit in performance. The unitized return instead links the
return before the deposit with the return afterwards.

.. _account-performance:

Compare performance on the same basis
--------------------------------------

The account covers **5–8 January 2026: three calendar days and three daily
returns**. It demonstrates account mechanics, rather than evidence of an
investment strategy. The command-line and JSON reports show this window
alongside an independent synthetic index:

>>> comparison = result.comparison
>>> round(comparison.portfolio.total_return * 100, 4)
2.4518
>>> round(comparison.benchmark.total_return * 100, 4)
3.0
>>> round(comparison.return_difference * 100, 4)
-0.5482
>>> comparison.portfolio.cagr, comparison.portfolio.sharpe_ratio
(None, None)

The account trails this illustrative index by **0.5482 percentage points**.
Both inputs are synthetic; the index is not a measured market return. Its
levels are supplied separately from the account and are never signal inputs.
The account includes booked fees and income; the index represents total return
before investor trading costs. For a real comparison, supply a suitable
total-return benchmark in the account's currency over the same period.

``PerformanceIndex`` names a typed series and its reporting currency.
``compare(account, benchmark, risk_free=cash, periods_per_year=252)`` requires
matching currencies and observation times. It returns both paths rebased to
100, period-by-period active returns, the difference in cumulative returns,
and relative wealth return ``account_growth / benchmark_growth - 1``. The last
two quantities have different denominators and are labelled separately.
``CSVIndexSource`` loads ``at,level`` observations for a reporting schedule;
choose any sampling or missing-data policy explicitly before comparison.

``measure`` is the shared calculation for account and benchmark statistics:

* **Total return** is ending index divided by starting index, minus one. Use
  ``path.unit_nav`` to remove external capital from account performance.
* **CAGR** is ``growth ** (365.25 / elapsed_calendar_days) - 1``. Its year basis
  is elapsed calendar time (ACT/365.25), independent of the sampling frequency.
* **Annualized Sharpe** is the mean interval excess return divided by its sample
  standard deviation, multiplied by ``sqrt(periods_per_year)``. Excess returns
  subtract the explicitly supplied risk-free index's interval returns. The
  demo labels its zero-return USD cash assumption. Omitting cash or having zero
  excess-return variance gives an unavailable Sharpe, rather than infinity.
* **Volatility** uses sample return variance, scaled by the supplied annual
  frequency. **Maximum drawdown** is the largest observed decline from a prior
  index peak; declines between observations are not measured.

For presentation, CAGR and annualized Sharpe are withheld until a full calendar
year is available. Short-period returns are kept unannualized, consistent with
the `GIPS guidance on periods below one year
<https://www.gipsstandards.org/standards/gips-standards-for-firms/gips-standards-handbook-for-firms/>`_.
Applying the same display threshold to Sharpe is this example's policy; one year
alone does not establish a reliable estimate. A zero observed drawdown across
three rising marks likewise says little about downside risk.

Longer histories exercise the same calculation. Here a quarterly total-return
index is measured against a quarterly cash index:

>>> from datetime import date
>>> from examples.portfolio.performance import measure
>>> from examples.compute.series import Series
>>> times = (date(2024, 1, 1), date(2024, 4, 1), date(2024, 7, 1), date(2024, 10, 1), date(2025, 1, 1))
>>> levels = Series(times, (100, 110, 104.5, 112.86, 115.1172))
>>> cash = Series(times, (100, 101, 102.01, 103.0301, 104.060401))
>>> performance = measure(levels, periods_per_year=4, risk_free=cash)
>>> round(performance.cagr * 100, 4)
15.084
>>> round(performance.sharpe_ratio, 4)
0.8146

Choose a regular valuation schedule and its periods per year for risk and
Sharpe. An event grid's intervals need not have equal duration; the example
does not infer a trading calendar from transaction timestamps. Square-root
annualization assumes uncorrelated interval excess returns, as discussed in
`William Sharpe's original explanation
<https://web.stanford.edu/~wfsharpe/art/sr/sr.htm>`_. These estimates do not account
for serial dependence, selection bias, or the uncertainty from a small sample.

``path.analytics(...).performance`` uses the same ``measure`` function. Its
separate latest-weight risk estimate uses equity **price-return** covariance;
it describes current exposures, rather than historical strategy performance.
Immutable account paths reuse their position, unit-NAV, and P&L projections
when several reporting layers access them.

Read executions as series and benchmark model slippage
------------------------------------------------------

The transaction log exposes ``executions(member)`` as signed quantity, gross
volume, notional, fees, and execution-VWAP series. Fills at the same timestamp
are aggregated for these vectors while their individual IDs remain in the log.

>>> executions = path.ledger.executions("aapl")
>>> executions.quantity.values, executions.vwap.values
((4.0, 3.0), (105.0, 105.2))

Execution analysis uses an explicit assessment time. For each order it reports
filled and unfilled shares, execution VWAP, an independent market-print VWAP over
``[arrival, assessment]``, and model and arrival slippage. Market VWAP uses
``sum(price * volume) / sum(volume)`` on the external prints; it is independent
of this account's fill prices. Missing prints in that window give an absent
VWAP benchmark.

Model slippage includes fees on executed shares. Its basis-point denominator is
executed quantity times model price. Arrival and market-VWAP basis points measure
price differences before fees. A positive cost is adverse for either side;
buying cheaper or selling dearer gives a negative price cost.

Full implementation shortfall also includes the unfilled shares' opportunity
cost. Write :math:`s=+1` for a buy and :math:`s=-1` for a sell, intended absolute
quantity :math:`Q`, filled quantity :math:`q`, decision price :math:`P_d`, arrival
price :math:`P_a`, terminal mark :math:`P_t`, and gross executed notional :math:`N`:

.. math::

   delay &= q s (P_a - P_d), \\
   trading &= s (N - q P_a), \\
   opportunity &= (Q-q)s(P_t-P_d), \\
   IS &= delay + trading + fees + opportunity.

The decomposition compares actual execution with a paper transaction at the
decision price, following the categories in `CFA Institute's trade strategy and
execution reading
<https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/trade-strategy-execution>`_.
The opportunity-cost horizon is the assessment time supplied by the caller.

>>> apple = result.execution["rebalance-1:aapl"]
>>> apple.filled_quantity, round(apple.remaining_quantity, 4)
(7.0, 2.3846)
>>> round(apple.model_slippage_cost, 4), round(apple.implementation_shortfall, 4)
(8.6, 10.5077)
>>> round(result.execution["rebalance-1:msft"].implementation_shortfall, 4)
4.0

The AAPL order was only partly filled, so its full shortfall also contains the
unfilled portion's opportunity cost. The MSFT sell was filled completely. Neither
execution metric is substituted for the account's independently reconciled P&L.

Keep the computation layer small
--------------------------------

The :doc:`research layer guide <layers>` describes how the numeric data,
symbolic executor, domain models, and presentation fit together. The generic
helpers live in ``examples/compute/``.

``ScalarPlan`` compiles named Modello expressions once and evaluates either one
point or aligned series. Both fixed-holding examples and account replay use it.
The examples share one set of path and risk statistics. ``Series`` implements
time operations; event replay, source adaptation, feature construction,
allocation policy, and benchmark definitions live in their own modules.

The scenario models funded, long-only, fractional USD holdings with immediate
trade cash movements. Its market prints are unadjusted; dividends are separate
income postings. Settlement balances, FX, corporate-action position changes,
tax lots, and market calendars would need explicit domain layers. Source
adapters must normalize corrections and provide unique, increasing market-print
and publication axes; transaction timestamps may repeat.

The numerical paths use floats. This makes the experiment useful for analytics
and numerical reconciliation; a production cash ledger would also need a
specified currency rounding policy. Allocation policies and broker submission
belong at the application boundary.

Other useful composition boundaries
-----------------------------------

The current symbolic composition is account portfolio → equity basket → named
holdings → quotes, with cash alongside the basket. The account owns the ledger
and valuation schedule; an instrument owns quote identity, while quantities
belong to holdings. This keeps shared market data independent of account state.

Further natural groupings are strategy sleeves or asset-class baskets, cash
balances by currency, and executions grouped by order. A multi-basket account
would need explicit capital bases and qualified holding names so the same
instrument can occur in several sleeves. Currency baskets also need FX
valuation before their totals can be added. These are extension points rather
than additional machinery in the current funded, USD-only example.

Signal attribution already has a separate source → instrument view. This is
useful because two signals can contribute to the same holding; forcing signals
into the ownership tree would lose that relationship. Similarly, market prints
and transactions retain their own observation/publication axes and are sampled
onto an account's timeline at the calculation boundary.

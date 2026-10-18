Accounts, signals, and execution research
=========================================

This worked system combines the :doc:`named-member portfolio <named_members>`
with an external transaction log and time-aware market observations. It follows
one USD account from opening holdings through signals, intended orders, partial
executions, a deposit, and dividend income.

Run the scenario from a checkout:

.. code-block:: console

   $ uv run --locked python -m examples.account_demo
   $ uv run --locked python -m examples.account_demo --json /tmp/account-report.json

All observations are synthetic. The two source files are independent:
``examples/data/portfolio/market.csv`` contains market trade prints, while
``transactions.csv`` contains broker/account postings. Change the adapters to
use another source; the computations consume typed observations and events.

Follow the layers
-----------------

The example has explicit boundaries:

.. code-block:: text

   market source -> published prices -> sampled feature grid -> signals
                           |                                      |
                           v                                      v
   transaction source -> account replay -> snapshot -> target weights -> order intentions
                           |                                              |
                           v                                              v
                 positions, cash, NAV             reported fills + independent market prints
                           |                                              |
                           v                                              v
                P&L, returns, covariance                       execution benchmarks

``MarketSource`` and ``TransactionSource`` are small protocols implemented by
CSV adapters. ``SignalSource`` supplies named signal vectors; the example uses
trailing momentum. ``TargetWeights`` is the allocation boundary: the example
equally weights positive signals with 80% equity exposure and leaves the rest
in cash. An optimiser could consume forecasts, risk estimates, and constraints
and return the same type. Accounting and expression execution need no knowledge
of that policy's objective or solver.

An ``OrderIntent`` records the account, named position, equity type, decision
time, arrival time, signed intended shares, and model execution price. Fills
come from the separate transaction log and join to intentions through
``order_id``. Manual fills without a model order still affect the account.

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
``rebalance_orders`` requires a snapshot whose knowledge ends at the decision
time, preventing a restated historical snapshot from entering a past decision.

Declare and value an account
----------------------------

Use a typed member group to specify the opening quantities and cash:
the market source supplies the price variables.

.. code-block:: python

   Portfolio = portfolio_model(EquityMembers)
   opening = Portfolio(
       "opening",
       equities=EquityMembers(
           "equities",
           aapl=ApplePosition("apple", quantity=10),
           msft=MicrosoftPosition("microsoft", quantity=10),
       ),
       cash=USDCash("cash", amount=1000),
   )
   account = Account("research", opened_at, opening)
   engine = AccountEngine(account, market, max_price_age=timedelta(days=4))
   path = engine.evaluate(transaction_log, valuation_times)

The account's opening balances already include history through ``opened_at``.
Replay applies postings in ``(opened_at, mark]``, including events exactly at a
mark. Simultaneous events retain source order: a deposit can fund a subsequent
buy at that timestamp. Account names isolate postings from other accounts.

A fill changes shares by its signed quantity and cash by
``-quantity * price - fee``. Deposits and withdrawals change capital. Income
changes cash and investment return. Each mark produces named quantities,
prices, values, and weights, together with cash and total NAV.

>>> from examples.account_demo import example_run
>>> result = example_run()
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

>>> round(path.analytics(periods_per_year=252).total_return * 100, 4)
2.4518

The account gained USD 566.90 in total value, consisting of USD 500 external
capital and USD 66.90 investment P&L. Dividing the NAV change by opening NAV
would include the deposit in performance. The unitized return instead links the
return before the deposit with the return afterwards.

Shared analytics use this index for return, volatility, and drawdown. The
separate latest-weight risk estimate uses equity **price-return** covariance.
Choose a regular valuation schedule and its periods per year when annualizing;
an event grid's intervals need not have equal duration. The example does not
infer a trading calendar from transaction timestamps.

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

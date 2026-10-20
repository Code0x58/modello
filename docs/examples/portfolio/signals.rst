6. Build signals, orders, and execution benchmarks
==================================================

An account path describes what happened. A trading decision needs the account
and observations available at its decision time, a rule producing targets,
and a model relating those targets to order quantities. We will build those
pieces before examining the independently recorded fills.

Continue with ``account``, ``market``, ``ledger``, ``times``, and ``age`` from
chapter 5. To run this page independently, expand the same setup:

.. raw:: html

   <details><summary>Recreate the account inputs from chapter 5</summary>

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

.. raw:: html

   </details>

Construct a decision-time view
------------------------------

January 7's decision occurs before that day's fills. Replay only through that
horizon, rather than using a later restated snapshot:

>>> decision_at = timestamp("2026-01-07T09:00:00Z")
>>> arrival_at = timestamp("2026-01-07T09:05:00Z")
>>> decision = account.replay(
...     ledger, (times[0], decision_at), market=market, max_price_age=age
... ).snapshots[-1]
>>> decision.at == decision.known_at == decision_at
True
>>> dict(decision.quantities), decision.nav
({'aapl': 10.0, 'msft': 10.0}, 2520.0)

Choose a feature grid with two prior closes and the decision observation.
Sampling uses the publication-time rule built in chapter 4:

>>> feature_times = (*times[:2], decision_at)
>>> prices = market.map(lambda history: history.sample(feature_times, max_age=age))
>>> prices["aapl"].values, prices["msft"].values
((100.0, 102.0, 104.0), (50.0, 49.0, 48.0))

Define the signal, then apply it to windows
-------------------------------------------

Momentum is the proportional return of the ``PriceChange`` model from chapter 5.
Bind its opening and closing quotes and expose that return as the signal:

.. literalinclude:: ../../../examples/portfolio/signals.py
   :pyobject: Momentum
   :language: python

A source of signals must also choose trailing inputs and their output times.
``MomentumSignals`` applies the same scalar model to lagged and current columns:

.. literalinclude:: ../../../examples/portfolio/signals.py
   :pyobject: MomentumSignals
   :language: python

>>> from examples.portfolio.signals import MomentumSignals
>>> signals = MomentumSignals(prices, lookback=2).load()
>>> {
...     name: tuple(round(value, 4) for value in series.values)
...     for name, series in signals.items()
... }
{'aapl': (0.04,), 'msft': (-0.04,)}
>>> signals["aapl"].times == (decision_at,)
True

The lookback counts observations, not elapsed days. Warmup periods have no
signal output. Each value is labelled with the time at which its inputs are
usable by a decision.

Turn observations into an allocation
------------------------------------

``TargetWeights`` is a timestamp and a named map of nonnegative weights summing
to at most one. Cash is the residual. Its illustrative policy divides an 80%
equity allocation equally among positive signals:

.. literalinclude:: ../../../examples/portfolio/signals.py
   :pyobject: TargetWeights.from_signals
   :dedent: 4
   :language: python

>>> from examples.portfolio.signals import TargetWeights
>>> targets = TargetWeights.from_signals(signals, decision_at, exposure=0.8)
>>> dict(targets.weights)
{'aapl': 0.8, 'msft': 0.0}

This selection rule is a policy. A later optimiser can return the same typed
``TargetWeights`` record; the accounting layer needs neither its objective nor
its solver.

Solve the order relationship
----------------------------

An allocation expresses a share of capital as a monetary value:

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: Allocation
   :language: python

Reuse the marked ``EquityPosition``, ``AccountValue``, and signed ``Trade`` models
from earlier chapters. The rebalance connects opening and target positions,
shares their quote with the order, and equates the target position's value to
the allocation's value:

.. literalinclude:: ../../../examples/portfolio/signals.py
   :pyobject: Rebalance
   :language: python

Before processing the account, check a small exact case in both directions:

>>> from sympy import Rational
>>> from examples.portfolio.models import (
...     AccountValue,
...     Allocation,
...     EquityPosition,
...     Quote,
...     Trade,
... )
>>> from examples.portfolio.signals import Rebalance
>>> held = EquityPosition("held", quantity=3, instrument=Quote("quote", price=100))
>>> proposed = Rebalance(
...     "target",
...     before=held,
...     account=AccountValue("account", nav=1000),
...     allocation=Allocation("allocation", weight=Rational(1, 2)),
... )
>>> proposed.after.quantity, proposed.order.quantity
(5, 2)
>>> observed = Rebalance(
...     "observed",
...     before=held,
...     account=AccountValue("account", nav=1000),
...     order=Trade("order", quantity=2),
... )
>>> observed.allocation.weight
1/2

There is no separate ``weight * nav / price`` instruction. SymPy derives the
required shares from the position valuation and allocation equations. An order
observation instead determines the weight. The root fields such as ``nav`` and
``target_weight`` alias the components for concise column bindings.

The adapter applies the compiled forward view to each named holding and
constructs an ``OrderIntent``. That record carries account, member and instrument
identity, decision and arrival times, signed shares, and model price:

.. literalinclude:: ../../../examples/portfolio/signals.py
   :pyobject: TargetWeights.orders
   :dedent: 4
   :language: python

>>> orders = decision.rebalance(targets, batch="rebalance-1", arrival_at=arrival_at)
>>> {name: round(order.quantity, 4) for name, order in orders.items()}
{'rebalance-1:aapl': 9.3846, 'rebalance-1:msft': -10.0}

The snapshot method delegates to these target equations. It requires a snapshot
whose knowledge ends at the decision time. These records express intentions;
the external transaction log continues to determine actual positions.

Define monetary execution costs
-------------------------------

We need a separate assessment window, and prices from the independent market
history at decision, arrival, and assessment. Monetary shortfall decomposes as:

.. literalinclude:: ../../../examples/portfolio/execution.py
   :pyobject: Shortfall
   :language: python

Delay and opportunity cost are applications of the same price-movement model;
trading cost and model slippage are applications of the same fill-cost model.
Define these two reusable pieces:

.. literalinclude:: ../../../examples/portfolio/execution.py
   :pyobject: PriceCost
   :language: python

.. literalinclude:: ../../../examples/portfolio/execution.py
   :pyobject: FillCost
   :language: python

The execution model connects their quantities, sides, and reference quotes to
the shortfall account. It counts fees once in shortfall; the model-slippage view
also includes those fees, while the trading-cost component excludes them:

.. literalinclude:: ../../../examples/portfolio/execution.py
   :pyobject: ExecutionCosts
   :language: python

Check Apple's filled shares directly. The ledger records buys of four shares
at USD 105 and three at USD 105.20, with USD 1 total fees. Relative to the model
price of USD 104:

>>> from examples.portfolio.execution import ExecutionCosts
>>> cost = ExecutionCosts(
...     "apple",
...     quantity=7,
...     remaining=0,
...     side=1,
...     notional=Rational("735.6"),
...     model_price=104,
...     decision_price=104,
...     arrival_price=Rational("104.5"),
...     terminal_price=Rational("104.8"),
...     fees=1,
... )
>>> cost.model_slippage_cost, cost.delay_cost, cost.trading_cost
(43/5, 7/2, 41/10)

Model slippage is USD 8.60, including fees. Positive cost denotes worse execution
for the order's side. Unfilled shares add opportunity cost when the intended
order exceeds the executed quantity.

For an independent traded benchmark, calculate VWAP from market prints in the
arrival-to-assessment window, using only prints published by assessment:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: PriceHistory.vwap
   :dedent: 4
   :language: python

Assess the independently recorded fills
---------------------------------------

``ExecutionBenchmark.assess`` joins known fills by order ID, checks account,
member, side, timing, and overfills, then sums filled quantities, notionals,
and fees. It supplies those inputs to ``ExecutionCosts``. ``Slippage`` expresses
signed price differences in basis points; ``CostRate`` expresses monetary cost
as basis points of filled model notional. A zero-fill order has no execution
price or filled-notional rate.

>>> from examples.portfolio.execution import ExecutionBenchmark
>>> assessment_at = timestamp("2026-01-07T09:15:00Z")
>>> execution = ExecutionBenchmark.assess(
...     orders, ledger, market, at=assessment_at, max_price_age=age
... )
>>> apple = execution["rebalance-1:aapl"]
>>> apple.filled_quantity, round(apple.remaining_quantity, 4)
(7.0, 2.3846)
>>> round(apple.model_slippage_cost, 4), round(apple.implementation_shortfall, 4)
(8.6, 10.5077)
>>> round(execution["rebalance-1:msft"].implementation_shortfall, 4)
4.0

A later change to targets may produce different intentions. It does not invent
new fills or change this external ledger. That separation will also matter when
attributing P&L to decisions.

The :doc:`next chapter <performance>` evaluates the resulting account against
an index and makes the short sample window visible before we build reports.

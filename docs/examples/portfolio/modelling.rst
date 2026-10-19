Models, cases, and reporting views
==================================

The portfolio experiment starts with equations. A case chooses which quantities
are known; solving or compiling that case supplies the desired view. Reporting
then applies the same definitions to observed intervals and named attribution
sleeves. Data providers and the event log supply observations at explicit times.

The financial definitions are ordinary ``Modello`` subclasses. The small
``SymbolicModel`` adapter in ``examples/symbolic.py`` adds expression compilation
for this experiment; the core model constructor and joint solver are unchanged.

Declare composition and relationships
--------------------------------------

The :doc:`holdings guide <holdings>` declares a basket and portfolio as classes:

.. code-block:: python

   class Holdings(EquityHoldings):
       aapl = equity(DummyEquityProvider(symbol="AAPL"))
       msft = equity(DummyEquityProvider(symbol="MSFT"))

   class Basket(EquityBasket[Holdings]):
       members = component(Holdings)
       market_value = EquityBasket.total(members)

   class Portfolio(NamedPortfolio[Holdings]):
       basket = component(Basket)
       cash = component(USDCash)
       nav = basket.capital
       nav = basket.market_value + cash.amount

Each holding and its provider are declared together. The basket projects
those requests under the component names, keeping acquisition and the symbolic
domain aligned. The reduction builds one symbolic value expression;
the basket's inherited weight equation uses the portfolio's capital base.
``EquityBasket.define(...)`` and ``NamedPortfolio.define(...)`` build equivalent
classes when the composition is selected from runtime configuration.

Change knowns, not the equation
--------------------------------

``Rebalance`` relates account NAV, price, opening quantity, target weight, and
signed order quantity. A target and an observed order are two cases of that
same model:

>>> from sympy import Rational, Symbol
>>> from examples.portfolio.signals import Rebalance
>>> target = Rebalance("target", nav=1000, price=100, opening_quantity=3, target_weight=Rational(1, 2))
>>> target.target_quantity, target.order_quantity
(5, 2)
>>> fitted = Rebalance("observed", nav=1000, price=100, opening_quantity=3, order_quantity=2)
>>> fitted.target_weight
1/2

The second construction fits the implied allocation to an exact observation.
It uses the existing joint solver. Statistical estimation from noisy samples
is a separate layer; these examples impose exact symbolic constraints.

A case is a new model instance with explicitly chosen knowns. This leaves the
other case's inputs, outputs, and independent instance symbols intact.

Compile a forward or inverse view
----------------------------------

A partially bound case can retain symbolic observations as runtime inputs.
``compile()`` selects those inputs and the outputs needed by a consumer:

>>> observed = Symbol("observed_order", real=True)
>>> inverse = Rebalance("inverse", nav=1000, price=100, opening_quantity=3, order_quantity=observed)
>>> view = inverse.compile(inputs=("order_quantity",), outputs=("target_weight",))
>>> from examples.compute.named_map import NamedMap
>>> dict(view.evaluate(NamedMap({"order_quantity": 2})))
{'target_weight': 0.5}

The output field becomes an observation by supplying a symbol for it when
constructing the case. Modello resolves the other fields in terms of that
symbol. The compiled view can then evaluate many such observations:

>>> from datetime import date
>>> from examples.compute.series import Series
>>> times = (date(2026, 1, 5), date(2026, 1, 6))
>>> orders = Series(times, (2, 4))
>>> tuple(round(value, 4) for value in view.over(times, NamedMap({"order_quantity": orders}))["target_weight"].values)
(0.5, 0.7)

Each scalar model also declares its usual forward ``inputs`` and ``outputs``.
``Rebalance.plan()`` caches that compiled view, and ``Rebalance.over(...)``
applies it on a supplied observation axis. A different case can compile another
view while keeping the same equation definition.

Share definitions across layers
--------------------------------

The application uses these model relationships throughout:

.. list-table::
   :header-rows: 1

   * - Model
     - Relationship and views
   * - ``FillPosting``
     - Signed fills change quantity and cash; an observed cash change can determine execution price.
   * - ``CapitalFlow``
     - Deposits issue units at the pre-flow unit price; observed units can determine the capital flow.
   * - ``Momentum`` and ``Rebalance``
     - Sampled prices produce a signal; targets and observed order quantities select different allocation views.
   * - ``ExecutionCosts`` and ``Shortfall``
     - Execution terms produce shortfall; a cost observation can determine an unknown fee or notional.
   * - ``InstrumentPnL``, ``Profit``, and ``AccountPnL``
     - The same interval-profit equations value actual positions, signal sleeves, and account reconciliation.

The shared P&L definitions live in ``examples/portfolio/pnl.py``. Their compiled
views evaluate actual account data and reporting sleeves; there is one
arithmetic definition for each relationship. Cash utility definitions keep
their expressions and symbolic derivative ratios available through
``CashUtility.cara(...)``, ``CashUtility.crra(...)``, or a custom expression.

Declare a scenario, then evaluate its views
-------------------------------------------

``ResearchScenario`` declares the opening account, transaction provider,
valuation and feature schedules, decision/arrival/assessment times, benchmark,
cash reference, utility definitions, and policy parameters. The worked offline
case is available as a class-owned example:

>>> from examples.portfolio.demo import ResearchScenario
>>> scenario = ResearchScenario.example()
>>> result = scenario.run()
>>> tuple(round(value, 2) for value in result.account.nav.values)
(2500.0, 2510.0, 3041.4, 3066.9)
>>> reporting = result.report()
>>> round(sum(reporting.account["net"].values), 2)
66.9

The operations belong to the objects whose views they produce:

.. code-block:: python

   market = opening.load_market()
   path = account.replay(ledger, valuation_times, market=market, max_price_age=age)
   orders = decision_snapshot.rebalance(targets, batch="rebalance", arrival_at=arrival)
   reporting = path.report(attributions, signals=signal_histories, utilities=utilities)
   html = reporting.to_html(comparison, context="Research assumptions and data source")

Chronology remains an explicit adapter concern. Providers acquire observations;
replay orders postings, honours publication times, and supplies successive
known states to the balance models. Sampling and aggregation build interval
inputs. Symbolic equations define the arithmetic within those states and
intervals; they do not choose the information horizon or invent missing fills.

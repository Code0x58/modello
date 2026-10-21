2. Evaluate models on external data
===================================

A scalar position answers a question at one point. To value fixed holdings on
several dates, we need an observation axis and a numerical evaluator. These are
separate requirements: the axis describes data, while the evaluator consumes
the equations already resolved by Modello.

Put observations on an explicit axis
------------------------------------

``Series`` pairs finite numeric values with strictly increasing dates or
UTC-normalised timestamps. Here are five synthetic daily observations:

>>> from datetime import date
>>> from examples.data import Series
>>> times = tuple(date(2026, 1, day) for day in range(5, 10))
>>> prices = Series(times, (100, 102, 98, 104, 103))
>>> prices.values
(100.0, 102.0, 98.0, 104.0, 103.0)

A lag labels the earlier values with the later observation dates. That makes
an interval's opening value align with its closing value:

.. literalinclude:: ../../../examples/data.py
   :pyobject: Series.lag
   :dedent: 4
   :language: python

>>> prices.lag().times == times[1:]
True
>>> prices.lag().values
(100.0, 102.0, 98.0, 104.0)
>>> tuple(round(value, 4) for value in prices.simple_returns().values)
(0.02, -0.0392, 0.0612, -0.0096)

``differences`` and ``simple_returns`` omit the warmup observation. Periods
count observations, including on an irregular axis; they do not insert missing
timestamps. ``combine`` requires exact axis agreement. ``cumulative_sum``
accumulates from zero, so adding an opening balance remains a visible choice:

>>> change = prices.differences()
>>> change.cumulative_sum().map(lambda value: value + prices.values[0]).values
(102.0, 98.0, 104.0, 103.0)

Compile a model-shaped view
---------------------------

The position already declares its quote, quantity, and valuation equation.
A compiled view should retain that structure. First define a known value as
an association with an actual model field:

.. code-block:: python

   from dataclasses import dataclass
   from typing import Generic, TypeVar
   from modello import SymbolicScalar

   Value = TypeVar("Value", covariant=True)

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: Known
   :language: python

``ModelPlan[Model]`` reads the resolved scalar fields and component paths from
an existing case. Independent symbolic fields become inputs; every declared
field remains in the result. Bind the fixed quantity as usual, leaving the
price unknown:

>>> from examples.portfolio.models import EquityPosition, Quote
>>> from examples.symbolic import Known, ModelPlan
>>> position = EquityPosition("fixed", quantity=10)
>>> view = ModelPlan(position)
>>> valued = view.evaluate(Known(position.instrument.price, 100))
>>> type(valued) is EquityPosition, type(valued.instrument) is Quote
(True, True)
>>> float(valued.instrument.price), float(valued.market_value)
(100.0, 1000.0)
>>> bool(position.instrument.price.free_symbols)
True

The result is another ``EquityPosition``, with its ordinary ``Quote`` child.
Numeric scalar fields remain SymPy numbers, as in any solved Modello case.
The calculation uses floating-point arithmetic; the source case retains its
exact expressions. A field reference identifies an input through
its model scope, so a similarly named field from another case cannot silently
supply its value.

Evaluate the same structure on an observation axis
--------------------------------------------------

Pair that field with the price series we built above. The vector operation
runs the compiled calculation once on aligned columns:

>>> path = view.over(times, Known(position.instrument.price, prices))
>>> path.column(position.market_value).values
(1000.0, 1020.0, 980.0, 1040.0, 1030.0)
>>> observed = path.at(2)
>>> float(observed.quantity), float(observed.instrument.price), float(observed.market_value)
(10.0, 98.0, 980.0)

``ModelPath`` keeps the model schema and observation axis. ``column`` selects a
scalar through a model reference; ``at`` materializes a typed numeric case when
needed. No joint solve is performed per observation. Properties derived from
model fields can also be selected as expressions.

Bind typed observation records
------------------------------

Repeated calculations can receive a typed data record instead of spelling out
the same field bindings at every call. Define how an observation supplies its
knowns to the model case selected by the evaluator:

>>> from dataclasses import dataclass
>>> @dataclass(frozen=True)
... class PriceObservation:
...     price: float
...
...     def knowns(self, model: EquityPosition) -> tuple[Known[float], ...]:
...         return (Known(model.instrument.price, self.price),)
>>> valued = view.evaluate(PriceObservation(100))
>>> float(valued.market_value)
1000.0

This follows the generic ``ModelData`` protocol. The observation receives the
compiled case's own model fields, so the same record can bind to another
compatible case without retaining symbols from the first:

.. code-block:: python

   from typing import Protocol, TypeVar, runtime_checkable
   from modello import Modello

   InputModel = TypeVar("InputModel", bound=Modello, contravariant=True)

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ModelData
   :language: python

A record containing aligned series can carry its observation axis as well:

>>> @dataclass(frozen=True)
... class PriceSamples:
...     times: tuple[date, ...]
...     prices: Series[date]
...
...     def knowns(self, model: EquityPosition) -> tuple[Known[Series[date]], ...]:
...         return (Known(model.instrument.price, self.prices),)
>>> sampled = view.over(PriceSamples(times, prices))
>>> sampled.column(position.market_value).values
(1000.0, 1020.0, 980.0, 1040.0, 1030.0)

The ``ModelSeries`` protocol adds that axis to ``ModelData``:

.. code-block:: python

   from examples.data import Time

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ModelSeries
   :language: python

Both forms use the same evaluator and return the same typed model or path.
Field ownership, missing/duplicate inputs, axis alignment, and numeric domains
are still checked centrally. ``evaluate(record, Known(...))`` can combine a
record with other inputs; ``over(times, record, Known(...))`` does the same for
series. Data acquisition remains explicit and outside these protocols.

Expose the declared fields for inspection
-----------------------------------------

A view can describe each field's real component path, its source expression,
and its role as an input, a known constant, or a derived value:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ModelField
   :language: python

>>> fields = tuple(view.walk(valued))
>>> tuple((".".join(field.path), field.value) for field in fields if field.role == "input")
(('instrument.price', 100.0),)
>>> derived = next(field for field in fields if field.path == ("market_value",))
>>> derived.expression == position.market_value, derived.value
(True, 1000.0)

A table or debugger can traverse these records while retaining the mathematical
expression that produced each value. The evaluated object itself keeps its
methods, aliases, concrete component types, and declaration metadata.

Select another solving direction
--------------------------------

Compilation consumes the equations resolved by Modello. To ask an inverse
question, change the case's knowns first, then compile that case. Build a small
view model from the position we already defined:

>>> from modello import Modello, component
>>> class PositionValue(Modello):
...     position = component(EquityPosition)
...     quantity = position.quantity
...     price = position.instrument.price
...     value = position.market_value
>>> inverse = PositionValue("quantity", price=100, value=1500)
>>> inverse.quantity
15

For a repeated numerical inverse, leave the observed value symbolic. The
resolved quantity becomes an expression of that observation, and the same
``ModelPlan`` finds the remaining input:

>>> from sympy import Symbol
>>> observed_value = Symbol("observed_value", positive=True)
>>> fitted = PositionValue("observed", price=100, value=observed_value)
>>> inverse_view = ModelPlan(fitted)
>>> result = inverse_view.evaluate(Known(fitted.value, 1500))
>>> float(result.quantity), float(result.position.quantity)
(15.0, 15.0)

No second valuation formula was introduced. Exact calibration to a supplied
observation is different from statistical estimation on noisy samples, which
needs its own model and estimation layer.

Build the adapter beneath these views
-------------------------------------

Here is how the view obtains its schema and selects independent symbolic fields:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ModelPlan.__init__
   :dedent: 4
   :language: python

The example layer isolates access to Modello's current component metadata.
``ScalarPlan`` compiles the resolved expressions with SymPy's ``lambdify``;
``ModelPlan`` restores the declared model shape after evaluation:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ModelPlan.evaluate
   :dedent: 4
   :language: python

.. raw:: html

   <details><summary>Inspect field traversal, copying, and the scalar compiler</summary>

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: _get_field
   :language: python

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: _copy_model
   :language: python

The copy operation reuses model classes and symbol scopes, replaces scalar
values, and recomputes cached properties. It invokes no symbolic solver.

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ScalarPlan.__init__
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: ScalarPlan.over
   :dedent: 4
   :language: python

.. raw:: html

   </details>

Reuse a model's compiled view
-----------------------------

Later models will use a small ``SymbolicModel`` subclass of ``Modello``. It
provides ``case.compile()`` for a bound or inverse case and ``Model.plan()``
for a cached, unbound case. Both return the same ``ModelPlan`` we just built:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: SymbolicModel
   :language: python

Independent fields are inferred from the resolved case. Every declared field
and component remains in the result; a numerical input can be on either side
of an equation. Additional input selection, when needed, uses actual field
references. A report can project chosen fields into labelled table columns
after the calculation. ``compile(properties=(...))`` can include derived model
properties in the same numerical batch; ``ModelPath.column`` also supports
computing such properties on demand.

Acquire observations for those fields
-------------------------------------

We have entered prices by hand. Next identify which instrument supplies each
price, configure its provider, and select the observations known at a given time.

Identify the data being supplied
--------------------------------

A holding name identifies a position in our account. A feed symbol identifies
an instrument in a provider's namespace. Keep those concepts separate: two
strategy sleeves may hold the same instrument, and two providers may use
different symbols for it.

Define a serializable identity record:

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: Equity
   :language: python

For timestamped market prints we also need price, publication time, and volume.
These are the fields of ``PriceHistory``:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: PriceHistory
   :end-before:     def __post_init__
   :language: python

Its constructor checks matching column lengths, positive prices and volumes,
and increasing observation/publication times. A print cannot become available
before its economic time.

Define a typed acquisition contract
-----------------------------------

A provider is a configured request returning one type when explicitly loaded.
This generic contract is enough for prices, transactions, or an index:

.. literalinclude:: ../../../examples/data.py
   :pyobject: Provider
   :language: python

For equities, add an identity that is available before acquisition:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: EquityProvider
   :language: python

We can implement that interface with a tiny offline provider before considering
files or services. It describes one instrument and creates one published print
when loaded:

>>> from dataclasses import dataclass
>>> from examples.data import Series
>>> from examples.portfolio.models import Equity
>>> from examples.portfolio.sources import EquityProvider
>>> from examples.portfolio.sources import PriceHistory, timestamp
>>> at = timestamp("2026-01-05T16:00:00Z")
>>> @dataclass(frozen=True)
... class FlatPrice(EquityProvider):
...     symbol: str
...     price: float
...
...     @property
...     def instrument(self) -> Equity:
...         return Equity(self.symbol, "local")
...
...     def load(self) -> PriceHistory:
...         return PriceHistory(self.instrument, Series((at,), (self.price,)), (at,), (1,))
>>> request = FlatPrice("AAPL", 100)
>>> request.instrument.symbol, request.load().prices.values
('AAPL', (100.0,))

Replace the acquisition implementation
--------------------------------------

For an external CSV, keep the same contract and change only ``load``. The real
adapter selects a namespace and symbol, validates the currency, and constructs
that same history record:

.. raw:: html

   <details><summary>Inspect the CSV adapter and bundled fixture provider</summary>

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: CSVEquityProvider
   :language: python

Rows use ``provider,symbol,currency,at,available_at,price,volume``. A service-backed
adapter could implement these two operations with its own query parameters.
For the walkthrough, the dummy provider supplies the bundled synthetic CSV:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: DummyEquityProvider
   :language: python

.. raw:: html

   </details>

>>> from examples.portfolio.sources import DummyEquityProvider
>>> apple_request = DummyEquityProvider(symbol="AAPL")
>>> history = apple_request.load()
>>> history.equity.symbol, history.prices.values[0]
('AAPL', 100.0)

Sample knowledge, not future data
---------------------------------

A consumer asks for the latest print published by its decision time. The age
limit is measured from the print's economic time. Here is that selection:

.. literalinclude:: ../../../examples/portfolio/sources.py
   :pyobject: PriceHistory.mark
   :dedent: 4
   :language: python

>>> from datetime import timedelta
>>> mark = timestamp("2026-01-05T16:01:00Z")
>>> history.mark(mark, max_age=timedelta(days=4))
100.0

``sample(times, max_age=...)`` applies this operation on a deliberate feature or
valuation grid. Publication time and economic time remain separate; a later
quote never fills an earlier decision.

Declare the component and request together
------------------------------------------

A second provider dictionary would repeat every holding name. Instead, pair a
model type with its binding in a declaration. ``ModelFactory`` provides the
two operations we need: create a model case, or declare a bound child:

.. code-block:: python

   from dataclasses import dataclass
   from typing import Generic, TypeVar, cast
   from modello import Modello, ScalarValue
   _Model = TypeVar("_Model", bound=Modello)
   _Binding = TypeVar("_Binding")

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: ModelFactory
   :language: python

.. raw:: html

   <details><summary>Inspect how a bound declaration becomes a model component</summary>

Its declaration operation uses this record and helper:

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: _BoundComponent
   :language: python

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: bound_component
   :language: python

While defining a ``BoundMembers`` class, its namespace records the binding and
passes the model type to Modello's normal component machinery:

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: _BindingNamespace.__setitem__
   :dedent: 4
   :language: python

.. raw:: html

   </details>

The adapter carries those bindings through inheritance and checks that their
names exactly match the component domain. ``EquityHoldings`` adds the financial
type checks:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: EquityHoldings
   :language: python

The configured factory ``equity`` specialises ``ModelFactory`` for
``EquityPosition`` and ``EquityProvider``. We can now declare our holdings once:

>>> from modello import component
>>> from examples.portfolio.valuation import EquityHoldings, EquityBasket, equity
>>> class Holdings(EquityHoldings):
...     aapl = equity.component(DummyEquityProvider(symbol="AAPL"))
...     msft = equity.component(DummyEquityProvider(symbol="MSFT"))
>>> class Basket(EquityBasket[Holdings]):
...     members = component(Holdings)
...     market_value = EquityBasket.total(members)

The basket extends the reductions from chapter 1 with an ``EquityHolding``
view: a symbolic position, its provider, and its capital base:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: EquityHolding
   :language: python

The provider map and contextual positions are projections of the declaration:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: EquityBasket.providers
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: EquityBasket.positions
   :dedent: 4
   :language: python

``instruments`` projects provider identities, while ``values`` and ``weights``
project these holding views. Adding or replacing a declared member carries its
request with it; each model instance still receives independent symbols.

Assemble and evaluate the portfolio
-----------------------------------

``NamedPortfolio`` inherits the ``SymbolicModel`` compilation methods developed
in chapter 2. Evaluation returns another instance of the same portfolio class,
with the same typed component tree and numeric scalar fields.

Declare the complete portfolio and give it known quantities and cash:

>>> from examples.portfolio.models import USDCash
>>> from examples.portfolio.valuation import NamedPortfolio
>>> class Portfolio(NamedPortfolio[Holdings]):
...     basket = component(Basket)
...     cash = component(USDCash)
...     nav = basket.capital
...     nav = basket.market_value + cash.amount
>>> book = Portfolio(
...     "book",
...     basket=Basket(
...         "basket",
...         members=Holdings(
...             "equities",
...             aapl=equity("apple", quantity=10),
...             msft=equity("microsoft", quantity=20),
...         ),
...     ),
...     cash=USDCash("cash", amount=100),
... )
>>> market = book.load_market()

For this fixed-holding view, select the position price fields as inputs. The
ordinary compiler checks that these are independent and that all remaining
fields are determined by them. Include the derived weight properties in the
same numerical batch for later analytics:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: NamedPortfolio.pricer
   :dedent: 4
   :language: python

>>> from examples.symbolic import ModelPlan
>>> pricer = book.pricer()
>>> type(pricer) is ModelPlan
True

The market boundary carries identified quotes. Define a scalar quote and a
timestamped group of them before using it:

.. literalinclude:: ../../../examples/portfolio/models.py
   :pyobject: EquityPrice
   :language: python

Before binding these observations, match holding names and complete instrument
identities. Histories, scalar quotes, and price vectors all carry an ``equity``
attribute, so they share this small typed contract:

.. code-block:: python

   from typing import Protocol, TypeVar

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: _EquityData
   :language: python

.. code-block:: python

   _Data = TypeVar("_Data", bound=_EquityData)

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: NamedPortfolio.match_market
   :dedent: 4
   :language: python

``MarketSnapshot`` carries the observation time and quotes. Its ``sample``
constructor selects causal prices; its ``knowns`` method implements the evaluator's
``ModelData`` contract, binding those prices to the model case supplied by the
shared evaluator:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: MarketSnapshot
   :language: python

>>> from examples.portfolio.valuation import MarketSnapshot
>>> snapshot = MarketSnapshot.sample(market, at=mark, max_price_age=timedelta(days=4))
>>> valued = pricer.evaluate(snapshot)
>>> float(valued.nav), float(valued.basket.members.aapl.market_value)
(2100.0, 1000.0)

This is a ``Portfolio`` containing a ``Basket``, ``Holdings``, and ordinary
``EquityPosition`` children. The contextual holding still supplies its weight
and provider. Select it with the typed member reference:

.. literalinclude:: ../../../examples/portfolio/valuation.py
   :pyobject: EquityBasket.holding
   :dedent: 4
   :language: python

>>> apple = valued.basket.holding(valued.basket.members.aapl)
>>> float(valued.nav), round(float(apple.weight), 4)
(2100.0, 0.4762)
>>> apple.instrument == apple.provider.instrument
True

The same declaration supplies symbolic expressions, evaluated values, and
traversable field paths. For example, a debugger or table renderer can inspect
the evaluated inputs together with their source expressions:

>>> observed = tuple(field for field in pricer.walk(valued) if field.role == "input")
>>> for field in observed:
...     print(".".join(field.path), field.value)
basket.members.aapl.instrument.price 100.0
basket.members.msft.instrument.price 50.0

The same snapshot can bind to another compatible portfolio case, because
``knowns`` receives that case's own fields. The original ``book`` stays symbolic;
repeated evaluation uses the compiled calculation without another joint solve.
Account replay will reuse this snapshot binding alongside known quantities and
cash, rather than build a separate price-input adapter.

Loading remains an explicit operation. The portfolio's ``load_market`` method
uses this batch loader to preserve holding names and reuse equivalent requests:

.. literalinclude:: ../../../examples/data.py
   :pyobject: load_providers
   :language: python

The supplied frozen requests compare all parameters. Cache keys include the
concrete provider class; a different path, symbol, namespace, currency, or
provider type remains a different request. A later load starts a fresh cache.
For example, separate named sleeves can share an acquired history while keeping
independent quantities and weights:

>>> from examples.data import NamedMap
>>> from examples.data import load_providers
>>> requests = NamedMap({"growth": apple_request, "income": apple_request})
>>> shared = load_providers(requests)
>>> shared["growth"] is shared["income"]
True

The finished two-equity declarations above are in
``examples/portfolio/demo.py``. We will reuse them to add an external
transaction log and a changing account state in :doc:`accounts`.

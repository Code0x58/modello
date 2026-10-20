4. Acquire data and bind it to holdings
=======================================

We have scalar models, named projections, and an evaluator. Now we need to say
which instrument supplies each price, how to acquire its history, and which
observations were available at a particular time.

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

.. literalinclude:: ../../../examples/compute/providers.py
   :pyobject: Provider
   :language: python

For equities, add an identity that is available before acquisition:

.. literalinclude:: ../../../examples/portfolio/providers.py
   :pyobject: EquityProvider
   :language: python

We can implement that interface with a tiny offline provider before considering
files or services. It describes one instrument and creates one published print
when loaded:

>>> from dataclasses import dataclass
>>> from examples.compute.series import Series
>>> from examples.portfolio.models import Equity
>>> from examples.portfolio.providers import EquityProvider
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

.. literalinclude:: ../../../examples/portfolio/providers.py
   :pyobject: CSVEquityProvider
   :language: python

Rows use ``provider,symbol,currency,at,available_at,price,volume``. A service-backed
adapter could implement these two operations with its own query parameters.
For the walkthrough, the dummy provider supplies the bundled synthetic CSV:

.. literalinclude:: ../../../examples/portfolio/providers.py
   :pyobject: DummyEquityProvider
   :language: python

.. raw:: html

   </details>

>>> from examples.portfolio.providers import DummyEquityProvider
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

.. literalinclude:: ../../../examples/portfolio/basket.py
   :pyobject: EquityHoldings
   :language: python

The configured factory ``equity`` specialises ``ModelFactory`` for
``EquityPosition`` and ``EquityProvider``. We can now declare our holdings once:

>>> from modello import component
>>> from examples.portfolio.basket import EquityHoldings, EquityBasket, equity
>>> class Holdings(EquityHoldings):
...     aapl = equity.component(DummyEquityProvider(symbol="AAPL"))
...     msft = equity.component(DummyEquityProvider(symbol="MSFT"))
>>> class Basket(EquityBasket[Holdings]):
...     members = component(Holdings)
...     market_value = EquityBasket.total(members)

The basket extends the reductions from chapter 2 with an ``EquityHolding``
view: a symbolic position, its provider, and its capital base:

.. literalinclude:: ../../../examples/portfolio/basket.py
   :pyobject: EquityHolding
   :language: python

The provider map and contextual positions are projections of the declaration:

.. literalinclude:: ../../../examples/portfolio/basket.py
   :pyobject: EquityBasket.providers
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/basket.py
   :pyobject: EquityBasket.positions
   :dedent: 4
   :language: python

``instruments`` projects provider identities, while ``values`` and ``weights``
project these holding views. Adding or replacing a declared member carries its
request with it; each model instance still receives independent symbols.

Assemble and evaluate the portfolio
-----------------------------------

``NamedPortfolio`` inherits the ``SymbolicModel`` compilation methods developed
in chapter 3. Evaluation returns another instance of the same portfolio class,
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

>>> from examples.model_plan import ModelPlan
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
constructor selects causal prices; its ``knowns`` method implements chapter 3's
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

.. literalinclude:: ../../../examples/portfolio/basket.py
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

.. literalinclude:: ../../../examples/compute/providers.py
   :pyobject: load_providers
   :language: python

The supplied frozen requests compare all parameters. Cache keys include the
concrete provider class; a different path, symbol, namespace, currency, or
provider type remains a different request. A later load starts a fresh cache.
For example, separate named sleeves can share an acquired history while keeping
independent quantities and weights:

>>> from examples.compute.named_map import NamedMap
>>> from examples.compute.providers import load_providers
>>> requests = NamedMap({"growth": apple_request, "income": apple_request})
>>> shared = load_providers(requests)
>>> shared["growth"] is shared["income"]
True

The finished two-equity declarations above are in
``examples/portfolio/explicit.py``. We will reuse them to add an external
transaction log and a changing account state in :doc:`accounts`.

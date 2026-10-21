1. Build a typed portfolio
==========================

We will build a portfolio holding ten AAPL shares, twenty MSFT shares, and USD
100 cash. First we need a price, a position, and a cash balance. Dates, providers,
and allocation policies can wait until we have equations to evaluate.

Define a price and a position
-----------------------------

A quote has an unknown positive price. A position combines a quote with a
number of shares. Specialise it to nonnegative shares for a long equity holding:

>>> from modello import Modello, InstanceDummy, component
>>> class Quote(Modello):
...     price = InstanceDummy("price", positive=True)
>>> class Position(Modello):
...     instrument = component(Quote)
...     quantity = InstanceDummy("quantity", real=True)
...     market_value = quantity * instrument.price
>>> class EquityPosition(Position):
...     quantity = InstanceDummy("quantity", nonnegative=True)

These are complete model definitions. ``component(Quote)`` scopes the quote's
symbols inside each position and exposes a typed ``Quote`` instance after
solving. The signed base position will also support virtual attribution sleeves
later; both use the same valuation equation. Give one position its quantity and
price:

>>> apple = EquityPosition("apple", quantity=10, instrument=Quote("quote", price=100))
>>> apple.market_value
1000

The expression also answers an inverse question. Supply market value and price
instead of quantity:

>>> inferred = EquityPosition(
...     "inferred", market_value=1000, instrument=Quote("quote", price=100)
... )
>>> inferred.quantity
10

We changed the known quantities, leaving the equation intact. Each construction
is a separate case with independent unknown symbols.

Group holdings and add cash
---------------------------

A portfolio needs named positions and a cash balance. Declare the position
domain explicitly, using the same model type for both equities:

>>> class Holdings(Modello):
...     aapl = component(EquityPosition)
...     msft = component(EquityPosition)
>>> class USDCash(Modello):
...     amount = InstanceDummy("amount", nonnegative=True)

Next, give the equity group a value and an allocation base. ``capital`` will
represent the amount against which its weight is measured:

>>> class Basket(Modello):
...     members = component(Holdings)
...     capital = InstanceDummy("capital", positive=True)
...     market_value = members.aapl.market_value + members.msft.market_value
...     weight = market_value / capital

A portfolio adds cash to the basket. Its net asset value (NAV) is the basket's
value plus cash. The two assignments to ``nav`` express both its equality to
the basket's capital base and its valuation equation:

>>> class Portfolio(Modello):
...     basket = component(Basket)
...     cash = component(USDCash)
...     nav = basket.capital
...     nav = basket.market_value + cash.amount

This is the hierarchy we have built so far:

.. mermaid::
   :caption: Scalar portfolio composition
   :alt: A portfolio contains a basket and cash. The basket contains a named holding group with two equity positions, each containing a quote.

   flowchart TD
       portfolio[Portfolio] --> basket[Basket]
       portfolio --> cash[USD cash]
       basket --> members[Holdings]
       members --> apple[aapl: EquityPosition]
       members --> microsoft[msft: EquityPosition]
       apple --> aq[Quote]
       microsoft --> mq[Quote]

Construct a known portfolio
---------------------------

Supply typed instances for the components we just declared:

>>> members = Holdings(
...     "equities",
...     aapl=apple,
...     msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=50)),
... )
>>> book = Portfolio(
...     "book", basket=Basket("basket", members=members), cash=USDCash("cash", amount=100)
... )
>>> book.basket.market_value, book.nav
(2000, 2100)
>>> book.basket.capital, book.basket.weight
(2100, 20/21)
>>> book.cash.amount / book.nav
1/21

The basket and cash weights add to one. A member's account weight is its own
market value divided by the same capital base; it need not sum to one with
only the other equities.

Solve through the complete hierarchy
------------------------------------

The joint solver can infer a quantity inside the basket from a portfolio-level
constraint. Leave Apple's quantity unknown and supply the total NAV:

>>> unknown = Holdings(
...     "equities",
...     aapl=EquityPosition("apple", instrument=Quote("quote", price=100)),
...     msft=members.msft,
... )
>>> target = Portfolio(
...     "target",
...     basket=Basket("basket", members=unknown),
...     cash=USDCash("cash", amount=100),
...     nav=2100,
... )
>>> target.basket.members.aapl.quantity
10
>>> bool(unknown.aapl.quantity.free_symbols)
True

The supplied source case remains symbolic; the new portfolio case contains
the inferred quantity. This is already useful Modello behaviour, before any
research infrastructure is added.

Derive aggregates from the declaration
---------------------------------------

The first basket repeated ``aapl`` and ``msft`` in its value equation. Derive
that expression from the typed members so adding a holding changes one declaration.

Keep names while projecting values
----------------------------------

``NamedMap[T]`` stores named values of one common type. Start with a small
numeric example to see the operations we need:

>>> from examples.data import NamedMap
>>> values = NamedMap({"aapl": 1000, "msft": 1000})
>>> weights = values.map(lambda value: value / 2100)
>>> tuple(weights), round(weights["aapl"], 4)
(('aapl', 'msft'), 0.4762)
>>> sum(values.values()) + 100
2100

A projection changes values while keeping names and order. Combining two maps
must align by name, even if the right-hand input arrives in another order. These
are the implementation's two operations:

.. literalinclude:: ../../../examples/data.py
   :pyobject: NamedMap.map
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/data.py
   :pyobject: NamedMap.combine
   :dedent: 4
   :language: python

>>> returns = NamedMap({"msft": -0.02, "aapl": 0.02})
>>> contributions = weights.combine(returns, lambda weight, change: weight * change)
>>> tuple(contributions), round(sum(contributions.values()), 12)
(('aapl', 'msft'), 0.0)

``combine`` requires the same names on both sides. It performs no implicit
subset selection or filling. A reduction such as ``sum`` can then consume the
projected values. An empty map has an empty projection and a zero sum.

Read the component declaration
------------------------------

Use the holding declaration above to derive its member domain.

``component_members(group, EquityPosition)`` turns those declared children into
a ``NamedMap[EquityPosition]``. The helper first identifies the model class,
then reads its component names and checks their common type:

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: component_members
   :language: python

This is an example-layer reflection adapter over Modello's current metadata.
Keeping it in one module gives the rest of the application a typed operation;
the core solver still receives ordinary scalar expressions and components.
The adapter accepts a model class, a class-body component reference, or a
populated instance:

>>> from examples.named_members import component_members
>>> tuple(component_members(Holdings, EquityPosition))
('aapl', 'msft')
>>> declared_values = component_members(Holdings, EquityPosition).map(
...     lambda position: position.market_value
... )
>>> tuple(declared_values)
('aapl', 'msft')

Build a basket from that domain
-------------------------------

Move the reduction and projections onto a basket model. The capital base and
weight equation are the same as in the scalar basket above; membership is now a
parameter of its declaration:

>>> from typing import Generic, TypeVar
>>> from sympy import Expr, S
>>> Members = TypeVar("Members", bound=Modello)
>>> Declared = TypeVar("Declared", bound=Modello)
>>> class ValueBasket(Modello, Generic[Members]):
...     members: Members
...     capital = InstanceDummy("capital", positive=True)
...     market_value = InstanceDummy("market_value", nonnegative=True)
...     weight = market_value / capital
...
...     @staticmethod
...     def total(members: Modello) -> Expr:
...         return sum(
...             (
...                 position.market_value
...                 for position in component_members(members, EquityPosition).values()
...             ),
...             S.Zero,
...         )
...
...     @staticmethod
...     def define(member_type: type[Declared]) -> "type[ValueBasket[Declared]]":
...         class DeclaredBasket(ValueBasket[Declared]):
...             members = component(member_type)
...             market_value = ValueBasket.total(members)
...
...         return DeclaredBasket
...
...     @property
...     def positions(self) -> NamedMap[EquityPosition]:
...         return component_members(self.members, EquityPosition)
...
...     @property
...     def values(self) -> NamedMap[Expr]:
...         return self.positions.map(lambda position: position.market_value)
...
...     @property
...     def weights(self) -> NamedMap[Expr]:
...         return self.values.map(lambda value: value / self.capital)
>>> Basket = ValueBasket.define(Holdings)
>>> members = Holdings(
...     "equities",
...     aapl=EquityPosition("apple", quantity=10),
...     msft=EquityPosition("microsoft", quantity=20),
... )
>>> basket = Basket("basket", members=members, capital=2100)
>>> tuple(basket.positions), tuple(basket.values), tuple(basket.weights)
(('aapl', 'msft'), ('aapl', 'msft'), ('aapl', 'msft'))

The reduction runs once when defining the class. It expands the member values
into a scalar expression that participates in the usual joint solve. Numeric
prices have not been acquired or evaluated yet.

Extend the declaration once
---------------------------

Add Alphabet using the same position type, then derive a basket for the larger
domain. The reduction and projection definitions stay the same:

>>> class ThreeHoldings(Holdings):
...     googl = component(EquityPosition)
>>> ThreeBasket = ValueBasket.define(ThreeHoldings)
>>> tuple(component_members(ThreeBasket.members, EquityPosition))
('aapl', 'msft', 'googl')

The installed ``EquityBasket`` adds provider bindings and contextual holding
views to this same pattern. Its ``define`` method retains the member type and
reuses the class for repeated compositions. The :doc:`next chapter <data>`
adds observation times, compilation, and configured providers.

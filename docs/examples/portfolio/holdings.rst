1. Value a holding, then a portfolio
====================================

We will build a portfolio holding ten AAPL shares, twenty MSFT shares, and USD
100 cash. First we need a price, a position, and a cash balance. Dates, providers,
and allocation policies can wait until we have equations to evaluate.

Define a price and a position
-----------------------------

A quote has an unknown positive price. A position combines a quote with a
nonnegative number of shares:

>>> from modello import Modello, InstanceDummy, component
>>> class Quote(Modello):
...     price = InstanceDummy("price", positive=True)
>>> class EquityPosition(Modello):
...     instrument = component(Quote)
...     quantity = InstanceDummy("quantity", nonnegative=True)
...     market_value = quantity * instrument.price

These are complete model definitions. ``component(Quote)`` scopes the quote's
symbols inside each position and exposes a typed ``Quote`` instance after
solving. Give one position its quantity and price:

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

Where repetition starts
-----------------------

Our basket explicitly adds ``aapl.market_value`` and ``msft.market_value``.
Adding an equity would require changing that formula, and repeating each name
again for values and weights would make the model harder to extend. In the
:doc:`next chapter <named_members>` we derive these expressions from the declared
member domain.

The shared scalar definitions in the installed example live in
``examples/portfolio/models.py``. Its completed two-equity composition is in
``examples/portfolio/explicit.py``; after building the additional layers, we
will be able to run it with ``python -m examples.portfolio``.

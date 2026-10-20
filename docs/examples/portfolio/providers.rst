Typed data providers
====================

A provider is a configured request with a declared result type.
``Provider[T]`` is an abstract class in ``examples/compute/providers.py``;
its ``load()`` method returns ``T``. Provider configuration is available while
constructing models, and data acquisition happens when the caller loads it.
This keeps symbolic solving separate from external I/O.

Configure an equity request
----------------------------

``EquityProvider`` specialises the contract as ``Provider[PriceHistory]``.
Its ``instrument`` property describes the requested equity before loading:

>>> from examples.portfolio.providers import DummyEquityProvider
>>> request = DummyEquityProvider(symbol="AAPL")
>>> request.instrument.symbol, request.instrument.currency
('AAPL', 'USD')
>>> history = request.load()
>>> type(history).__name__, history.prices.values[0]
('PriceHistory', 100.0)

The dummy implementation serves the bundled synthetic AAPL/MSFT print fixtures.
``CSVEquityProvider`` reads the same format from a supplied path:

.. code-block:: python

   from pathlib import Path
   from examples.portfolio.providers import CSVEquityProvider

   request = CSVEquityProvider(
       symbol="AAPL",
       path=Path("market.csv"),
       namespace="my-feed",
       currency="USD",
   )
   history = request.load()

Rows use ``provider,symbol,currency,at,available_at,price,volume``. The CSV
provider selects the namespace and symbol, checks currency, and preserves the
observation/publication times and independent trade volumes. The ``provider``
text in an ``Equity`` record or CSV is a serializable symbol namespace; the
configured object supplies the acquisition behaviour and request parameters.

Keep holding names and source identifiers separate
--------------------------------------------------

A holding name identifies an account position. A provider symbol identifies
an instrument in a particular feed. For example, separate growth and income
holdings can request the same equity:

>>> from examples.portfolio.basket import EquityBasket, EquityHoldings, equity
>>> from examples.portfolio.valuation import NamedPortfolio
>>> class Sleeves(EquityHoldings):
...     growth = equity.component(request)
...     income = equity.component(request)
>>> Basket = EquityBasket.define(Sleeves)
>>> portfolio = NamedPortfolio.define(Basket)("sleeves")
>>> market = portfolio.load_market()
>>> tuple(market), market["growth"] is market["income"]
(('growth', 'income'), True)

Each ``equity.component(request)`` declaration couples the symbolic position and its
provider under one holding name. The basket projects ``providers`` and
``instruments`` from those declarations; inheritance carries both parts
together. Quantities and weights stay independent for each holding. Symbols
may differ between feeds or contain punctuation that is inconvenient in a
Python component attribute.

The loader caches by concrete provider class and request during that call. The supplied
providers are frozen dataclasses whose equality includes their concrete type
and every parameter. Equivalent requests share data; different paths, symbols,
currencies, namespaces, or provider classes remain separate. A later call starts
a fresh cache, and loading errors propagate to the caller.

Extend the contract
--------------------

An equity adapter for another data service implements ``EquityProvider``:
``instrument`` returns the quote identity, and ``load()`` returns a matching
``PriceHistory``. A Yahoo-backed adapter could accept its symbol and query
parameters through its constructor and implement these same two operations.
This example ships local fixture and CSV implementations.

The generic ``Provider[T]`` contract also serves other data types. The CSV
transaction source returns ``TransactionLog``; the CSV index source returns
``Series[datetime]``. New providers choose their own result type and parameters.
Custom providers must be hashable, with equality stable during a batch load;
immutable request records make equivalent-query reuse straightforward.

Shared symbol text alone is insufficient to combine instruments or requests.
Other instrument kinds can have their own typed provider specialisations.
Cross-provider security identification, currency conversion, and corporate
changes need explicit data and models at their respective boundaries.

Named members and portfolio computation
=======================================

The :doc:`portfolio experiment <holdings>` can be extended by treating members
as a finite map from names to typed values. Positions, market values, weights,
and returns all use the same names. A projection changes the values while
retaining that domain; a reduction turns those values into an aggregate such
as NAV. Covariance has two member-name axes.

The named map lives in ``examples/compute/named_map.py``; the Modello adapter is
in ``examples/named_members.py``. The portfolio application is in
``examples/portfolio/named.py``. Install the examples and run it:

.. code-block:: console

   $ pip install "modello[examples]"
   $ python -m examples.portfolio.named

Project and combine named values
--------------------------------

``NamedMap[T]`` is an immutable mapping with values of type ``T``. It owns its
membership and preserves declaration order. A name identifies a member;
distinct names can refer to equal values without being collapsed.

>>> from examples.compute.named_map import NamedMap
>>> values = NamedMap({"aapl": 1000, "msft": 1000})
>>> weights = values.map(lambda value: value / 2100)
>>> tuple(weights)
('aapl', 'msft')
>>> round(weights["aapl"], 4)
0.4762
>>> sum(values.values()) + 100
2100

``combine`` joins by name, using the left map's declaration order. For example,
weights and returns can arrive in different orders:

>>> returns = NamedMap({"msft": -0.02, "aapl": 0.02})
>>> contributions = weights.combine(returns, lambda weight, change: weight * change)
>>> tuple(contributions)
('aapl', 'msft')
>>> round(sum(contributions.values()), 12)
0.0

Both maps must have exactly the same names. Missing or additional members are
reported before arithmetic. Subset selection, filling missing observations, or
joining different domains would each need an explicit operation with its own
meaning. An empty map has an empty projection and its values sum to zero.

Declare the portfolio's member domain
-------------------------------------

Members are still declared as named, typed model components:

.. code-block:: python

   class EquityMembers(Modello):
       aapl = component(EquityPosition)
       msft = component(EquityPosition)

``component_members(group, EquityPosition)`` exposes those components as a
``NamedMap[EquityPosition]``. It works on model classes, class-body references,
and populated models. The adapter checks that every component belongs to the
supplied common type.

The declared ``Basket`` binds an ``EquityProvider`` request to each holding
name. Its reduction is an ordinary scalar equation. For definitions assembled
from runtime configuration, ``EquityBasket.define(EquityMembers, providers)``
builds the same composition. Its total is:

.. code-block:: python

   market_value = sum(
       position.market_value
       for position in component_members(members, EquityPosition).values()
   )

``NamedPortfolio.define(Basket)`` places that basket beside cash and relates its
capital base to account NAV:

.. code-block:: python

   nav = basket.capital
   nav = basket.market_value + cash.amount

The populated basket associates every symbolic position with its configured
provider, derived ``Equity`` identity, and capital base through an ``EquityHolding`` view. Its ``positions``,
``values``, and ``weights`` maps preserve the declared names. Member allocation
is defined once by ``EquityHolding.weight``; the basket projects that property:

.. code-block:: python

   self.positions.map(lambda position: position.weight)

The numerical plan compiles these weight expressions alongside values and NAV,
so the time-series evaluation uses the model's allocation formulas too.

The portfolio forwards these maps for account-level consumers. The named
holding group also retains ordinary typed attributes such as
``portfolio.basket.members.aapl``. Value and weight projections follow the
declared holdings automatically.

Add a member once
-----------------

The worked example adds Alphabet by extending the declaration:

.. code-block:: python

   class ThreeEquities(EquityMembers):
       googl = component(EquityPosition)

   class ThreeEquityBasket(EquityBasket[ThreeEquities]):
       members = component(ThreeEquities)
       providers = NamedMap({**EQUITY_PROVIDERS, "googl": DummyEquityProvider(symbol="GOOGL")})
       market_value = EquityBasket.total(members)

Supply its holding and price vector under ``googl``. The same aggregate,
valuation, weight, attribution, and risk calculations handle all three members:

>>> from examples.portfolio.named import (
...     example_named_market_data, example_named_portfolio,
... )
>>> holdings = example_named_portfolio()
>>> tuple(holdings.positions), tuple(holdings.weights)
(('aapl', 'msft', 'googl'), ('aapl', 'msft', 'googl'))
>>> path = holdings.pricer().over(example_named_market_data())
>>> path.nav.values
(3100.0, 3105.0, 3030.0, 3190.0, 3230.0)
>>> tuple(path.weights)
('aapl', 'msft', 'googl')
>>> result = path.analytics(periods_per_year=252)
>>> round(result.performance.total_return * 100, 4)
4.1935
>>> round(result.latest_weights_risk.volatility * 100, 4)
51.483
>>> round(sum(result.latest_weights_risk.contributions.values()) * 100, 4)
51.483

Covariance is indexed as ``result.annualized_covariance["aapl"]["googl"]``.
Cash retains its own value and weight paths and has zero return and risk
contribution. An empty equity declaration also works: NAV is the cash balance,
the projected equity maps are empty, and equity risk is zero.

The pricing plan binds holding names to instrument records before constructing a numeric
input layout. Feed order can change without changing the result. Each feed
member must also match the declared instrument identity and observation dates.
Names identify positions, so separate named positions may hold the same equity.
Provider-symbol identity and member identity remain distinct concepts. The same
instrument can appear in separate strategy sleeves with independent quantities
and weights; :doc:`equivalent provider requests <providers>` share one history
within a load. The provider map is the stored binding; instrument identities
are derived from it.

The original experiment's assumptions and distinction between observed-path
volatility and fixed-reference-weight risk apply here as well.

The :doc:`account system <accounts>` extends the same valuation and analytics
with changing positions, an external transaction log, signals, and execution
benchmarks. It separates the member-name axis from the observation, publication,
and decision times.

Implications for a compute framework
------------------------------------

This gives the experiment a small algebra: lookup, name-preserving projection,
exact-domain combination, and reduction. The position domain and observation
axis remain separate. Market value is pointwise on both axes; returns shift the
observation axis; covariance reduces time and introduces a second member axis.

The basket definition expands the member reduction into ordinary scalar equations
once per definition. NAV still supports inverse valuation through the joint
solver. A new member declaration produces a new model and pricing plan. Adding
holdings and data is then a binding task, rather than a rewrite of the formulas.

Weights in this prototype are projected queries over the solved model. Making
an entire map a solver input, or imposing constraints on selected projected
members, would require model-field support for named domains. That is a useful
next question before making this example layer part of the library API.

``NamedMap`` preserves the value type through projection and combination. The
current typing checks a common component type and checks names at runtime;
static knowledge of particular keys and their concrete subtypes is a further
typing problem. Domain types could also distinguish unrelated maps that happen
to use the same names. Units, currency conversion, two-axis alignment, and
execution backends remain independent dimensions of the design.

The example isolates its use of Modello's internal field metadata in the
``component_members`` adapter. This lets the experiment exercise the proposed
semantics while retaining the existing scalar solver and typed member
declarations.

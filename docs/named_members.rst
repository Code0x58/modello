Named members and portfolio computation
=======================================

The :doc:`portfolio experiment <portfolio>` can be extended by treating members
as a finite map from names to typed values. Positions, market values, weights,
and returns all use the same names. A projection changes the values while
retaining that domain; a reduction turns those values into an aggregate such
as NAV. Covariance has two member-name axes.

The named map lives in ``examples/compute/named_map.py``; the Modello adapter is
in ``examples/named_members.py``. The portfolio application is in
``examples/named_portfolio.py``. Run it from a checkout:

.. code-block:: console

   $ uv run --locked python -m examples.named_portfolio

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
       aapl = component(ApplePosition)
       msft = component(MicrosoftPosition)

``component_members(group, EquityPosition)`` exposes those components as a
``NamedMap[EquityPosition]``. It works both while defining equations and on a
populated model. Concrete instrument types remain attached to each position.
The adapter checks that all members belong to the supplied common type.

``portfolio_model(EquityMembers)`` builds a model whose NAV equation is:

.. code-block:: python

   nav = sum(
       position.market_value
       for position in component_members(equities, EquityPosition).values()
   ) + cash.amount

The resulting portfolio exposes ``positions``, ``values``, and ``weights`` as
maps. Its weight query is simply:

.. code-block:: python

   self.values.map(lambda value: value / self.nav)

Construct the member group using position instances, then supply that group to
the portfolio. The named group also retains its ordinary typed attributes such
as ``equities.aapl``.

Add a member once
-----------------

The worked example adds Alphabet by extending the declaration:

.. code-block:: python

   class ThreeEquities(EquityMembers):
       googl = component(AlphabetPosition)

   Portfolio = portfolio_model(ThreeEquities)

Supply its holding and price vector under ``googl``. The same aggregate,
valuation, weight, attribution, and risk calculations handle all three members:

>>> from examples.named_portfolio import (
...     NamedPortfolioPricer, example_named_market_data, example_named_portfolio,
... )
>>> holdings = example_named_portfolio()
>>> tuple(holdings.positions), tuple(holdings.weights)
(('aapl', 'msft', 'googl'), ('aapl', 'msft', 'googl'))
>>> path = NamedPortfolioPricer(holdings)(example_named_market_data())
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

The pricing plan binds names to instrument types before constructing a numeric
input layout. Feed order can change without changing the result. Each feed
member must also match the declared instrument identity and observation dates.
Names identify positions, so separate named positions may hold the same equity.
Ticker identity and member identity remain distinct concepts.

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

The model factory expands the member reduction into ordinary scalar equations
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

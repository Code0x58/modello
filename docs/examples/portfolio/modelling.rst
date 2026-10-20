3. Evaluate symbolic models over time
=====================================

A scalar position answers a question at one point. To value fixed holdings on
several dates, we need an observation axis and a numerical evaluator. These are
separate requirements: the axis describes data, while the evaluator consumes
the equations already resolved by Modello.

Put observations on an explicit axis
------------------------------------

``Series`` pairs finite numeric values with strictly increasing dates or
UTC-normalised timestamps. Here are five synthetic daily observations:

>>> from datetime import date
>>> from examples.compute.series import Series
>>> times = tuple(date(2026, 1, day) for day in range(5, 10))
>>> prices = Series(times, (100, 102, 98, 104, 103))
>>> prices.values
(100.0, 102.0, 98.0, 104.0, 103.0)

A lag labels the earlier values with the later observation dates. That makes
an interval's opening value align with its closing value:

.. literalinclude:: ../../../examples/compute/series.py
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

Compile one resolved position
-----------------------------

A fixed position has known shares and one remaining price symbol. We can name
that input and the value output using the maps from the preceding chapter:

>>> from examples.portfolio.models import EquityPosition
>>> from examples.compute.named_map import NamedMap
>>> from examples.compute.scalar_plan import ScalarPlan
>>> position = EquityPosition("fixed", quantity=10)
>>> plan = ScalarPlan(
...     NamedMap({"price": position.instrument.price}),
...     NamedMap({"value": position.market_value}),
... )
>>> dict(plan.evaluate(NamedMap({"price": 100})))
{'value': 1000.0}
>>> plan.over(times, NamedMap({"price": prices}))["value"].values
(1000.0, 1020.0, 980.0, 1040.0, 1030.0)

Here is the compilation step. The plan requires independent input symbols and
checks that every output's remaining symbols are supplied. ``lambdify`` turns
the resolved expressions into one callable using the standard-library math
backend and common-subexpression elimination:

.. literalinclude:: ../../../examples/compute/scalar_plan.py
   :pyobject: ScalarPlan.__init__
   :dedent: 4
   :language: python

The vector operation aligns names and axes, then calls that calculation on
each row. It does no symbolic solving per observation:

.. literalinclude:: ../../../examples/compute/scalar_plan.py
   :pyobject: ScalarPlan.over
   :dedent: 4
   :language: python

``ScalarPlan`` works with ordinary SymPy expressions too. It has no knowledge
of portfolios, prices, or Modello classes. Its numerical outputs are floats;
the original symbolic case still supports exact arithmetic.

Give a model its evaluation methods
-----------------------------------

Repeatedly spelling out inputs and outputs would distract from the model.
The small ``SymbolicModel`` subclass packages those view-building operations.
It declares default ``inputs`` and ``outputs`` as tuples of field names:

.. code-block:: python

   from functools import cache
   from typing import ClassVar, TypeAlias, cast
   from sympy import Expr
   from modello import Modello
   from examples.compute.named_map import NamedMap
   from examples.compute.scalar_plan import ScalarPlan
   from examples.compute.series import Series, Time

   ViewFields: TypeAlias = tuple[str, ...] | NamedMap[Expr]

``Time`` is the series module's axis type variable. ``ViewFields`` selects
either model fields by name or arbitrary named expressions:

.. literalinclude:: ../../../examples/symbolic.py
   :pyobject: SymbolicModel
   :language: python

``compile`` uses the current case, ``plan`` caches the class's usual unbound
view, and ``over`` applies that default view to aligned columns.

Build a position-valuing model on top of the component we already defined:

>>> from typing import ClassVar
>>> from modello import component
>>> from examples.symbolic import SymbolicModel
>>> class PositionValue(SymbolicModel):
...     position = component(EquityPosition)
...     quantity = position.quantity
...     price = position.instrument.price
...     value = position.market_value
...     inputs: ClassVar[tuple[str, ...]] = ("quantity", "price")
...     outputs: ClassVar[tuple[str, ...]] = ("value",)
>>> fixed = PositionValue("fixed", position=EquityPosition("holding", quantity=10))
>>> view = fixed.compile(inputs=("price",))
>>> view.over(times, NamedMap({"price": prices}))["value"].values
(1000.0, 1020.0, 980.0, 1040.0, 1030.0)

No additional valuation formula was introduced: the model aliases the existing
position's fields. The default unbound plan can instead vary both quantities
and prices, which will be useful when the account trades.

Select another solving direction
--------------------------------

Suppose market value is observed and quantity is the desired output. The same
model can solve the exact constraint:

>>> inverse = PositionValue("quantity", price=100, value=1500)
>>> inverse.quantity
15

To repeat that inverse question numerically, supply a symbolic observation when
constructing the case, then choose its input and output fields:

>>> from sympy import Symbol
>>> observed = Symbol("observed_value", positive=True)
>>> fitted = PositionValue("observed", price=100, value=observed)
>>> inverse_view = fitted.compile(inputs=("value",), outputs=("quantity",))
>>> dict(inverse_view.evaluate(NamedMap({"value": 1500})))
{'quantity': 15.0}

This is exact calibration to an observation. Statistical estimation from noisy
samples requires its own model and estimation layer. Providers and calendars
also remain outside this numerical executor.

We can now evaluate a model on data, but we have entered those prices by hand.
The :doc:`next chapter <providers>` defines the data contract, acquires prices,
and couples each request to its holding declaration.

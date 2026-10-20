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

.. literalinclude:: ../../../examples/model_plan.py
   :pyobject: Known
   :language: python

``ModelPlan[Model]`` reads the resolved scalar fields and component paths from
an existing case. Independent symbolic fields become inputs; every declared
field remains in the result. Bind the fixed quantity as usual, leaving the
price unknown:

>>> from examples.portfolio.models import EquityPosition, Quote
>>> from examples.model_plan import Known, ModelPlan
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

Expose the declared fields for inspection
-----------------------------------------

A view can describe each field's real component path, its source expression,
and its role as an input, a known constant, or a derived value:

.. literalinclude:: ../../../examples/model_plan.py
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

.. literalinclude:: ../../../examples/model_plan.py
   :pyobject: ModelPlan.__init__
   :dedent: 4
   :language: python

The example layer isolates access to Modello's current component metadata.
``ScalarPlan`` compiles the resolved expressions with SymPy's ``lambdify``;
``ModelPlan`` restores the declared model shape after evaluation:

.. literalinclude:: ../../../examples/model_plan.py
   :pyobject: ModelPlan.evaluate
   :dedent: 4
   :language: python

.. raw:: html

   <details><summary>Inspect field traversal, copying, and the scalar compiler</summary>

.. literalinclude:: ../../../examples/model_plan.py
   :pyobject: _get_field
   :language: python

.. literalinclude:: ../../../examples/model_plan.py
   :pyobject: _copy_model
   :language: python

The copy operation reuses model classes and symbol scopes, replaces scalar
values, and recomputes cached properties. It invokes no symbolic solver.

.. literalinclude:: ../../../examples/compute/scalar_plan.py
   :pyobject: ScalarPlan.__init__
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/compute/scalar_plan.py
   :pyobject: ScalarPlan.over
   :dedent: 4
   :language: python

.. raw:: html

   </details>

Some interval-report adapters need selected numeric columns rather than a
complete numeric model. ``examples/symbolic.py`` supplies a small
``SymbolicModel`` helper for those named scalar views. Its default ``inputs``
and ``outputs`` are tuples of model field names:

.. raw:: html

   <details><summary>Inspect the selected-scalar-view helper used by later reports</summary>

.. literalinclude:: ../../../examples/symbolic.py
   :language: python

.. raw:: html

   </details>

We can now evaluate a model on data, but we have entered those prices by hand.
The :doc:`next chapter <providers>` defines the data contract, acquires prices,
and couples each request to its holding declaration.

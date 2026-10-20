2. Derive calculations from named holdings
==========================================

The :doc:`first chapter <holdings>` gave every holding a typed component, but
repeated its name in aggregate expressions. We now need two small tools: an
immutable map that preserves member names, and an adapter that obtains those
names from a Modello declaration.

Keep names while projecting values
----------------------------------

``NamedMap[T]`` stores named values of one common type. Start with a small
numeric example to see the operations we need:

>>> from examples.compute.named_map import NamedMap
>>> values = NamedMap({"aapl": 1000, "msft": 1000})
>>> weights = values.map(lambda value: value / 2100)
>>> tuple(weights), round(weights["aapl"], 4)
(('aapl', 'msft'), 0.4762)
>>> sum(values.values()) + 100
2100

A projection changes values while keeping names and order. Combining two maps
must align by name, even if the right-hand input arrives in another order. These
are the implementation's two operations:

.. literalinclude:: ../../../examples/compute/named_map.py
   :pyobject: NamedMap.map
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/compute/named_map.py
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

Reuse the scalar position definition we built previously. The packaged module
contains that definition, so this chapter can also run independently:

>>> from modello import Modello, InstanceDummy, component
>>> from examples.portfolio.models import EquityPosition
>>> class Holdings(Modello):
...     aapl = component(EquityPosition)
...     msft = component(EquityPosition)

``component_members(group, EquityPosition)`` turns those declared children into
a ``NamedMap[EquityPosition]``. The helper first identifies the model class,
then reads its component names and checks their common type:

.. raw:: html

   <details><summary>Inspect the component-reflection adapter</summary>

.. code-block:: python

   from typing import TypeVar, cast
   from modello import Modello, _ModelReference
   _Model = TypeVar("_Model", bound=Modello)

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: _model_type
   :language: python

.. literalinclude:: ../../../examples/named_members.py
   :pyobject: component_members
   :language: python

.. raw:: html

   </details>

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
weight equation are the same as in the first chapter; membership is now a
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

The installed ``EquityBasket`` will add provider bindings and contextual holding
views to this pattern. Its ``define`` method also retains the concrete member
type and reuses the class for a repeated composition. We will add those details
when data acquisition becomes a requirement in :doc:`providers`.

There are two domains here: named holdings and observation times. We have
built the first. The :doc:`next chapter <modelling>` adds the observation axis
and evaluates these scalar expressions repeatedly over it.

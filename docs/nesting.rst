Component models
================

Use ``component(Model)`` when a part of your problem already has its own
quantities and relationships. The parent can use those quantities directly;
observations on either side can determine unknowns on the other.

This guide develops the second step of :doc:`getting-started`.

Reuse a smaller model
---------------------

A parcel's diagonal depends on its base diagonal. Keep the base relationship
in a triangle model and reuse it:

>>> from sympy import sqrt
>>> from modello import InstanceDummy, Modello, component
>>> class Triangle(Modello):
...     a = InstanceDummy("a", positive=True)
...     b = InstanceDummy("b", positive=True)
...     c = sqrt(a**2 + b**2)
>>> class Box(Modello):
...     base = component(Triangle)
...     height = InstanceDummy("height", positive=True)
...     diagonal = sqrt(base.c**2 + height**2)
>>> parcel = Box("parcel", base=Triangle("base", a=3, b=4), height=12)
>>> parcel.base.c, parcel.diagonal
(5, 13)

``parcel.base`` is a ``Triangle`` instance. Component paths such as ``base.c``
let you express relationships across the whole model.

Infer a quantity inside a component
-----------------------------------

Supply exactly 13 units of diagonal clearance, rather than the base width:

>>> partial_base = Triangle("base", b=4)
>>> inferred = Box("width", base=partial_base, height=12, diagonal=13)
>>> inferred.base.a, inferred.base.c
(3, 5)

The parent and child equations are solved together. The supplied component
remains available for other questions; completing this parcel does not change it:

>>> bool(partial_base.a.free_symbols)
True

Omitting a component leaves its quantities unknown. Reusing a partial instance
in several components gives each its own unknowns unless you connect them.

Connect quantities in different children
----------------------------------------

Suppose two braces share a horizontal run. A parent can declare that equality
while each child keeps its own diagonal relationship:

>>> class LinkedBraces(Modello):
...     first = component(Triangle)
...     second = component(Triangle)
...     run = second.a
...     run = first.a
>>> linked = LinkedBraces(
...     "shared", first=Triangle("first", a=3, b=4), second=Triangle("second", b=8)
... )
>>> linked.first.a, linked.second.a
(3, 3)

The first ``run`` assignment names the existing ``second.a`` quantity. The
second relates it to ``first.a``. An observation of ``run`` can therefore
constrain either child. Components can contain further components; the same
attribute-path syntax works at each level.

Use the same model for new observations
---------------------------------------

Construct a new case to change measurements. Changing an attribute on a solved
instance does not recalculate dependent quantities. Use :doc:`solutions` when
you want to retain alternatives and add information to the same question.

Next, :doc:`vectors` covers collections of scalar observations, and
:doc:`model-vectors` covers collections whose entries are complete models.
The :doc:`API reference <api>` describes inheritance, copying, and methods.

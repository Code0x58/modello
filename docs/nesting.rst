Nested models
=============

Use ``component(Model)`` to build a model from smaller models. You can refer to
each component's quantities in expressions, then supply model instances with
the values you know.

Build a line from points
------------------------

First define a point, then a line whose length depends on its two endpoints:

>>> from sympy import sqrt
>>> from modello import InstanceDummy, Modello, component
>>> class Point(Modello):
...     x = InstanceDummy("x", real=True)
...     y = InstanceDummy("y", real=True)
>>> class Line(Modello):
...     start = component(Point)
...     end = component(Point)
...     length = sqrt((end.x - start.x)**2 + (end.y - start.y)**2)

Create the endpoints and pass them to the line:

>>> origin = Point("origin", x=0, y=0)
>>> endpoint = Point("endpoint", x=3, y=4)
>>> line = Line("line", start=origin, end=endpoint)
>>> line.length
5
>>> line.end.x, line.end.y
(3, 4)

``line.start`` and ``line.end`` are ``Point`` instances, with that type available
to editors and type checkers. Each component has its own quantities, even when
several components use the same model class.

Add another level
------------------

Components can themselves contain components. Use attribute paths to reach
quantities at any depth:

>>> class Drawing(Modello):
...     segment = component(Line)
...     horizontal_extent = segment.end.x - segment.start.x
>>> drawing = Drawing("drawing", segment=line)
>>> drawing.segment.length, drawing.horizontal_extent
(5, 3)

Solve across components
-----------------------

An equation in a parent can determine a missing value in a child. For example,
a box's height and diagonal determine the missing side of its triangular base:

>>> class Triangle(Modello):
...     a = InstanceDummy("a", positive=True)
...     b = InstanceDummy("b", positive=True)
...     c = sqrt(a**2 + b**2)
>>> class Box(Modello):
...     base = component(Triangle)
...     height = InstanceDummy("height", positive=True)
...     diagonal = sqrt(base.c**2 + height**2)
>>> base = Triangle("base", b=4)
>>> box = Box("box", base=base, height=12, diagonal=13)
>>> box.base.a, box.base.b, box.base.c
(3, 4, 5)

Modello solves the parent and its components together. The positive assumptions
on the triangle's sides select the positive solution.

The parent receives a copy of its component's values, so completing the base
inside the box leaves the original triangle symbolic:

>>> bool(base.a.free_symbols)
True
>>> box.base.a
3

Connect components through shared quantities
--------------------------------------------

A parent can also bind quantities in different children to each other. Build a
two-segment path whose second line starts where the first ends:

>>> class Path(Modello):
...     first = component(Line)
...     second = component(Line)
...     x = second.start.x
...     x = first.end.x
...     y = second.start.y
...     y = first.end.y
...     length = first.length + second.length
>>> path = Path(
...     "path", first=line, second=Line("next", end=Point("end", x=6, y=8))
... )
>>> path.second.start.x, path.second.start.y, path.length
(3, 4, 10)

The first ``x`` assignment makes it an alias of ``second.start.x``. The second
equates that quantity to ``first.end.x``; the ``y`` assignments do the same for
the other coordinate. Each line keeps its length equation. The path supplies
only the connections and the combined length, so the whole remains one
symbolic model. Known values on either side can determine unknowns on the other.

Work with symbolic inputs
-------------------------

Omit a value to leave it symbolic. Derived quantities remain expressions in
those unknowns until other equations determine them:

>>> partial = Point("partial", x=3)
>>> segment = Line("segment", start=origin, end=partial)
>>> bool(segment.end.y.free_symbols)
True
>>> segment.length == sqrt(9 + segment.end.y**2)
True

Omitting an entire component leaves its inputs symbolic too:

>>> symbolic_line = Line("symbolic", start=origin)
>>> symbolic_line.length == sqrt(symbolic_line.end.x**2 + symbolic_line.end.y**2)
True

You can reuse an instance to give several components the same starting values.
Each copy has its own unknowns, so subsequent equations can determine those
values independently:

>>> repeated = Line("repeated", start=partial, end=partial)
>>> repeated.start.x, repeated.end.x
(3, 3)
>>> repeated.start.y == repeated.end.y
False

Models are solved when they are constructed. Create a new model to solve with
different inputs; changing an attribute later does not recalculate other values.
See the :doc:`API reference <api>` for input types, inheritance, and model methods.
The :doc:`debugging guide <debugging>` covers unresolved quantities, conflicting
constraints, and multiple possible solutions.

The :doc:`practical applications <examples/applications/index>` apply the same
pattern to invoices, production yield, and connected circuits. The
:doc:`trade reconciliation example <examples/reconciliation/index>` keeps a
trusted PnL fixed and releases individual observations to investigate a break.

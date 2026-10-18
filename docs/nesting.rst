Nested models
=============

Declare a named component with ``component(Model)`` in another model's body.
Expressions can reference any descendant through its attribute path. For
example, a line contains two independently scoped points:

>>> from sympy import sqrt
>>> from modello import BoundInstanceDummy, InstanceDummy, Modello, component
>>> class Point(Modello):
...     x = InstanceDummy("x", real=True)
...     y = InstanceDummy("y", real=True)
>>> class Line(Modello):
...     start = component(Point)
...     end = component(Point)
...     length = sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
>>> line = Line("L", start=Point("start", x=0, y=0), end=Point("end", x=3, y=4))
>>> line.length
5
>>> isinstance(line.start, Point), line.start is line.end
(True, False)

The two ``Point`` declarations create independent symbols, even though they use
the same model class. Each root construction creates fresh symbols; names label
those symbols and do not establish shared identity.
``component(Point)`` gives ``line.start`` and ``line.end`` the static instance
type ``Point``. Assigning the raw model class also remains supported, but type
checkers infer that declaration as a class instead of an instance.

Deeper composition
------------------

Compose instances to the same depth as the declared component tree. The entire
tree forms one system of equations; children are populated from its solution.

>>> class Drawing(Modello):
...     segment = component(Line)
...     horizontal_extent = segment.end.x - segment.start.x
>>> drawing = Drawing("D", segment=line)
>>> drawing.segment.length, drawing.horizontal_extent
(5, 3)

Constraints on a parent can solve a child's inputs. For a positive right-angle
triangle, a box's diagonal can determine a missing side of its base:

>>> class Triangle(Modello):
...     a = InstanceDummy("a", positive=True)
...     b = InstanceDummy("b", positive=True)
...     c = sqrt(a**2 + b**2)
>>> class Box(Modello):
...     base = component(Triangle)
...     height = InstanceDummy("height", positive=True)
...     diagonal = sqrt(base.c**2 + height**2)
>>> base = Triangle("base", b=4)
>>> box = Box("B", base=base, height=12, diagonal=13)
>>> box.base.a, box.base.b, box.base.c
(3, 4, 5)

Omitted inputs and existing instances
-------------------------------------

Omitting inputs leaves parameters symbolic. Declared formulas are expanded
before solving supplied constraints so that an unconstrained output is expressed
in terms of its inputs. Underconstrained equations no longer ask SymPy to choose
arbitrary particular values, such as zeroes for unspecified coordinates.

Pass an existing instance to copy its declared values into the new component's
scope. All descendant values are included, and the source is not mutated.
Unresolved symbols owned by that instance are rebound to the new tree:

>>> source = Point("source", x=3)
>>> copied = Line("copy", start=Point("origin", x=0, y=0), end=source)
>>> copied.end.x, copied.end is source
(3, False)
>>> isinstance(copied.end.y, BoundInstanceDummy), copied.end.y == source.y
(True, False)
>>> copied.length == sqrt(9 + copied.end.y**2)
True

Reusing an instance for two component arguments makes two independent snapshots.
Ordinary instance attributes and extra fields of a supplied subclass are not
copied: only the declared component's field tree is used. Compatible subclasses
must retain that tree's attribute paths.

Scalar input expressions can reference the component class's own symbols, or
the containing model's full paths. External SymPy symbols remain parameters;
solving the tree does not assign values to external model instances.

Constructor inputs are annotated as ``ScalarValue | Modello``. Scalar values
include SymPy expressions, Python numbers, decimals, fractions, and strings that
SymPy can parse. Dictionaries and other mappings are not constructor inputs.
The exact component class and declared field names are validated at runtime.
The package ships a ``py.typed`` marker, so installed distributions expose the
same annotations as the source checkout without a type checker plugin.

Inheritance and validation
--------------------------

Replacing a component in a subclass rebinds inherited expressions by matching
relative attribute paths. Existing base classes are not modified. A replacement
that removes a referenced field, or splits a referenced alias into different
symbols, is rejected rather than leaving a stale symbol in an inherited formula.
Redefine the affected expressions when changing a component's shape.

As in earlier versions, later model bases override earlier model bases when
multiple inheritance merges model fields. Ordinary methods use Python's normal
method resolution order. A name cannot change between scalar and component
kinds. Attributes starting with ``_modello_`` are reserved for metadata.

Unknown input names and malformed values raise :class:`TypeError` with the full
field path. Component arguments must be instances of their declared model class
(including compatible subclasses); dictionaries, other mappings, and ``None``
are rejected. Omit a component argument to leave its inputs symbolic.
Conflicting values supplied through scalar aliases are all retained as equations.
Inconsistent systems and multiple discrete solutions raise :class:`ValueError`.
Symbol assumptions are preserved, and candidates that provably violate them or
produce undefined values are rejected. SymPy may leave assumptions on external
symbolic parameters undecided.

Construction boundaries
-----------------------

Every supplied component is first constructed normally and must be valid on its
own: it can have symbolic inputs, but inconsistent constraints or multiple
discrete solutions are rejected before composition. Omit inputs that the parent
will determine, as in the triangle example above.

Composition copies supplied values into objects of the declared component types.
Those copies have the declared methods, but their custom ``__init__`` and
``__new__`` methods are not called again. Express component constraints on the
class instead of depending on constructor side effects.

Models are solved at construction time. Mutating a scalar or source instance
later does not recompute an existing tree. Named components are supported;
collections of components and dynamically changing tree shapes are not.
Declare another independent component with ``component(Model)`` again; assigning an
existing component reference as a new class attribute is rejected.
Cycles between scalar equations are solved simultaneously; they do not create
cycles in the component tree.

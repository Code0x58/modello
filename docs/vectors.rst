Vectors and indexed formulas
============================

Declare a finite vector when a model quantity has an ordered collection of
scalar entries. Its coordinates participate in the same joint solve as scalar
fields and nested components. Keep sums, products, and calculus as SymPy
operators to preserve their mathematical form in inspection and the explorer.

A window with a missing observation
-----------------------------------

.. code-block:: pycon

   >>> from sympy import Sum, Product
   >>> from modello import Index, Modello, Vector, inspect_model, vector
   >>> class Window(Modello):
   ...     samples = Vector("x", size=3, real=True)
   ...     i = Index("i")
   ...     total = Sum(samples[i], (i, 0, len(samples) - 1))
   ...     growth = Product(1 + samples[i], (i, 0, len(samples) - 1))
   ...     changes = vector(samples[i] - samples[i - 1], (i, 1, len(samples) - 1))
   >>> result = Window("window", samples=[1, None, 3], total=6)
   >>> result.samples
   [1, 2, 3]
   >>> result.total, result.growth, result.changes
   (6, 24, [1, 1])

``None`` in a constructor's vector observation leaves that coordinate
unobserved. It does not assign a special symbolic value or replace its defining
equation. Observing the total determines the missing sample. Without that
observation the entry stays symbolic, and the other results depend on it.

Each vector has a fixed, nonnegative integer size. An input sequence must have
exactly that many entries. Inputs may also be explicit ``Vector`` values;
``Vector(1, 2, 3)`` and ``Vector([1, 2, 3])`` are equivalent. Assumptions on a
declared vector apply to **every coordinate**. Declare an assumed vector first,
then constrain it with a formula to enforce requirements on derived entries:

.. code-block:: python

   class GrowthFactors(Modello):
       returns = Vector("r", size=3, real=True)
       factors = Vector("g", size=3, positive=True)
       i = Index("i")
       factors = vector(1 + returns[i], (i, 0, 2))
       growth = factors.product(i)

The positive factors exclude returns at or below −100%, including when those
returns are inferred from another observation.

Indices, limits, and vector operations
--------------------------------------

``Index`` creates a local integer symbol. It can be declared in a model class
body without becoming an observable model field. Distinct declarations have
distinct identities, including two indices with the same printed name.
``Sum`` and ``Product`` bind their indices using normal SymPy semantics.

Indexing and slicing use Python's zero-based order, including negative integer
indices. Symbolic indexing creates a scalar expression; substituting an integer
selects that coordinate. ``len(v)`` and iteration describe its finite shape.

.. code-block:: pycon

   >>> x, y = Vector(1, 2, 3), Vector(3, 2, 1)
   >>> x[-1], x[1:]
   (3, [2, 3])
   >>> (x + y).doit(), (2 * x).doit()
   ([4, 4, 4], [2, 4, 6])
   >>> x.dot(y).doit(), (x @ y).doit()
   (10, 10)
   >>> x.sum().doit(), x.product().doit()
   (6, 6)

Addition and subtraction are elementwise and require equal sizes.
Multiplication and division scale a vector by a scalar. ``dot`` and ``@`` give a
bilinear contraction without complex conjugation. Explicitly construct any
conjugated entries needed for a Hermitian inner product. No operation silently
truncates a vector or broadcasts a scalar through addition.

The reduction helpers return unevaluated SymPy ``Sum`` and ``Product`` objects.
Use their ``.doit()`` method for immediate symbolic evaluation outside a model.
Empty vectors have sum zero and product one. Dot products of two empty vectors
are zero.

``vector(expression, (index, first, last))`` describes a finite comprehension.
Its endpoints are **inclusive**, like SymPy's sum and product limits, rather
than Python's ``range`` stop. The resulting vector itself is always zero-based:

.. code-block:: pycon

   >>> i = Index("i")
   >>> squares = vector(i**2, (i, 2, 4))
   >>> squares[0], squares[-1], len(squares)
   (4, 16, 3)
   >>> squares.doit()
   [4, 9, 16]

The original function and iteration domain remain available for rendering.
``.doit()`` materializes a comprehension as explicit vector values. Symbolic
equality follows SymPy's structural convention, so use evaluated vectors when
comparing a comprehension with an explicit list of expressions.

Use model factories for different finite shapes; see
:doc:`examples/timeseries/index`. A subclass may refine assumptions or a formula
without changing the inherited size. Changing the size of an inherited vector
raises ``ModelDefinitionError`` so inherited indices cannot silently change
meaning.

Composition, observations, and inspection
-----------------------------------------

Vectors can alias other declared coordinates or whole vectors, including those
on a component. Aliases retain one scalar identity per coordinate, and
contradictory observations remain constraints. Component copies preserve vector
shape and supplied/inferred provenance without mutating their sources. Reusing
the same source component twice still gives independent scopes.

Inspection does not evaluate the declared operators:

.. code-block:: pycon

   >>> problem = inspect_model(Window, "formula", samples=[1, None, 3])
   >>> isinstance(problem.constraints[0].right, Sum)
   True
   >>> problem.symbols.samples[1] == problem.field("samples", "1").symbol
   True
   >>> problem.vector("samples") == problem.symbols.samples
   True
   >>> problem.format(problem.constraints[0].right)
   'Sum(samples[i], (i, 0, 2))'

``problem.fields`` includes scalar coordinate paths such as
``("samples", "1")``. The explorer and printed expressions show these as
``samples[1]``. ``problem.without("samples")`` removes the complete constructor
observation group while retaining definitions and domains. A supplied component
still forms one observation group for all its fields, including its vectors.
Integer coordinate path parts are also accepted, such as
``problem.field("samples", 1)``.
``problem.vectors`` is a read-only mapping from field paths to symbolic vectors,
including nested and empty fields. ``problem.vector("window", "samples")``
selects a complete vector by component path without depending on private model
metadata.

Solution families accept vector observations and queries. Querying a vector
returns coupled tuples, preserving order and relationships between coordinates:

.. code-block:: pycon

   >>> from sympy import FiniteSet, Tuple
   >>> family = problem.solutions()
   >>> solved = family.observe(total=6)
   >>> solved.values(family.symbols.samples) == FiniteSet(Tuple(1, 2, 3))
   True
   >>> solved.expression(family.symbols.samples)
   [1, 2, 3]
   >>> solved.one().samples
   [1, 2, 3]

Pass a vector to ``problem.solutions(problem.symbols.samples)`` to prefer its
coordinates, subject to the remaining relationships. This does not make them
independent if constraints couple them. The :doc:`solutions` guide explains
conditional families and finite-state selection.

Family observations accumulate constraints. A ``None`` entry adds no new
observation and does not release an earlier one. Explorer proposals instead
replace the selected observation pattern; ``None`` or ``?`` there releases a
coordinate. These follow the respective scalar observation semantics.

Integrals, derivatives, and limits
----------------------------------

Use SymPy's ``Integral``, ``Derivative``, and ``Limit`` directly. Keep ordinary
continuous integration variables outside the model class, or construct them
within an expression; an ``Index`` is specifically integer-valued.

.. code-block:: pycon

   >>> from sympy import Integral, Symbol
   >>> from modello import InstanceDummy
   >>> t = Symbol("t", real=True)
   >>> class Exposure(Modello):
   ...     rate = InstanceDummy("rate", real=True)
   ...     horizon = InstanceDummy("horizon", positive=True)
   ...     accumulated = Integral(rate * t, (t, 0, horizon))
   >>> Exposure("forward", rate=2, horizon=3).accumulated
   9
   >>> Exposure("inverse", horizon=3, accumulated=9).rate
   2

Definitions retain operators for display. Algebra expands referenced model
formulas within their binder scope, evaluates operators, and then applies
observations. This ordering matters for a derivative of another declared
formula:

.. code-block:: pycon

   >>> from sympy import Derivative
   >>> class Sensitivity(Modello):
   ...     x = InstanceDummy("x", real=True)
   ...     square = x**2
   ...     slope = Derivative(square, x)
   >>> Sensitivity("point", x=3).slope
   6

The derivative is taken before substituting the observed point. Unresolved
operators remain symbolic when the constructor's backend can retain them;
the exact solution-family API requires them to evaluate. Calculus through
cyclic model definitions is unsupported. These are algebraic models, rather
than a general differential-equation solver or automatic numerical quadrature.

The native operators follow the
`SymPy concrete mathematics <https://docs.sympy.org/latest/modules/concrete.html>`_
and `calculus <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>`_
APIs, including their limits and evaluation behaviour.

The explorer
-------------

The optional :doc:`gui` automatically shows vector symbols with an arrow,
scalar coordinates with subscripts, sums with Σ, products with Π, and integrals
with ∫. Vector comprehensions display an indexed equality and its explicit
iteration domain. Multiple sum/product limits and integral differentials retain
their full order.

Select **Inspect vector** for its values, entry roles, connected quantities,
and defining formula. Supply an entire observation such as ``[1, ?, 3/2]`` or
edit entries independently. ``?`` removes an observation for that coordinate;
it does not remove the model's equation. Whole-vector editing is available when
all entries can be edited. Coordinate labels, units, and numeric formatting
inherit vector metadata unless explicitly overridden.

Run the finite-path examples locally:

.. code-block:: console

   $ uv run --extra gui python -m examples.gui_timeseries
   $ uv run --extra gui python -m examples.gui_timeseries --example returns
   $ uv run --extra gui python -m examples.gui_timeseries --example rolling
   $ uv run --extra gui python -m examples.gui_timeseries --example exposure

Open http://127.0.0.1:5059. The guides explain the models' assumptions; the
example data are illustrative. Vector positions do not acquire timestamps,
units, interpolation, missing-data filling, or annualization rules implicitly.

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
``v.shape`` returns its single axis as a tuple, such as ``(3,)`` or ``(0,)``.

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

During model evaluation, reductions over finite vector entries iterate over the
stated bounds directly. Ordinary symbolic sums and products retain SymPy's
closed-form algorithms. This avoids searching for a closed form for an
arbitrary observation sequence while preserving the original formula for display.
Native empty, negative, and reversed-bound conventions remain in effect.

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

A formula assigned to an already declared vector constrains its coordinates
and retains their assumptions. This includes direct references to other fields:
declaring a positive vector before assigning it another vector's formula still
requires positive entries. A fresh ``Vector(name, size=..., ...)`` declaration
can refine inherited coordinate assumptions. See the
:doc:`recursive state example <examples/timeseries/index>` for a formula that
relates each coordinate to its predecessor.

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

``problem.vector_definitions`` is a read-only mapping of complete vector formulas,
with their original function, local index, and bounds retained. Its keys are
paths such as ``("changes",)`` or ``("window", "changes")``. Only defined vectors
appear there; independent vectors and direct aliases remain in ``problem.vectors``.
The formulas use this problem's symbols and remain unchanged by observations.

Export formulas for notebooks and reports with ``problem.latex(expression)``
or ``family.latex(expression)``. These preserve model paths and scalar indices,
show vector symbols with arrows, and retain native sum and calculus notation:

.. code-block:: pycon

   >>> problem.latex(problem.symbols.samples) == r"\vec{\mathrm{samples}}"
   True
   >>> problem.latex(problem.constraints[0].right) == r"\sum_{i=0}^{2} \mathrm{samples}_{i}"
   True
   >>> definition = problem.vector_definitions[("changes",)]
   >>> "\\middle|" in problem.latex(definition)
   True

Exporting a family does not substitute values automatically; pass
``family.expression(...)`` when you want its reduced formula. Native
``sympy.latex`` also prints explicit vectors as columns and comprehensions
with their iteration domains. The model-aware methods supply the field and
vector names without changing the expressions. They require no GUI dependency.

Use ``symbol_names`` and ``vector_names`` for report notation. Their keys are
actual symbols or vectors; their values are LaTeX fragments. A vector name also
labels its scalar coordinates, with an explicit scalar override taking priority:

.. code-block:: pycon

   >>> names = {problem.symbols.samples: "x"}
   >>> problem.latex(problem.symbols.samples[1], vector_names=names)
   'x_{1}'
   >>> problem.latex(
   ...     problem.symbols.samples[1], vector_names=names,
   ...     symbol_names={problem.symbols.samples[1]: r"\xi"},
   ... )
   '\\xi'

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

Individual coordinates also work as controls in ``family.section(...)`` and
the plotting examples' model views. A section fixes the other coordinates at
a checked reference case; vector results retain their original order.

An observed bound may select a prefix of a fixed-size vector. Its sum or
product evaluates after that observation is applied. Exact families require
all remaining indexing and bound-variable operators to evaluate: supply an
integer selection or finite bounds before calling ``solutions()``. They do
not enumerate free symbolic vector indices. Bounds must select valid entries;
substituting an out-of-range index gives an undefined scalar, excluded from
finite solution states. Direct integer indexing still raises ``IndexError``.

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

Queries and conditions use the same formula expansion:

.. code-block:: pycon

   >>> from sympy import Eq
   >>> family = inspect_model(Sensitivity, "sensitivity").solutions()
   >>> gradient = Derivative(family.symbols.square, family.symbols.x)
   >>> family.expression(gradient) == 2 * family.symbols.x
   True
   >>> family.where(Eq(gradient, 6)).one().x
   3
   >>> family.observe(x=3).expression(gradient)
   6

The derivative is taken before substituting the observed point. It follows
the quantity's declared formula, treating the differentiation variable as
independent. To study a response along a constrained family, first obtain its
reduced expression. For example, fixing the total of two allocation weights
means that increasing one weight decreases the other:

.. code-block:: pycon

   >>> class Allocation(Modello):
   ...     weights = Vector("w", size=2, nonnegative=True)
   ...     total = weights.sum()
   ...     concentration = weights.dot(weights)
   >>> problem = inspect_model(Allocation, "allocation", total=1)
   >>> family = problem.solutions(problem.symbols.weights)
   >>> coordinate = family.parameters[0].symbol
   >>> response = family.expression(family.symbols.concentration)
   >>> response == coordinate**2 + (1 - coordinate)**2
   True
   >>> gradient = Derivative(response, coordinate)
   >>> family.expression(gradient) == 4 * coordinate - 2
   True
   >>> family.where(Eq(gradient, 0)).one().weights
   [1/2, 1/2]

Here the reduced family has one independent coordinate. More general family
coordinates may remain coupled by conditions; obtain a suitable parameterization
before interpreting their derivatives as sensitivities along that family.
The stationary-point condition is explicit, as in the
:doc:`calibration example <examples/timeseries/index>`; the solver does not
choose an objective or optimize one automatically.

Unresolved operators remain symbolic when the constructor's backend can retain them;
the exact solution-family API requires them to evaluate. Calculus through
cyclic or multiply defined model quantities is unsupported, including through
intermediate formulas. Several defining equations can jointly constrain one
quantity without defining a unique function to differentiate. The ordinary
joint solve still retains those equations. These are algebraic models, rather
than a general differential-equation solver or automatic numerical quadrature.

During model evaluation, limits wait until their free quantities are resolved
by substitutions. Evaluating a generic parameterized limit first can discard exceptional parameter values.
For example, a limit's observed endpoint may be a removable singularity:

.. code-block:: pycon

   >>> from sympy import Limit, exp
   >>> class Response(Modello):
   ...     endpoint = InstanceDummy("endpoint", real=True)
   ...     response = Limit((exp(t) - 1) / t, t, endpoint)
   >>> Response("origin", endpoint=0).response
   1
   >>> inspect_model(Response, "origin", endpoint=0).solutions().one().response
   1

Supply the necessary observations before requesting an exact family for a model
with a parameterized limit. An unresolved limit remains available in symbolic
construction and inspection, and is rejected explicitly by the exact-family API.

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
   $ uv run --extra gui python -m examples.gui_timeseries --example trend

Open http://127.0.0.1:5059. The guides explain the models' assumptions; the
example data are illustrative. Vector positions do not acquire timestamps,
units, interpolation, missing-data filling, or annualization rules implicitly.

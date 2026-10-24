Vectors and indexed formulas
============================

Use ``Vector`` for an ordered collection of scalar quantities. Its entries use
the same equations, observations, and domains as other model fields. Use
:doc:`model-vectors` when each entry should contain a complete model, or
:doc:`series` for a relative law independent of its observation window.

Start with observations
-----------------------

Let the data determine the size instead of building a class for every length:

.. code-block:: pycon

   >>> from modello import InstanceDummy, Modello, Vector, inspect_model
   >>> class Window(Modello):
   ...     N = InstanceDummy("N", integer=True, nonnegative=True)
   ...     samples = Vector("x", size=N, real=True)
   ...     total = samples.sum()
   >>> case = Window("data", samples=[1, None, 3], total=6)
   >>> case.N, case.samples
   (3, [1, 2, 3])
   >>> Window("shorter", samples=[2, 5]).total
   7

``None`` leaves an entry unknown. The total determines the missing value;
without that observation it stays symbolic. Vector assumptions apply to every
entry, including inferred values.

Observed lengths and supplied size fields are solved together before scalar
equations. All lengths must agree. ``N=3`` also prepares three unknown entries
without supplying a vector. A literal ``size=3`` fixes the declaration's size.
Use ``v.shape[0]`` for an unresolved size; ``len(v)`` needs a finite integer.
Prepare the original class again for a different horizon.

Operations
----------

.. code-block:: pycon

   >>> x, y = Vector(1, 2, 3), Vector([3, 2, 1])
   >>> x[-1], x[1:], (x + y).doit(), (2 * x).doit()
   (3, [2, 3], [4, 4, 4], [2, 4, 6])
   >>> x.sum().doit(), x.product().doit(), x.dot(y).doit()
   (6, 6, 10)

Vectors are immutable. Integer indexing and slicing follow Python, including
negative indices. Addition and subtraction require equal shapes; multiplication
and division scale by a scalar. ``dot`` and ``@`` give a bilinear contraction
without complex conjugation. Empty vectors have sum zero and product one.

The reduction helpers return ordinary unevaluated SymPy ``Sum`` and ``Product``
objects. Model solving evaluates them; outside a model, call ``.doit()``.
Native sums, products, integrals, derivatives, and limits can also be declared
directly. Their formulas remain intact for inspection and rendering.

Comprehensions
--------------

``vector(expression, (index, first, last))`` includes both endpoints, like
SymPy limits. Its own positions always start at zero:

.. code-block:: pycon

   >>> from modello import Index, vector
   >>> i = Index("i")
   >>> squares = vector(i**2, (i, 2, 4))
   >>> squares[0], squares[-1], squares.doit()
   (4, 16, [4, 9, 16])

Use ``Index`` for a local integer variable; it is not an observable model
field. Bounds may contain declared integer size fields, such as
``vector(2*samples[i], (i, 0, N-1))``. A symbolic origin is also supported when
the length is known, for example ``(i, n, n+2)``.

To retain domains on a calculated vector, declare its coordinates before
assigning the formula:

.. code-block:: python

   factors = Vector("g", size=N, positive=True)
   factors = vector(1 + returns[i], (i, 0, N - 1))

This requires every gross return to be positive. Inheritance can refine domains
or formulas while keeping the declared shape; use a new model shape to change
its size expression.

Inspection and solution families
--------------------------------

.. code-block:: pycon

   >>> problem = inspect_model(Window, "window", samples=[1, None, 3])
   >>> problem.vector("samples") == problem.symbols.samples
   True
   >>> problem.field("samples", 1).symbol == problem.symbols.samples[1]
   True
   >>> family = problem.solutions().observe(total=6)
   >>> family.one().samples
   [1, 2, 3]
   >>> total = next(c for c in problem.constraints if c.path == ("total",))
   >>> "\\sum" in problem.latex(total.right)
   True

The read-only ``problem.vectors`` mapping contains complete vectors by field
path. ``problem.vector_definitions`` contains their declared formulas, and
``problem.fields`` includes individual coordinates. Nested paths work in the
same way. ``problem.latex`` and ``family.latex`` export named mathematical
formulas; ``symbol_names`` and ``vector_names`` can supply report notation.

Aliases share coordinate identities. Copied components keep their own scopes
and observation provenance. ``problem.without("samples")`` releases the whole
observation group while retaining its equations. Family observations accumulate:
``None`` adds no constraint and does not release an earlier observation.

Exact families require remaining indices and bound operators to resolve. Supply
finite bounds and valid coordinates before requesting one. Direct invalid
integer indexing raises ``IndexError``; an undefined symbolic selection is
excluded from finite solutions.

Calculus uses SymPy
-------------------

Use a real ``Coordinate`` for a local continuous variable, and ``InstanceDummy``
for a quantity that should be observed or inferred:

.. code-block:: pycon

   >>> from sympy import Derivative, Integral
   >>> from modello import Coordinate
   >>> class Exposure(Modello):
   ...     t = Coordinate("t", real=True)
   ...     rate = InstanceDummy("rate", real=True)
   ...     duration = InstanceDummy("duration", positive=True)
   ...     area = Integral(rate * t, (t, 0, duration))
   >>> Exposure("inverse", duration=3, area=9).rate
   2
   >>> class Sensitivity(Modello):
   ...     x = InstanceDummy("x", real=True)
   ...     square = x**2
   ...     slope = Derivative(square, x)
   >>> Sensitivity("point", x=3).slope
   6

Referenced formulas are expanded before differentiation; differentiation
variables stay independent until their derivatives are taken. To differentiate
along a constrained family, first obtain its reduced expression with
``family.expression``. Calculus through cyclic or multiply defined quantities
is unsupported. Parameterized limits wait for the necessary observations so
exceptional values are preserved.

See the native `SymPy concrete mathematics
<https://docs.sympy.org/latest/modules/concrete.html>`_ and
`calculus <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>`_
APIs for operator syntax. The :doc:`example gallery <examples/gallery/index>`
shows these operations, relative sequences, and typed account models in one app.

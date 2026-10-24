Vectors and indexed formulas
============================

Use ``Vector`` for measurements, predictions, or other ordered scalar quantities.
The :doc:`calibration walkthrough <getting-started>` shows how vector expressions
define a fitting objective. This guide covers the individual operations. Use
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

Observed lengths must agree with the declared sizes: two vectors of size ``N``
need matching lengths; a vector of size ``N+1`` needs one more entry. ``N=3``
also requests three unknown entries without supplying a vector. A literal
``size=3`` fixes the size.
Use ``v.shape[0]`` for an unresolved size; ``len(v)`` needs a finite integer.
Construct another case for a different observation window.

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

Inside a model, reductions are evaluated as part of answering the question.
For standalone calculations like these, call ``.doit()``.
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
   >>> family = problem.solutions().observe(total=6)
   >>> family.one().samples
   [1, 2, 3]

Use ``family.expression(...)`` to obtain a SymPy expression for further analysis,
or ``problem.latex(...)`` to export a named formula. See :doc:`solutions` for
adding conditions and selecting outputs, and :doc:`diagnostics` for field lookup.
Family observations accumulate: ``None`` adds no constraint and does not release
an earlier observation. Construct a new case to replace a dataset.

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
needs an explicit parametrization. Some limits and integrals remain symbolic
until the relevant observations are supplied. See :doc:`generality` for the
supported calculus and domain checks.

See the native `SymPy concrete mathematics
<https://docs.sympy.org/latest/modules/concrete.html>`_ and
`calculus <https://docs.sympy.org/latest/tutorials/intro-tutorial/calculus.html>`_
APIs for operator syntax. Continue with :doc:`model-vectors` to relate typed
states across a sequence, or try line fitting in the
:doc:`gallery <examples/gallery/index>`.

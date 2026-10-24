Work with SymPy
===============

Modello uses SymPy expressions for scalar relationships. Use native operators
and functions when writing a model, take its results to SymPy for further
analysis, and return observations or candidate answers to the same model.
There is no separate scalar expression language to translate.

Write native expressions
------------------------

Import mathematical operations from SymPy and model declarations from Modello:

.. code-block:: pycon

   >>> from sympy import Derivative, Integral, Rational
   >>> from modello import Coordinate, InstanceDummy, Modello
   >>> class Exposure(Modello):
   ...     t = Coordinate("t", real=True)
   ...     rate = InstanceDummy("rate", real=True)
   ...     duration = InstanceDummy("duration", positive=True)
   ...     area = Integral(rate*t, (t, 0, duration))
   ...     sensitivity = Derivative(area, rate)
   >>> case = Exposure("measured", duration=3, area=Rational(9, 2))
   >>> case.rate, case.sensitivity
   (1, 9/2)

Named dependencies expand before differentiation. Native ``Sum``, ``Product``,
``Piecewise``, ``Limit``, and symbolic functions follow the same expression
path. See :ref:`symbolic-calculus` for calculus semantics and :doc:`generality`
for the supported solver classes. Exact solution families require exact
values: prefer ``Rational("0.1")`` to a binary floating-point observation.
Custom scalar wrappers can use SymPy's ``_sympy_`` conversion protocol. Inputs
use this conversion without a numerical round trip; call ``sympify`` explicitly
when adapting such a wrapper for a statically annotated caller.

Choose symbols by their role
----------------------------

* ``InstanceDummy`` declares a quantity to observe or infer. Each prepared
  problem and component receives its own identities.
* ``Index`` and ``Coordinate`` name local integer and continuous variables.
  They do not add observable fields.
* An ordinary SymPy ``Symbol`` remains an external parameter. Declaring a model
  field does not make every symbol in its expression a model quantity.

Use ``problem.symbols`` or ``family.symbols`` to build expressions about a
prepared model. Creating another symbol with the same printed name does not
refer to that quantity. A whole ``Modello`` instance or ``ModelVector`` remains
structured; select a scalar field or ``project`` its members to a scalar vector.

``Series`` describes an indexed scalar family, rather than the Taylor expansion
produced by ``sympy.series``. ``Recurrence`` supplies its causal history and
relative rule; an unrelated native ``Function`` does not acquire that rule
automatically. Query a law for a scalar expression or a finite vector view.
See :doc:`series` and :doc:`model-vectors` for scalar laws versus typed windows.

Exchange finite vector data
---------------------------

``Vector`` accepts native one-dimensional arrays and concrete row or column
matrices, as well as scalar sequences:

.. code-block:: pycon

   >>> from sympy import Array, Matrix
   >>> from modello import Vector
   >>> Vector(Array([1, 2, 3])) == Vector(Matrix([[1, 2, 3]]))
   True
   >>> Vector(Matrix([1, 2, 3]))
   [1, 2, 3]
   >>> class Baseline(Modello):
   ...     values = Array([1, 2, 3])
   ...     total = values.sum()
   >>> Baseline("baseline").total
   6

A native array or matrix assigned in a class body becomes a ``Vector`` field.
Mutable containers are copied into immutable scalar entries. The same
containers work as observations and determine symbolic sizes:

.. code-block:: pycon

   >>> from sympy import Lambda
   >>> class Calibration(Modello):
   ...     N = InstanceDummy("N", integer=True, nonnegative=True)
   ...     readings = Vector("x", size=N, real=True)
   ...     gain = InstanceDummy("gain", real=True)
   ...     u = Coordinate("u", real=True)
   ...     adjusted = readings.applyfunc(Lambda(u, gain*u))
   >>> calibrated = Calibration("forward", readings=Matrix([1, 2, 3]), gain=2)
   >>> calibrated.N, calibrated.adjusted
   (3, [2, 4, 6])
   >>> Calibration("inverse", readings=Array([1, 2, 3]), adjusted=Matrix([[2, 4, 6]])).gain
   2

Array rank must be one; matrices must have one row or one column. Empty vectors
also accept the canonical empty matrix. Higher-rank arrays, genuinely
two-dimensional matrices, and abstract ``MatrixSymbol`` objects are rejected
rather than flattened. Use a list or tuple with ``None`` entries for partial
observations.

Convert a finite vector back using native constructors. An array is rank one;
the default matrix conversion is a column:

.. code-block:: pycon

   >>> Array(calibrated.adjusted).shape, Matrix(calibrated.adjusted).shape
   ((3,), (3, 1))

Resolve a symbolic length before exporting individual entries. Modello does
not add general tensor or matrix algebra; use SymPy containers directly for it.

Reuse your own operations
-------------------------

An ordinary helper can return any supported scalar expression. ``applyfunc``
and ``vector`` accept native functions and ``Lambda`` expressions too:

.. code-block:: pycon

   >>> def squared_error(value):
   ...     return (value - 1)**2
   >>> Vector(1, 2, 3).applyfunc(squared_error).doit()
   [0, 1, 4]

Inside a class body, use an inline ``Lambda`` to capture model quantities, as
in ``Calibration``. Python lambdas do not close over class namespaces. Keep
reusable function objects outside the model class; assigning a named ``Lambda``
as a field raises an explanatory error because it is a function, not its scalar
result. Applied SymPy ``Function`` subclasses can supply their own evaluation
and differentiation rules. An arbitrary numerical callable does not acquire
an inverse or derivative automatically. See :ref:`custom-vector-operations`.

Continue an analysis in SymPy
-----------------------------

``family.expression`` resolves dependencies while retaining the family's
remaining coordinates. The returned scalar is a native SymPy expression:

.. code-block:: pycon

   >>> from sympy import diff, lambdify
   >>> from modello import inspect_model
   >>> class Square(Modello):
   ...     x = InstanceDummy("x", real=True)
   ...     squared = x**2
   >>> problem = inspect_model(Square, "analysis")
   >>> family = problem.solutions()
   >>> x = problem.symbols.x
   >>> response = family.expression(problem.symbols.squared)
   >>> diff(response, x) == 2*x
   True
   >>> calculate = lambdify(x, response, modules="math")
   >>> calculate(3)
   9

For a finite vector, pass its ``elements`` tuple to ``lambdify``:

.. code-block:: pycon

   >>> from sympy import Symbol
   >>> gain = Symbol("gain", real=True)
   >>> output = Vector(1, 2, 3) * gain
   >>> calculate_vector = lambdify(gain, output.elements, modules="math")
   >>> calculate_vector(2)
   (2, 4, 6)

Conditions and coordinate domains remain on the family. A numerical function
evaluates its formula without checking those conditions.

Return an answer for validation
-------------------------------

A native solver may answer just one equation. Add its candidate back to the
family to check the complete model, including domains and other observations:

.. code-block:: pycon

   >>> from sympy import Eq, solve
   >>> roots = solve(Eq(response, 4), x)
   >>> roots
   [-2, 2]
   >>> positive = family.observe(squared=4).where(x > 0)
   >>> positive.where(Eq(x, roots[0])).is_empty
   True
   >>> chosen = positive.where(Eq(x, roots[1])).one()
   >>> isinstance(chosen, Square), chosen.x
   (True, 2)

Use constructor arguments or ``family.observe`` for new scalar or vector
observations. Use ``where`` for symbolic conditions, and ``one`` or ``cases``
to materialize checked model states. See :doc:`solutions`.

Preserve the original mathematical question
-------------------------------------------

Imported operators remain unevaluated until model observations are available.
For example, this limit is zero when ``a=0`` and one when ``a=1``:

.. code-block:: pycon

   >>> from sympy import Limit, oo
   >>> class LimitReading(Modello):
   ...     a = InstanceDummy("a", real=True)
   ...     value = InstanceDummy("value", real=True)
   >>> t = Coordinate("t", real=True)
   >>> source = Limit(LimitReading.a*t / (LimitReading.a*t + 1), t, oo)
   >>> LimitReading("zero", a=0, value=source).value
   0
   >>> LimitReading("one", a=1, value=source).value
   1

Modello retains explicit denominator exclusions through its own rebinding and
evaluation. It cannot recover a pole or exceptional parameter case already
removed by SymPy before the expression reaches the model. Keep such domain
requirements explicit, and avoid eagerly evaluating a parameterized limit
whose exceptional cases matter. Unsupported calculations remain symbolic or
raise a clear error rather than validating an unestablished answer.

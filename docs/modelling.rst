Choose a modelling approach
===========================

Start with the relationship you want to express. These approaches compose;
choose the notation and scope that fit your question. These runnable examples
show their imports; the linked guides develop each approach.

Relate quantities or reuse a component
--------------------------------------

.. math::

   c^2 = a^2 + b^2

Declare lengths and their relationship, then supply the known measurements:

.. code-block:: pycon

   >>> from sympy import sqrt
   >>> from modello import InstanceDummy, Modello, component
   >>> class Triangle(Modello):
   ...     a = InstanceDummy("a", positive=True)
   ...     b = InstanceDummy("b", positive=True)
   ...     c = sqrt(a**2 + b**2)
   >>> Triangle("brace", a=3, b=4).c
   5

Reuse that relationship in typed components, then join their shared quantity:

.. code-block:: pycon

   >>> class LinkedBraces(Modello):
   ...     first = component(Triangle)
   ...     second = component(Triangle)
   ...
   ...     length = first.c
   ...     length = second.c
   >>> LinkedBraces("shared", first=Triangle("first", a=3, b=4),
   ...              second=Triangle("second", b=3)).second.a
   4

The repeated assignments mean ``length = first.c`` **AND** ``length = second.c``;
both components retain their own equations. See :ref:`joining-components`.

See :doc:`getting-started` for inverse questions and :doc:`nesting` for
connecting quantities across components.

Differentiate or integrate a formula
------------------------------------

.. math::

   \frac{\partial f}{\partial x},\qquad
   A = \int_0^{\tau} r t\,dt

Use native SymPy operators such as ``Derivative(f, x)`` and ``Integral``.
``Coordinate`` names a local continuous variable; ``InstanceDummy`` names a
quantity to observe or infer:

.. code-block:: pycon

   >>> from sympy import Derivative, Integral
   >>> from modello import Coordinate, InstanceDummy
   >>> t = Coordinate("t", real=True)
   >>> rate = InstanceDummy("rate", real=True)
   >>> duration = InstanceDummy("duration", positive=True)
   >>> area = Integral(rate * t, (t, 0, duration))
   >>> area.subs({rate: 2, duration: 3}).doit()
   9
   >>> Derivative(area, rate).doit() == duration**2 / 2
   True

A finite integration interval does not imply a finite sample vector.
See :ref:`symbolic-calculus` for using these formulas inside a model.

When a calculation needs mathematical shape or regularity guarantees, use the
opt-in :doc:`research workflow <contrib/research>`. It records explicit and
operation-derived requirements, parameter conditions, and proof evidence
alongside independent research cases.

Work with finite data and indexed formulas
------------------------------------------

.. math::

   \vec{x} = (x_0,\ldots,x_{N-1}),\qquad
   y_i = a x_i + b

Use ``Vector`` for finite scalar data and ``vector`` for indexed predictions.
``Index`` is a local integer coordinate; comprehension bounds are inclusive:

.. code-block:: pycon

   >>> from sympy import Product, Sum
   >>> from modello import Index, Vector, vector
   >>> i = Index("i")
   >>> x = Vector(1, 2, 3)
   >>> N = x.shape[0]
   >>> y = vector(2*x[i] + 1, (i, 0, N-1))
   >>> y.doit()
   [3, 5, 7]

Compact operations and native SymPy notation give the same finite reductions
:math:`\sum_i x_i`, :math:`\prod_i x_i`, and :math:`\sum_i x_i y_i`:

.. code-block:: pycon

   >>> x.sum().doit(), Sum(x[i], (i, 0, N-1)).doit()
   (6, 6)
   >>> x.product().doit(), Product(x[i], (i, 0, N-1)).doit()
   (6, 6)
   >>> x.dot(y).doit(), Sum(x[i]*y[i], (i, 0, N-1)).doit()
   (34, 34)

For your own pointwise operation :math:`f(x_i)`, use ``applyfunc`` with a
Python function or SymPy ``Lambda``. It uses the same name as SymPy arrays:

.. code-block:: pycon

   >>> from sympy import Array, Lambda, Symbol
   >>> z = Symbol("z")
   >>> operation = Lambda(z, z**2 + 1)
   >>> x.applyfunc(operation).doit(), Array(x).applyfunc(operation)
   ([2, 5, 10], [2, 5, 10])
   >>> x.applyfunc(lambda value: value**2 + 1).sum().doit()
   17
   >>> Sum(operation(x[i]), (i, 0, N-1)).doit()
   17

Use an indexed comprehension when the operation needs positions or several
vectors. See :ref:`custom-vector-operations` for composition and
:doc:`interoperability` for exchanging native containers, formulas, and solver
answers.

See :doc:`vectors` for partial data, domains, and fitting with these formulas.
For a collection of complete models, use :doc:`model_vector <model-vectors>`.
For an infinite bound, such as :math:`\sum_{i=0}^{\infty} q^i`, use native
``Sum`` directly:

.. code-block:: pycon

   >>> from sympy import Rational, Sum, oo
   >>> q = Rational(1, 2)
   >>> Sum(q**i, (i, 0, oo)).doit()
   2

A ``Vector`` always has finite size. Infinite-bound convergence and evaluation
follow SymPy's supported cases and conditions.

Express relative evolution
--------------------------

.. math::

   T_n = aT_{n-1} + b,\qquad T_0 = u

Two representations describe this law. Both examples below calculate
:math:`a=2, b=1, u=1`.

Query a scalar law without choosing a horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: pycon

   >>> from sympy import symbols
   >>> from modello import Index, Recurrence, Series
   >>> a, b, opening = symbols("a b opening", real=True)
   >>> n = Index("n")
   >>> T = Series("T", n, real=True)
   >>> scalar_law = Recurrence(T[n], a*T[n-1] + b, initial={0: opening})
   >>> bound = scalar_law.subs({a: 2, b: 1, opening: 1})
   >>> bound.at(3).doit(), bound.window(0, 3).doit()
   (15, [1, 3, 7, 15])

The rule applies for :math:`n\ge 1`. Use ``scalar_law.at(k)`` for a position or
``scalar_law.window(0, N)`` for a finite view. No infinite vector is materialized;
symbolic-position queries use verified supported closed forms.
See :doc:`series`.

Solve a finite window of model states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: pycon

   >>> from modello import Index, InstanceDummy, Modello, model_recurrence, model_vector
   >>> class State(Modello):
   ...     value = InstanceDummy("value", real=True)
   >>> class StateWindow(Modello):
   ...     N = InstanceDummy("N", integer=True, nonnegative=True)
   ...     a = InstanceDummy("a", real=True)
   ...     b = InstanceDummy("b", real=True)
   ...     n = Index("n")
   ...     states = model_vector(State, size=N + 1)
   ...     state_law = model_recurrence(states[n], value=a*states[n-1].value + b)
   ...     opening = states[0].value
   ...     closing = states[-1].value
   ...     levels = states.project(lambda s: s.value)
   >>> case = StateWindow("forward", N=3, a=2, b=1, opening=1)
   >>> case.closing, case.levels
   (15, [1, 3, 7, 15])
   >>> StateWindow("inverse", N=3, a=2, b=1, closing=15).opening
   1
   >>> case.states.project(lambda state: state.value**2).sum().doit()
   284

The equations apply for :math:`n\in\{1,\ldots,N\}`. Observing ``opening``
gives the same finite initial-value path as the scalar law. Observing ``closing``
instead asks an inverse question.

Use this form when each state has related quantities, or the equations need
implicit updates, forward references, or terminal boundaries. It solves the
whole declared window jointly. ``project`` can combine a member's fields into
any scalar formula, ready for reductions and ``applyfunc``. See
:doc:`model-vectors` and the
:ref:`comparison of sequence semantics <sequence-representations>`.

In every form, observations choose the question: calculate results, infer
missing inputs, or retain formulas. Use :doc:`solutions` to keep alternatives
and obtain SymPy expressions with ``family.expression(...)``, and
:doc:`debugging` to check conflicting observations.

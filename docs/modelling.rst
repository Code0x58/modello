Choose a modelling approach
===========================

Start with the relationship you want to express. These approaches compose;
choose the notation and scope that fit your question. The snippets are
declaration fragments; follow the links for complete runnable models.

Relate quantities or reuse a component
--------------------------------------

.. math::

   c^2 = a^2 + b^2

Declare lengths with ``InstanceDummy(..., positive=True)`` and write
``c = sqrt(a**2 + b**2)``. Supply two lengths to infer the third:
:doc:`start modelling <getting-started>`.

If the triangle already has a model, use ``base = component(Triangle)`` and
refer to ``base.c``. It retains the same relationship within a reusable part:
:doc:`component models <nesting>`.

Differentiate or integrate a formula
------------------------------------

.. math::

   \frac{\partial f}{\partial x},\qquad
   A = \int_0^{\tau} r t\,dt

Use native SymPy operators such as ``Derivative(f, x)`` and ``Integral``.
``Coordinate`` names a local continuous variable; ``InstanceDummy`` names a
quantity to observe or infer. For the integral:

.. code-block:: python

   t = Coordinate("t", real=True)
   area = Integral(rate * t, (t, 0, duration))

Here ``rate`` and ``duration`` are model quantities. A finite integration
interval does not imply a finite sample vector. See :ref:`symbolic-calculus`.

Work with finite data and indexed formulas
------------------------------------------

.. math::

   \vec{x} = (x_0,\ldots,x_{N-1}),\qquad
   y_i = a x_i + b

Use ``x = Vector("x", size=N, real=True)`` for scalar observations and
``vector(a*x[i] + b, (i, 0, N-1))`` for their predictions. ``i = Index("i")``
is a local integer coordinate. Observation lengths can determine ``N``.

Compact operations and native SymPy notation express the same finite reductions:

.. list-table::
   :header-rows: 1

   * - Relationship
     - Vector notation
     - SymPy notation
   * - :math:`\sum_{i=0}^{N-1} x_i`
     - ``x.sum()``
     - ``Sum(x[i], (i, 0, N-1))``
   * - :math:`\prod_{i=0}^{N-1} x_i`
     - ``x.product()``
     - ``Product(x[i], (i, 0, N-1))``
   * - :math:`\sum_{i=0}^{N-1} x_i y_i`
     - ``x.dot(y)``
     - ``Sum(x[i]*y[i], (i, 0, N-1))``

See :doc:`vectors` for partial data, domains, and fitting with these formulas.
For a collection of complete models, use :doc:`model_vector <model-vectors>`.
For an infinite bound, such as :math:`\sum_{i=0}^{\infty} q^i`, use native
``Sum(q**i, (i, 0, oo))`` directly; a ``Vector`` always has finite size.
Convergence and evaluation follow SymPy's supported cases and conditions.

Express relative evolution
--------------------------

.. math::

   T_n = aT_{n-1} + b,\qquad T_0 = u

Two representations describe this law. In these fragments, ``a``, ``b``, and
``opening`` are model quantities, and ``n = Index("n")``.

Query a scalar law without choosing a horizon
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   T = Series("T", n, real=True)
   scalar_law = Recurrence(T[n], a*T[n-1] + b, initial={0: opening})

The rule applies for :math:`n\ge 1`. Use ``scalar_law.at(k)`` for a position or
``scalar_law.window(0, N)`` for a finite view. No infinite vector is materialized;
symbolic-position queries use verified supported closed forms.
See :doc:`series`.

Solve a finite window of model states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   states = model_vector(State, size=N + 1)
   state_law = model_recurrence(states[n], value=a*states[n-1].value + b)
   opening = states[0].value

Here ``State`` is a model with a real scalar ``value`` field, and ``N`` is a
nonnegative integer size quantity. The equations apply for
:math:`n\in\{1,\ldots,N\}`. Observe ``opening`` for the same initial-value
problem; the values in ``states.project(lambda s: s.value)`` then describe
the same finite path as ``scalar_law.window(0, N)``.

Use this form when each state has related quantities, or the equations need
implicit updates, forward references, or terminal boundaries. It solves the
whole declared window jointly. See :doc:`model-vectors` and the
:ref:`comparison of sequence semantics <sequence-representations>`.

In every form, observations choose the question: calculate results, infer
missing inputs, or retain formulas. Use :doc:`solutions` to keep alternatives
and add conditions, and :doc:`debugging` to check conflicting observations.

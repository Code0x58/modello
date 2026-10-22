Finite paths and time-series formulas
=====================================

An ordered vector lets a model retain complete formulas across observation or
event boundaries. The reusable definitions in ``examples.timeseries`` keep
data acquisition and presentation separate. They demonstrate state transitions,
compounding, complete trailing windows, and continuous exposure under a stated
interpolation rule. See :doc:`../../vectors` for the underlying API.

From one fill to a path
-----------------------

The portfolio walkthrough composes a holding's ``before``, ``fill``, and
``after``. A finite inventory path extends that idea to several signed fills:

.. math::

   q_k &= q_0 + \sum_{j=0}^{k-1} f_j,
       \qquad k\in\{0,\ldots,n\}, \\
   q_{\mathrm{after}} &= q_0 + \sum_{i=0}^{n-1} f_i, \\
   C_{\mathrm{after}} &= C_0 - \sum_{i=0}^{n-1} f_i p_i.

An empty prefix sum gives the opening inventory. Buys are positive fills and
sells negative; prices are positive. Fees, income, and external cash flows are
excluded from this small settlement model. Signed inventory and cash allow a
researcher to state additional account policies separately.

.. code-block:: pycon

   >>> from examples.timeseries import HoldingState, inventory_path
   >>> Inventory = inventory_path(4)
   >>> opening = HoldingState("opening", quantity=5, cash=1000)
   >>> case = Inventory(
   ...     "reconcile", before=opening,
   ...     fills=[2, -1, None, -2], prices=[100, 110, 105, 120], cash=835,
   ... )
   >>> case.fills
   [2, -1, 3, -2]
   >>> case.quantities
   [5, 7, 6, 9, 7]
   >>> case.after.quantity, case.after.cash
   (7, 835)
   >>> opening.quantity, opening.cash
   (5, 1000)

The same equations can infer an execution price, an opening state, or a missing
fill from suitable observations. They do not silently select a solution when
several prices and quantities remain coupled.

``inventory_path(n)`` chooses a fixed finite shape and reuses its model class.
The result retains typed opening and closing components and immutable vectors.
An instance remains a holding state, so it can be composed with another model.

Compounding
------------

``return_path(n)`` keeps the return vector, positive gross factors, and wealth
at every boundary:

.. math::

   g_i = 1+r_i,\quad g_i>0,\qquad
   W_k = W_0\prod_{j=0}^{k-1}g_j,\qquad
   R = \prod_{i=0}^{n-1}g_i-1.

.. code-block:: pycon

   >>> from sympy import Rational
   >>> from examples.timeseries import return_path
   >>> Returns = return_path(3)
   >>> case = Returns("returns", before=100, returns=[Rational(1, 10), Rational(-1, 20), Rational(1, 5)])
   >>> case.wealth
   [100, 110, 209/2, 627/5]
   >>> case.after, case.total_return
   (627/5, 127/500)
   >>> missing = Returns("inverse", before=100, returns=[Rational(1, 10), None, Rational(1, 5)], after=case.after)
   >>> missing.returns == case.returns
   True

The positive-factor domain excludes total loss and negative capital in this
example. No observation frequency or annualization convention is assumed.

Rolling windows
----------------

``rolling_window(size, width)`` uses complete, equally weighted trailing
windows. It does not pad boundaries or fill missing observations. Output zero
corresponds to the first complete window, ending at input ``width-1``:

.. math::

   m_{i-w+1}=\frac1w\sum_{j=i-w+1}^{i}x_j,
       \qquad i\in\{w-1,\ldots,n-1\}.

.. code-block:: pycon

   >>> from examples.timeseries import rolling_window
   >>> Window = rolling_window(5, 3)
   >>> Window("forward", samples=[1, 2, 3, 4, 5]).means
   [2, 3, 4]
   >>> observed = Window("inverse", samples=[1, 2, None, 4, 5], means=[None, 3, None])
   >>> observed.samples
   [1, 2, 3, 4, 5]

A missing observation remains an unknown quantity. A supplied window statistic
can determine it; no imputation policy is applied automatically.

Continuous exposure
--------------------

``LinearExposure`` explicitly assumes linear interpolation between two levels:

.. math::

   A=\int_0^T\left(a+\frac{b-a}{T}t\right)\,dt
       =\frac{a+b}{2}T,\qquad T>0.

.. code-block:: pycon

   >>> from examples.timeseries import LinearExposure
   >>> LinearExposure("area", opening=2, closing=4, duration=3).exposure
   9
   >>> LinearExposure("inverse", opening=2, exposure=9, duration=3).closing
   4

An integral of an explicitly stated function differs from a sum of samples.
Choose durations, interpolation, and units as part of the model; the vector
layer does not supply them.

Evaluation and exploration
---------------------------

These models extend the example numerical adapter ``SymbolicModel``. A compiled
view evaluates already resolved scalar formulas and reconstructs the same model
type, including vectors and components. Bind individual vector coordinates with
``Known(case.fills[i], value)``. The compiled view does not invoke the symbolic
solver for each evaluation.

The adapter's outer observation axis remains separate from a vector inside one
model case. Timestamp alignment, instrument identity, and causal sampling stay
with typed data adapters such as those in the portfolio example.

Run the console walkthrough with ``python -m examples.timeseries``. For the
interactive view, run ``python -m examples.gui_timeseries`` with the GUI extra
and open http://127.0.0.1:5059. Select ``--example returns``, ``rolling``, or
``exposure`` for the other models. Each example includes both forward and inverse
guided questions. The :doc:`GUI guide <../../gui>` describes editing and validation.

9. Derive execution from a continuous model
===========================================

The earlier chapters reused scalar equations in several solving directions.
Now we will reuse a mathematical *functional*: define execution costs in
continuous time, derive its differential equation, integrate it onto a discrete
grid, and generate the equations for a schedule. The
:doc:`following study <execution_calibration>` will use the same definition for
calibration and liquidity reporting.

Take the ten-share MSFT sell intention from chapter 6. The earlier execution
benchmark describes its actual fills. Here we ask a different question: how
would a model schedule those shares over four hours? We will use separate
synthetic liquidity parameters, not infer them from the three account fills.

Define the continuous economics
-------------------------------

Let :math:`q(t)` be shares still to sell and :math:`v(t)=-q'(t)` the selling
speed. A quadratic participation cost and a CARA price-risk penalty give

.. math::

   J[q] = \int_0^T \left[
       \frac{\eta v(t)^2}{V(t)}
       + \frac{\gamma}{2}\sigma^2 q(t)^2
   \right]dt,
   \qquad q(0)=Q,\quad q(T)=0.

This is a quadratic specialisation of the deterministic execution functional in
`Guéant's optimal execution framework <https://arxiv.org/html/1210.6372>`_.
His general participation function permits nonlinear costs; our initial choice
is :math:`L(\rho)=\eta\rho^2`, with no permanent impact or spread. The risk
term uses deterministic inventory and an arithmetic Brownian reference price.
Under CARA utility, Gaussian cash risk contributes half its variance times
absolute risk aversion to the certainty-equivalent cost.

Define the rates as an ordinary Modello model. These are the economic equations
from which the later views will be derived:

.. code-block:: python

   from modello import InstanceDummy
   from examples.symbolic import SymbolicModel

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionDensity
   :end-before:     def lagrangian
   :language: python

The units are explicit:

.. list-table::
   :header-rows: 1

   * - Quantity
     - Unit
   * - ``inventory``, ``speed``, ``volume``
     - shares, shares/hour, market shares/hour
   * - ``eta``
     - USD/share
   * - ``sigma``
     - USD/share/sqrt(hour), an absolute price volatility
   * - ``gamma``
     - 1/USD, absolute risk aversion
   * - ``impact``, ``variance``, ``risk``
     - USD/hour, USD²/hour, USD/hour

Reuse the cash utility from chapter 8 to supply the preference parameter:

>>> from sympy import Function, Rational, Symbol, diff, simplify, solve
>>> from examples.portfolio.optimal_execution import ExecutionDensity
>>> from examples.portfolio.utility import CashUtility
>>> from examples.model_plan import Known, ModelPlan
>>> utility = CashUtility.cara(0.001)
>>> gamma = utility.expressions["absolute_risk_aversion"]
>>> model = ExecutionDensity("execution", eta=2, volume=100, sigma=3, gamma=gamma)
>>> rates = ModelPlan(model)
>>> point = rates.evaluate(Known(model.inventory, 10), Known(model.speed, 4))
>>> round(float(point.impact), 4), round(float(point.risk), 4)
(0.32, 0.45)

Expected impact is cash cost. ``risk`` is a preference penalty, not an additional
fill fee or account posting. Here ``gamma`` is the CARA coefficient; it is neither
the relative-aversion parameter of CRRA utility nor a permanent-impact slope.

Lift the scalar density to a variational problem
------------------------------------------------

The model's inventory and speed are still unknown independent symbols. Replace
them with :math:`q(t)` and :math:`-q'(t)`:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionDensity.lagrangian
   :dedent: 4
   :language: python

Then let SymPy differentiate the functional using
`euler_equations <https://docs.sympy.org/latest/modules/calculus/index.html#sympy.calculus.euler.euler_equations>`_:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionDensity.euler_equation
   :dedent: 4
   :language: python

>>> t = Symbol("t", real=True)
>>> q = Function("q")(t)
>>> equation = model.euler_equation(t, q)
>>> acceleration = solve(equation, diff(q, t, 2))[0]
>>> simplify(acceleration / q)
9/40

For constant coefficients the generated equation is
:math:`q''=\kappa^2q`, where :math:`\kappa^2=\gamma\sigma^2V/(2\eta)`.
``continuous_curve`` passes that equation and the two inventory boundary
conditions to ``dsolve``; it stores no separately coded trajectory formula:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionDensity.continuous_curve
   :dedent: 4
   :language: python

>>> curve = model.continuous_curve(10, 4).rhs
>>> simplify(curve.subs(t, 0)), simplify(curve.subs(t, 4))
(10, 0)
>>> simplify(equation.lhs.subs(q, curve).doit())
0

The familiar hyperbolic-sine liquidation curve is a result of these equations.
With zero risk aversion, asking the same solver produces uniform liquidation:

>>> neutral = ExecutionDensity("neutral", eta=2, volume=100, sigma=3, gamma=0)
>>> neutral.continuous_curve(10, 4)
Eq(q(t), 10 - 5*t/2)

Restrict the same functional to an explicit grid
------------------------------------------------

For a discrete system, choose a mesh and a path representation. On an interval
of width :math:`h`, interpolate linearly from :math:`a` to :math:`b` and hold
volume and coefficients constant. Substitute that path into the rates and let
SymPy integrate them:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionDensity.interval
   :dedent: 4
   :language: python

>>> model.interval(10, 6, 1)
IntervalCost(impact=8/25, variance=588, risk=147/500, objective=307/500)

The quadratic model consequently gives

.. math::

   I_i = \frac{\eta(a-b)^2}{V_i h_i},\qquad
   R_i = \frac{\gamma\sigma^2h_i}{6}(a^2+ab+b^2).

These expressions are outputs of integration, not additional implementation
formulas. Integrating inventory risk exactly retains the cross term ``a*b``;
simply charging risk at the opening mark would define a different discretisation.
The integral is exact for the chosen piecewise-linear path; restricting paths
to that mesh approximates the continuous optimisation problem.

Sum interval objectives, differentiate with respect to interior inventories,
and differentiate again for curvature:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionGrid.objective
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionGrid.gradient
   :dedent: 4
   :language: python

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionGrid.hessian
   :dedent: 4
   :language: python

>>> grid = model.discretize(10, (1, 1, 1, 1))
>>> len(grid.unknowns), grid.hessian.shape
(3, (3, 3))
>>> grid.hessian.is_positive_definite
True

Here the stationarity equations are linear, even though the objective is
quadratic. ``stationary`` extracts their coefficient matrix and solves it
exactly. It uses ``nsolve`` with an explicit initial guess for nonlinear laws:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionGrid.stationary
   :dedent: 4
   :language: python

>>> result = grid.stationary()
>>> tuple(round(float(value), 4) for value in result.inventory)
(10.0, 5.9741, 3.3448, 1.4974, 0.0)
>>> result.residuals
(0, 0, 0)
>>> all(float(quantity) >= 0 for quantity in result.trades), sum(result.trades)
(True, 10)

All shares are sold, quantities decrease, and positive curvature establishes
the unique minimum of this quadratic grid problem. ``stationary`` itself solves
unconstrained first-order conditions. It does not enforce participation limits
or nonnegative inventory; those checks and any constrained optimiser belong to
the scheduling layer. Coarse meshes or other laws can produce infeasible points.

Check what the discretisation changes
-------------------------------------

Compare the midpoint with the generated continuous solution on refined meshes:

>>> midpoint = float(curve.subs(t, 2))
>>> errors = []
>>> for count in (2, 4, 8):
...     solution = model.discretize(10, (Rational(4, count),) * count).stationary()
...     errors.append(abs(float(solution.inventory[count // 2]) - midpoint))
>>> errors[2] < errors[1] < errors[0]
True

Volume curves need no new scheduling equations. Keep volume unbound in the
density and supply one rate per interval. In the risk-neutral case, the optimal
trades follow available volume-time:

>>> neutral = ExecutionDensity("volume_curve", eta=2, sigma=3, gamma=0)
>>> liquid = neutral.discretize(12, (1, 2, 1), volumes=(100, 100, 200)).stationary()
>>> liquid.trades
(12/5, 24/5, 24/5)

Nothing in this study invents actual fills. Its schedule can inform order
intentions; chapter 6's independent ledger and execution assessment still
determine what happened. The :doc:`next study <execution_calibration>` turns the
same functional into a calibration problem and a liquidity-cost report.

10. Fit impact and report liquidity from the same model
=======================================================

The preceding study derived a schedule. We will now ask inverse and reporting
questions of the same functional: fit impact from trade buckets, infer a
preference under an optimal-policy assumption, and derive the sensitivity and
liquidity cost of a block. Finally, change the impact law and reuse the pipeline.

Build observations from independently recorded trades
-----------------------------------------------------

Execution price alone mixes temporary impact with spread, market movement,
permanent impact, and noise. A calibration bucket needs an independently
estimated unaffected reference, market volume, and duration. Fees stay separate.
Define that input and its fill adapter:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/calibration.py
   :pyobject: ImpactObservation
   :end-before:     def __post_init__
   :language: python

The constructor checks finite values and positive quantity, duration, and market
volume. Its adapter aggregates same-side fills for one account and holding:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/calibration.py
   :pyobject: ImpactObservation.from_fills
   :dedent: 4
   :language: python

For a controlled recovery experiment, create three synthetic sell buckets with
known unaffected price USD 100. Their different participation rates make the
cost curve observable. These are extra training observations, separate from
the account ledger and the earlier out-of-sample execution benchmarks:

>>> from examples.portfolio.sources import Fill, timestamp
>>> from examples.portfolio.optimal_execution import ExecutionDensity, ImpactObservation
>>> at = timestamp("2026-01-07T09:00:00Z")
>>> observations = tuple(
...     ImpactObservation.from_fills(
...         (Fill(f"training-{i}", "training", at, at, "msft", -quantity, price, 0),),
...         reference_price=100,
...         duration=1,
...         volume=64,
...     )
...     for i, (quantity, price) in enumerate(((8, 99.75), (16, 99.5), (4, 99.875)))
... )
>>> tuple((row.quantity, row.cost) for row in observations)
((8, 2.0), (16, 8.0), (4, 0.5))

The reference in real data requires its own estimation and identification
assumptions. This synthetic recovery exercise does not establish that observed
slippage is causal temporary impact.

Generate a fit from the integrated cost
---------------------------------------

For each bucket, integrate the density along its constant-speed liquidation
and subtract the observed cost. Fit only the impact component; the risk penalty
is not an observed cash expense:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionDensity.fit_impact
   :dedent: 4
   :language: python

The small calibration adapter forms the sum of squared residuals, differentiates
it with respect to the unknown coefficients, and solves the normal equations.
It checks rank before claiming separate identification. It supports residuals
affine in the fitted parameters, which includes the quadratic and later
two-coefficient impact laws:

.. raw:: html

   <details><summary>Build the least-squares adapter from symbolic residuals</summary>

.. literalinclude:: ../../../examples/portfolio/optimal_execution/calibration.py
   :pyobject: Calibration.fit
   :dedent: 4
   :language: python

.. raw:: html

   </details>

>>> unknown = ExecutionDensity("impact_fit")
>>> fit = unknown.fit_impact(observations)
>>> fit.estimates["eta"], fit.loss
(2, 0)

There is no independently implemented regression formula for
``quantity**2 / (volume * duration)``. That predictor came from integrating
``ExecutionDensity.impact``. Decimal observation values are converted to exact
rationals for this small algebraic demonstration; that does not make noisy
measurements exact. Nonzero residuals remain visible:

>>> noisy = unknown.fit_impact((observations[0], ImpactObservation(16, 1, 64, 8.1)))
>>> round(float(noisy.estimates["eta"]), 6), noisy.loss > 0
(2.023529, True)

Infer preferences only under a policy model
-------------------------------------------

Impact observations do not contain ``gamma``. Asking to fit it alongside ``eta``
therefore fails identification:

>>> unknown.fit_impact(observations, parameters=("eta", "gamma"))
Traceback (most recent call last):
...
ValueError: Observations do not identify these parameters separately

Instead, suppose a schedule was chosen by minimising this functional, with
known volatility, volume, time mesh, and impact coefficient. Its interior
inventories should satisfy the same first-order equations derived previously.
Substituting observed inventories gives residuals in the remaining preference
parameter:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionGrid.fit_policy
   :dedent: 4
   :language: python

Test recovery from the preceding study's exact policy, without passing its
risk-aversion coefficient to the inverse case:

>>> from sympy import Rational, Symbol, diff, simplify, solve
>>> forward = ExecutionDensity(
...     "forward", eta=fit.estimates["eta"], volume=100, sigma=3, gamma=Rational(1, 1000)
... )
>>> observed = forward.discretize(10, (1, 1, 1, 1)).stationary().inventory
>>> preference = ExecutionDensity(
...     "preference", eta=fit.estimates["eta"], volume=100, sigma=3
... )
>>> inverse_grid = preference.discretize(10, (1, 1, 1, 1))
>>> inferred = inverse_grid.fit_policy(observed)
>>> inferred.estimates["gamma"], inferred.loss
(1/1000, 0)

This is an *implied* CARA preference, conditional on optimal behaviour and this
model. Real schedules may reflect participation caps, urgency, alpha, discretion,
or different beliefs about volatility and liquidity. Those omitted effects
cannot be distinguished from preferences by this inverse problem alone.

There is also a structural scale ambiguity: multiplying ``eta`` and ``gamma``
by the same positive factor multiplies the objective without changing its
minimiser. Schedule observations identify their ratio, not both levels. With
both left unknown, the exact recovery example is rank deficient:

>>> both = ExecutionDensity("both", volume=100, sigma=3).discretize(10, (1, 1, 1, 1))
>>> both.fit_policy(observed, parameters=("eta", "gamma"))
Traceback (most recent call last):
...
ValueError: Observations do not identify these parameters separately

Fitting market impact first provides the missing scale. Production statistical
estimation also needs uncertainty, out-of-sample checks, appropriate error
weighting, and attention to conditioning. The adapter solves small symbolic
least-squares examples; it is not a general statistical or constrained solver.

Differentiate the decision and its value
----------------------------------------

How does greater risk aversion change the schedule? Differentiate the generated
stationarity system :math:`F(q,\gamma)=0` implicitly:

.. math::

   \frac{dq^*}{d\gamma}
   = -\left(\frac{\partial F}{\partial q}\right)^{-1}
       \frac{\partial F}{\partial\gamma}.

The implementation solves that Hessian system; it does not maintain a separate
sensitivity formula for a particular impact law:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: ExecutionSolution.sensitivity
   :dedent: 4
   :language: python

Use a two-hour, two-interval case to keep the symbolic result compact:

>>> symbolic = preference.discretize(10, (1, 1)).stationary()
>>> sensitivity = symbolic.sensitivity(preference.gamma)[0]
>>> simplify(sensitivity - diff(symbolic.interior[0], preference.gamma))
0
>>> sensitivity.subs(preference.gamma, Rational(1, 1000)) < 0
True

Greater aversion leaves fewer shares at the midpoint. This derivative is local
to an interior stationary solution; changing active constraints would require
the corresponding constrained equations.

For reporting, optimise a symbolic block size ``Q`` with the fitted coefficient
and specified preference. Its value is a liquidity deduction from an unaffected
mark, including the risk penalty:

>>> Q = Symbol("Q", positive=True)
>>> block = forward.discretize(Q, (1, 1, 1, 1)).stationary()
>>> value = block.costs.objective
>>> deduction_per_share = value / Q
>>> marginal_cost = diff(value, Q)
>>> simplify(marginal_cost - 2 * deduction_per_share)
0

The equality follows from this model's quadratic homogeneity. Average cost per
share and marginal cost differ; both were derived from the optimised value.
The same expression can answer a budget question:

>>> capacity = solve(value - 10, Q)[0]
>>> round(float(capacity), 4)
31.6231

This is the block fitting a USD 10 certainty-equivalent cost budget under the
stated four-hour model, not a guaranteed execution capacity. A book-value report
could show unaffected NAV, expected execution cash cost, risk penalty, and this
liquidity-adjusted value separately. Only realised cash costs enter replay.

Change the economics without rewriting its applications
-------------------------------------------------------

Make high participation more expensive by adding a quartic term to the impact
density. This changes one economic definition; the risk model is inherited:

.. literalinclude:: ../../../examples/portfolio/optimal_execution/model.py
   :pyobject: CrowdedImpact
   :language: python

This is a smooth convex illustration, not an empirical claim about a particular
market's impact law. SymPy now produces a nonlinear Euler equation, a quartic
discrete objective, and its cubic stationarity equations. The same solver
adapter can evaluate a stationary point with an explicit starting guess:

>>> from examples.portfolio.optimal_execution import CrowdedImpact
>>> from examples.compute.named_map import NamedMap
>>> crowded = CrowdedImpact("crowded", eta=2, volume=100, sigma=3, gamma=Rational(1, 1000))
>>> grid = crowded.discretize(10, (1, 1))
>>> solution = grid.stationary(at=NamedMap({"beta": 200.0}), guess=(4,))
>>> abs(float(solution.residuals[0])) < 1e-10
True
>>> float(grid.hessian.subs(solution.substitutions)[0, 0]) > 0
True
>>> all(float(quantity) >= 0 for quantity in solution.trades)
True

Keeping ``beta`` symbolic until the solve point also preserves the original
equations for implicit differentiation:

>>> float(solution.sensitivity(crowded.beta)[0]) > 0
True

Increasing the participation penalty leaves more shares at the midpoint. The
derivative comes from the generated nonlinear Hessian and parameter derivative;
it does not require a closed-form schedule.

The predictor for ``beta`` is also derived by interval integration. New
observations can fit both coefficients using the same calibration adapter:

>>> extended = CrowdedImpact("extended")
>>> fit = extended.fit_impact(
...     (ImpactObservation(10, 1, 100, 2.02), ImpactObservation(20, 1, 100, 8.32)),
...     parameters=("eta", "beta"),
... )
>>> fit.estimates["eta"], fit.estimates["beta"], fit.loss
(2, 2, 0)

No ODE, bucket-cost formula, regression predictor, stationarity equation,
Hessian, or implicit-sensitivity expression was reimplemented for this law.
Symbolic differentiation and integration propagate the changed definition.
General nonlinear boundary-value problems need appropriate numerical solvers;
``dsolve`` is not guaranteed to produce a closed form for a new density.

.. mermaid::
   :caption: Applications derived from one mathematical definition
   :alt: The continuous density supplies a variational equation and integrated interval costs. Interval costs supply both calibration residuals and grid stationarity. Stationarity supplies policy calibration, implicit sensitivities, and optimised liquidity values.

   flowchart TD
       density[Modello cost density] --> variation[Euler-Lagrange equation]
       variation --> curve[Continuous boundary-value solution]
       density --> integral[Integrated interval costs]
       integral --> costs[Impact calibration residuals]
       integral --> stationarity[Grid gradient and Hessian]
       stationarity --> policy[Implied preference calibration]
       stationarity --> sensitivity[Implicit schedule sensitivities]
       stationarity --> value[Optimised value and marginal liquidity costs]

Run the two studies together with the installed examples:

.. code-block:: console

   $ python -m examples.portfolio.optimal_execution

Their definitions and tests live under
``examples/portfolio/optimal_execution/`` and
``tests/examples/portfolio/optimal_execution/``. Scheduling, inverse policy
assumptions, and liquidity reporting remain a separate example layer above
the unchanged scalar Modello solver.

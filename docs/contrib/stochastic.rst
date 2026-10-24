Stochastic processes (contrib)
==============================

Import this domain extension explicitly from ``modello.contrib.stochastic``.
It builds on ordinary Modello declarations and the same flat equation solver.

For one standard Wiener driver, ``ItoProcess(X, a, b, t)`` declares
``dX = a*dt + b*dW``. For a sufficiently smooth deterministic function ``f``,
its transformed drift is ``f_t + a*f_x + b**2*f_xx/2`` and diffusion is ``b*f_x``:

.. code-block:: pycon

   >>> from sympy import log
   >>> from modello import Coordinate, InstanceDummy
   >>> from modello.contrib.stochastic import ItoProcess
   >>> t = Coordinate("t", real=True)
   >>> S = Coordinate("S", positive=True)
   >>> mu = InstanceDummy("mu", real=True)
   >>> sigma = InstanceDummy("sigma", positive=True)
   >>> process = ItoProcess(S, mu * S, sigma * S, t)
   >>> process.drift(log(S)).doit() == mu - sigma**2 / 2
   True
   >>> process.diffusion(log(S)).doit() == sigma
   True

The positive state makes the logarithm's domain explicit. State and time remain
independent during differentiation; named formulas expand first. Smoothness and
adaptedness are hypotheses of the model. A one-state transform rejects another
unmodelled process because its quadratic covariation would need a declaration.

``generator(f)`` returns the spatial drift without ``f_t``.
``quadratic_variation_rate(f)`` returns the coefficient of ``dt`` in the
transformed quadratic variation.

Supported process queries
-------------------------

``exact(initial, elapsed, increment)`` gives arithmetic or geometric Brownian
values conditional on a supplied Wiener increment. ``mean`` and ``variance``
give conditional moments, also supporting affine additive diffusions such as
mean reversion. A realised increment of zero generally differs from the mean.
Elapsed time must be nonnegative; zero time requires a zero increment. Initial
values and exact realised endpoints must satisfy the state domain.
Zero elapsed time returns the initial value or zero variance even when no
nonzero-time closed form is supported.

The :doc:`diffusion demo <../examples/gallery/index>` uses those formulas for inverse
parameter inference:

.. code-block:: pycon

   >>> from sympy import Rational
   >>> from examples.gallery.extensions import GeometricDiffusion
   >>> fitted = GeometricDiffusion(
   ...     "coefficients", opening=100, duration=1, wiener_increment=0,
   ...     log_drift=Rational(2, 25), log_diffusion=Rational(1, 5),
   ... )
   >>> fitted.growth, fitted.volatility
   (1/10, 1/5)

An exact mean-reverting path requires a weighted stochastic integral; a single
increment is insufficient. Zero diffusion retains deterministic exact paths.
Multiple drivers, correlations, arbitrary stochastic integrals, and general
SDE boundary problems are outside the supported solvers.
Known process dependencies remain attached to returned laws: observing another
state at a point does not make it a constant coefficient for exact moments.
Declare related processes together when their dynamics matter. These checks do
not prove that an entire path stays in its declared state domain.
The :doc:`solver boundaries <../generality>` describe further mathematical and
computational boundaries and the independent regression scenarios.

``problem.templates`` exposes scoped, read-only laws. The explorer retains
relative subscripts, histories, differentials, and conditional moment notation;
``problem.latex`` exports the same formulas.


API
---

.. currentmodule:: modello.contrib.stochastic

.. autoclass:: ItoProcess(state, drift, diffusion, time, driver=None)
   :members: drift, diffusion, generator, quadratic_variation_rate, exact, mean, variance, equations

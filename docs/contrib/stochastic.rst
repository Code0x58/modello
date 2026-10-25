Stochastic processes
=====================

A price starts at 100 and follows a model with growth rate 0.10 and volatility
0.20. Use it to calculate the expected price after one period and the growth of
its logarithm. The :doc:`gallery's price example <../examples/gallery/index>`
uses ``GeometricDiffusion`` to calculate both, or infer the coefficients from
observations of the log-price drift and diffusion.

The model declares :math:`dS_t=\mu S_t\,dt+\sigma S_t\,dW_t`. Its
``ItoProcess`` law applies Itô's formula to obtain

.. math::

   d\log S_t = (\mu-\tfrac12\sigma^2)\,dt + \sigma\,dW_t.

Reuse that example to calculate the log-price coefficients, a realised closing
price, and the conditional mean:

.. code-block:: pycon

   >>> from sympy import Rational
   >>> from examples.gallery.extensions import GeometricDiffusion
   >>> price = GeometricDiffusion(
   ...     "price", opening=100, growth=Rational(1, 10), volatility=Rational(1, 5),
   ...     duration=1, wiener_increment=0,
   ... )
   >>> price.log_drift, price.log_diffusion
   (2/25, 1/5)
   >>> price.closing, price.mean
   (100*exp(2/25), 100*exp(1/10))

The log-price drift is 0.08. A realised Wiener increment of zero gives the
closing price :math:`100e^{0.08}`, while the mean over possible increments is
:math:`100e^{0.10}`.

Declare your own process
------------------------

Import ``ItoProcess`` from ``modello.contrib.stochastic``. For one standard
Wiener driver, ``ItoProcess(X, a, b, t)`` declares ``dX = a*dt + b*dW``.
For a sufficiently smooth deterministic function ``f``, its transformed drift
is ``f_t + a*f_x + b**2*f_xx/2`` and diffusion is ``b*f_x``.

The example declares a positive state, making the logarithm's domain explicit.
State and time remain independent during differentiation; named formulas expand
first. Smoothness and adaptedness are hypotheses of the model. A one-state
transform rejects another
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

To recover the same coefficients from the log-process drift and diffusion,
supply those observations to the same example:

.. code-block:: pycon

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

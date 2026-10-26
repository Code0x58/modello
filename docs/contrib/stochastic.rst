Stochastic processes
=====================

The :doc:`gallery's price model <../examples/gallery/index>` calculates realised
prices, conditional moments, and log-price growth.

The model declares :math:`dS_t=\mu S_t\,dt+\sigma S_t\,dW_t`. Its
``ItoProcess`` law applies Itô's formula to obtain

.. math::

   d\log S_t = (\mu-\tfrac12\sigma^2)\,dt + \sigma\,dW_t.

Start at 100 with growth rate 0.10 and volatility 0.20:

.. doctest::

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

The log-price drift is 0.08. A zero Wiener increment gives the realised closing
price :math:`100e^{0.08}`; the mean over possible increments is
:math:`100e^{0.10}`.

Declare your own process
------------------------

Import ``ItoProcess`` from ``modello.contrib.stochastic``. For one standard
Wiener driver, ``ItoProcess(X, a, b, t)`` declares ``dX = a*dt + b*dW``.
For a sufficiently smooth deterministic function ``f``, its transformed drift
is ``f_t + a*f_x + b**2*f_xx/2`` and diffusion is ``b*f_x``.

State and time remain independent during differentiation; named formulas expand
first. The state domain must permit the transform, as positivity permits the
logarithm above. Smoothness and adaptedness are model hypotheses. A transform
rejects an unmodelled process whose quadratic covariation is undeclared.

``generator(f)`` returns the spatial drift without ``f_t``.
``quadratic_variation_rate(f)`` returns the coefficient of ``dt`` in the
transformed quadratic variation.

Supported process queries
-------------------------

``exact(initial, elapsed, increment)`` gives arithmetic or geometric Brownian
values conditional on a supplied Wiener increment. ``mean`` and ``variance``
give conditional moments, also supporting affine additive diffusions such as
mean reversion. Elapsed time must be nonnegative; zero time requires a zero
increment and returns the initial value or zero variance, even without a
nonzero-time closed form. Initial values and exact endpoints must satisfy the
state domain.

Recover the coefficients by supplying the log-price drift and diffusion:

.. doctest::

   >>> fitted = GeometricDiffusion(
   ...     "coefficients", opening=100, duration=1, wiener_increment=0,
   ...     log_drift=Rational(2, 25), log_diffusion=Rational(1, 5),
   ... )
   >>> fitted.growth, fitted.volatility
   (1/10, 1/5)

An exact mean-reverting path needs a weighted stochastic integral, not a single
increment. Zero diffusion supports deterministic exact paths. Multiple drivers,
correlations, arbitrary stochastic integrals, and general SDE boundary problems
are unsupported. Declare related processes together: a point observation does
not make another process a constant coefficient. These checks do not establish
that an entire path stays in its state domain. See :doc:`../generality` for
solver boundaries.

``problem.templates`` exposes scoped, read-only laws. The explorer retains
relative subscripts, histories, differentials, and conditional moment notation;
``problem.latex`` exports the same formulas.


API
---

.. currentmodule:: modello.contrib.stochastic

.. autoclass:: ItoProcess(state, drift, diffusion, time, driver=None)
   :members: drift, diffusion, generator, quadratic_variation_rate, exact, mean, variance, equations

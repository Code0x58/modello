Relative series and process laws
================================

``Series`` names an indexed scalar family. ``Recurrence`` relates discrete
positions; ``ItoProcess`` describes continuous drift and diffusion. Their
queries are scalar formulas in an ordinary Modello model, so the same solver
can calculate results or infer parameters from observations.

One relative law, many queries
------------------------------

.. code-block:: pycon

   >>> from modello import Index, InstanceDummy, Modello, Recurrence, Series
   >>> class Growth(Modello):
   ...     n = Index("n", nonnegative=True)
   ...     T = Series("T", n, positive=True)
   ...     opening = InstanceDummy("opening", positive=True)
   ...     ratio = InstanceDummy("ratio", positive=True)
   ...     periods = InstanceDummy("periods", integer=True, nonnegative=True)
   ...     law = Recurrence(T[n], ratio * T[n - 1], initial={0: opening})
   ...     closing = T[periods]
   ...     samples = law.window(0, 4)
   >>> case = Growth("forward", opening=1, ratio=2, periods=10)
   >>> case.closing, case.samples
   (1024, [1, 2, 4, 8, 16])
   >>> Growth("inverse", closing=1024, ratio=2, periods=10).opening
   1
   >>> case.law.at(20).doit()
   1048576

The law has no fixed horizon. The query at position ten and the window from
zero through four are separate views. In model definitions and family queries,
``T[k]`` uses its declared recurrence; ``law.at(k)`` explicitly selects one law.
Use the populated instance's law for direct evaluation outside a model.

Series indices are mathematical positions: ``T[n-1]`` is the predecessor and
``T[-1]`` is position minus one. A series has no length or implicit last entry.
``law.window(first, last)`` creates an inclusive finite :doc:`vector <vectors>`
whose Python position zero represents ``T[first]``.

History and domains
-------------------

A recurrence needs contiguous history ending immediately before ``start``.
Its fixed positive lags must be covered by that history:

.. code-block:: pycon

   >>> n = Index("n")
   >>> F = Series("F", n, integer=True)
   >>> law = Recurrence(F[n], F[n - 1] + F[n - 2], initial={0: 0, 1: 1})
   >>> law.start, law.at(10).doit()
   (2, 55)

History may start at a negative or shifted position. Queries before it are
undefined. Series domains apply to intermediate states as well as exposed
outputs. For example, a positive terminal value cannot excuse a negative
predecessor in a positive series.

Finite causal queries support nonlinear laws. Verified closed forms support
constant first-order affine laws and suitable higher-order linear laws through
SymPy's recurrence solver. Zero and unit coefficients retain their special
cases. Unsupported symbolic queries remain unevaluated; exact solution families
require them to resolve.

Let observations choose the window
----------------------------------

The gallery's autoregression declares ``N`` once. State observations have
``N+1`` entries and innovations have ``N``:

.. literalinclude:: ../examples/gallery/models.py
   :start-at: class ObservedAutoregression
   :end-before: class AccountSequence
   :language: python

.. code-block:: pycon

   >>> from examples.gallery.models import ObservedAutoregression
   >>> case = ObservedAutoregression("states", retention=1, levels=[1, 3, 6, 10])
   >>> case.N, case.opening, case.innovations
   (3, 1, [2, 3, 4])

Either vector's length, or an explicit ``N``, determines the finite window.
``None`` entries still count toward its length. Observing states and innovations
constrains the same equation; contradictory observations are rejected. The GUI's
innovation-driven question releases the original state observations first.

History and observation length are separate. A recurrence of order ``p`` needs
``p`` history positions, but those quantities may be observed separately or
inferred. A vector of length at least ``p`` does not by itself identify unknown
coefficients or innovations. The innovation vector carries no distribution
unless the model declares one.

Continuous Itô laws
-------------------

For one standard Wiener driver, ``ItoProcess(X, a, b, t)`` declares
``dX = a*dt + b*dW``. For a sufficiently smooth deterministic function ``f``,
its transformed drift is ``f_t + a*f_x + b**2*f_xx/2`` and diffusion is ``b*f_x``:

.. code-block:: pycon

   >>> from sympy import log
   >>> from modello import Coordinate, ItoProcess
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
values must satisfy the state domain.

The :doc:`gallery <examples/gallery/index>` uses those formulas for inverse
parameter inference:

.. code-block:: pycon

   >>> from sympy import Rational
   >>> from examples.gallery.models import GeometricDiffusion
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

``problem.templates`` exposes scoped, read-only laws. The explorer retains
relative subscripts, histories, differentials, and conditional moment notation;
``problem.latex`` exports the same formulas.

API
---

.. currentmodule:: modello

.. autoclass:: Coordinate

.. autoclass:: Series
   :members: index, discrete

.. autoclass:: Recurrence
   :members: at, window, initial, start, equations

.. autoclass:: ItoProcess
   :members: drift, diffusion, generator, quadratic_variation_rate, exact, mean, variance, equations

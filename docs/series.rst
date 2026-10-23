Relative series and process laws
================================

Declare an evolution law independently of the observations used to analyse it.
``Series`` represents an indexed scalar family, ``Recurrence`` relates discrete
positions, and ``ItoProcess`` declares a continuous drift and diffusion. Queries
from those laws are ordinary scalar expressions in a flat ``Modello`` model.
Their parameters, boundary values, and derived quantities use the usual joint
solver, observations, domains, diagnostics, and solution families.

A relative recurrence
---------------------

The equation below has no prescribed terminal position or sample-vector size:

.. math::

   T_n=aT_{n-1},\qquad n\ge1,\qquad T_0=T_{\mathrm{opening}}.

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

The recurrence and its opening condition remain available as formulas. The
separate sample window contains positions zero through four; selecting position
ten does not change that window. Verified closed forms can answer distant
queries without materializing the preceding observations.

In model field definitions and family queries, ``T[k]`` uses its declared
recurrence. ``law.at(k)`` explicitly selects a law, including when comparing
several laws for the same symbolic family. Ambiguous implicit queries raise an
error. The series object itself remains a symbolic family; use the populated
instance's law for direct evaluation outside field definitions.

``Series`` indexing describes mathematical positions. ``T[n-1]`` is the
predecessor of ``T[n]``; ``T[-1]`` is position minus one. It never wraps to the
last stored observation. A series has no ``len`` or implicit last position.
``law.window(first, last)`` creates an inclusive finite :doc:`vector <vectors>`
view whose own Python positions start at zero. Its first entry represents
``T[first]``.

Declare each initial-history position explicitly. The history must be contiguous
and end immediately before ``start``; by default ``start`` is one after its last
position. Fixed positive lags must be covered by that history. For example:

.. code-block:: pycon

   >>> n = Index("n", nonnegative=True)
   >>> F = Series("F", n, integer=True)
   >>> fibonacci = Recurrence(F[n], F[n - 1] + F[n - 2], initial={0: 0, 1: 1})
   >>> fibonacci.start, fibonacci.at(10).doit()
   (2, 55)

Negative prehistory positions are permitted. Querying before the supplied
history gives an undefined value, excluded from finite model states. Future
references, zero lags, data-dependent lags, and incomplete histories raise
``ModelInputError``. These declarations specify causal recurrences; algebraic
cycles and general simultaneous systems still use ordinary model equations.

Coordinate assumptions remain requirements on the evaluated history, including
intermediate states that are not separately exposed as fields. A positive
terminal value does not permit a negative predecessor in a positive series.
Closed forms must preserve these requirements and the initial conditions;
induction proves them when scalar assumptions suffice. This does not establish
the consistency of arbitrary unobserved infinite systems.

Symbolic and finite evaluation
------------------------------

Constant first-order affine laws retain their exceptional zero and unit
coefficient cases. Supported higher-order linear recurrences use SymPy's
`recurrence solver <https://docs.sympy.org/latest/modules/solvers/solvers.html#recurrence-equations>`_,
with candidate formulas checked against the equation and initial history.
Generic higher-order coefficients wait for observations rather than silently
discarding singular parameter cases. Nonlinear causal recurrences can still
evaluate finite queries without an analytic formula.

Use ``.doit()`` for direct evaluation outside a model. Inspection preserves the
law; model evaluation expands referenced definitions, applies known constant
observations while protecting local calculus coordinates, and evaluates the
operators. An exact solution family rejects an operator that remains unresolved.

``Index`` and ``Coordinate`` are local mathematical variables, so they do not
become observable model fields. Use an ``InstanceDummy`` such as ``periods``
when a coordinate value itself should be observed or inferred. A bare series
term has no supplied law or probability distribution; declaring a ``Series``
does not automatically create infinitely many observable unknowns.

Observed innovations and a later boundary
------------------------------------------

A finite innovation vector supplies data to a relative equation. The same law
can infer an earlier unknown from a later boundary:

.. math::

   T_n=\rho T_{n-1}+\epsilon_{n-1},\qquad n\ge1.

.. code-block:: pycon

   >>> from sympy import Rational
   >>> from examples.relative_series import autoregressive_path
   >>> Path = autoregressive_path(4)
   >>> case = Path(
   ...     "reconcile", opening=2, retention=Rational(1, 2),
   ...     innovations=[1, 0, None, 0], closing=Rational(5, 4),
   ... )
   >>> case.innovations, case.levels
   ([1, 0, 2, 0], [2, 2, 1, 5/2, 5/4])

Here innovations are deterministic supplied or unknown quantities. The model
does not assign them independence, Gaussian distributions, or a time-spacing
rule. A larger data window changes the available inputs, while retaining the
relative equation. Missing innovations remain unknowns in that equation.

An observation-selected horizon
---------------------------------

The same recurrence can use an arbitrary finite ``N``. Here the state vector
has ``N+1`` entries and the innovation vector has ``N``:

.. code-block:: pycon

   >>> from examples.relative_series import ObservedAutoregression
   >>> case = ObservedAutoregression("states", retention=1, levels=[1, 3, 6, 10, 15])
   >>> case.N, case.opening, case.innovations
   (4, 1, [2, 3, 4, 5])
   >>> ObservedAutoregression("longer", retention=1, levels=[1, 2, 3, 4, 5, 6, 7]).N
   6

The declaration contains ``innovations = Vector("epsilon", size=N)`` and
``levels = law.window(0, N)``; it does not choose four periods. The vector length
supplies ``N+1 = len(levels)``. Alternatively, an innovation vector supplies
``N = len(innovations)``, or the caller can specify ``N`` directly. Partially
observed vectors still supply their complete length.

States and innovations are observations of the same equation, so supplying
both can overdetermine it. Consistent observations are checked together;
contradictions are rejected. The GUI's state-driven example starts with inferred
innovations. Its innovation-driven scenarios explicitly release the state
observations before supplying innovations.

A recurrence of order ``p`` needs ``p`` initial-history positions for causal
evaluation. That history is declared independently of the observation window.
There is therefore no universal rule requiring every observation vector to
contain ``p`` states, nor does a length of at least ``p`` ensure that unknown
coefficients or innovations are identifiable. A shorter window can be a valid
partial observation of a law with separately declared history. In this first-order
example, one state describes the opening boundary, with ``N=0`` and no transitions.

Continuous Itô laws
-------------------

For one standard Wiener driver, declare

.. math::

   \mathrm{d}X_t=a(t,X_t)\,\mathrm{d}t+b(t,X_t)\,\mathrm{d}W_t.

For a deterministic function that is once differentiable in time and twice in
state, the Itô drift and diffusion are

.. math::

   a_f &= f_t+a f_x+\tfrac12 b^2 f_{xx},\\
   b_f &= b f_x.

These are the one-state
`Itô formula <https://math.mit.edu/~sheffield/177/2011_177problemset2.pdf>`_.
``process.generator(f)`` gives the spatial operator
``a*f_x + b**2*f_xx/2``; ``process.drift(f)`` additionally includes ``f_t``.
``quadratic_variation_rate(f)`` is the coefficient of ``dt`` in the transformed
quadratic variation, rather than its time integral.

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

The positive state makes the logarithm's domain explicit. Transforms assume the
stated smoothness and adapted coefficients; this layer does not prove those
analytic hypotheses. A one-state transform rejects an additional unmodelled
process, whose quadratic covariation would need to be specified.

Named model formulas are expanded before differentiation. The state and time
stay independent during the partial derivatives; observations are applied to
the result. A local state coordinate is a function argument. A declared scalar
state field is an observable current-state quantity instead. As with other
symbolic external coordinates, materializing an exact family requires any
remaining state-dependent outputs to be resolved.

Flat inference for diffusion parameters
----------------------------------------

The geometric example uses the same constructor interface as the recurrence:

.. code-block:: pycon

   >>> from examples.relative_series import GeometricDiffusion
   >>> fitted = GeometricDiffusion(
   ...     "coefficients", opening=100, duration=1, wiener_increment=0,
   ...     log_drift=Rational(2, 25), log_diffusion=Rational(1, 5),
   ... )
   >>> fitted.growth, fitted.volatility
   (1/10, 1/5)
   >>> fitted.log_drift, fitted.log_diffusion
   (2/25, 1/5)
   >>> from sympy import exp
   >>> fitted.closing == 100 * exp(Rational(2, 25))
   True
   >>> fitted.mean == 100 * exp(Rational(1, 10))
   True

``exact(initial, elapsed, increment)`` gives an exact arithmetic or geometric
Brownian value conditional on the supplied Wiener increment. It does not
sample that increment. ``mean`` and ``variance`` are conditional moments given
the opening state and the declared Wiener law. Elapsed time must be nonnegative;
zero elapsed time requires a zero Wiener increment. Initial values must satisfy
the declared state domain. A conditional path evaluated
at increment zero is generally different from its conditional mean.

Affine additive diffusions also have exact conditional moments, including the
zero mean-reversion-rate limit:

.. code-block:: pycon

   >>> from examples.relative_series import MeanReversion, BrownianCompensator
   >>> reverting = MeanReversion(
   ...     "moments", opening=1, equilibrium=4, rate=log(2),
   ...     volatility=2, duration=1,
   ... )
   >>> reverting.mean
   5/2
   >>> BrownianCompensator("compensated", drift=0).rate
   1

An exact mean-reverting path requires a weighted Wiener integral. It is not
computed by substituting a single ordinary Wiener increment. Unsupported exact
path classes raise an explicit error. Zero diffusion retains the exact
deterministic limit. Zero drift is a local calculus condition;
general martingale claims also require integrability. Multiple drivers,
correlations, stochastic integrals, and general SDE boundary problems need
additional declared structure and solution methods.

Inspection, rendering, and numerical views
-------------------------------------------

``problem.templates`` is a read-only mapping of series and laws by component
path. ``problem.symbols.law`` accesses a law with this problem's bound scalar
and function identities. Components, inheritance, solution-family cases, and
compiled numerical views preserve those scopes and the evaluated law parameters.

The explorer's equations view shows relative subscripts, initial histories, and
continuous differentials. A direct scalar law query links back to its complete
law. LaTeX export uses the same scoped quantities without expanding the
recurrence into absolute scalar expressions.

Run ``python -m examples.relative_series`` for the console walkthrough. The
interactive timeseries demo includes relative and stochastic examples described
in :doc:`gui`.

API
----

.. currentmodule:: modello

.. autoclass:: Coordinate

.. autoclass:: Series
   :members: index, discrete

.. autoclass:: Recurrence
   :members: at, window, initial, start, equations

.. autoclass:: ItoProcess
   :members: drift, diffusion, generator, quadratic_variation_rate, exact, mean, variance, equations

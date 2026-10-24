Generality and solver boundaries
================================

Modello composes mathematical declarations and solves their finite scalar
equations with SymPy. The same definitions support forward calculations,
inverse problems, inspection, and exact solution families.

Representations and composition
--------------------------------

* :class:`~modello.Vector` contains scalar coordinates. Observed lengths and
  explicit size fields determine its finite shape before scalar solving.
  :func:`~modello.model_vector` retains a typed model at each position,
  including its component equations and fixed inner vectors.
* :func:`~modello.model_recurrence` generates joint equations over a finite
  window. Current-step, implicit, backward, and forward references are allowed
  within that window. ``start`` leaves a prefix available for boundary data.
  Piecewise boundaries can express centered difference or boundary-value
  equations using the ordinary solver.
* Relative vector positions are checked without Python negative-index
  wraparound. Explicit selections such as ``states[-1]`` keep Python indexing.
  The checks include local indices whose reduction bounds depend on a relative
  position, including nested moving windows.
  Native sums, products, integrals, derivatives, and limits retain their own
  lexical coordinates, including when a bound name also occurs freely in its
  limits.
* :class:`~modello.Series` names a scalar family without a chosen horizon.
  :class:`~modello.Recurrence` evaluates strictly causal self-lags from
  contiguous history; fixed references to that history are also allowed.
  Explicit queries of another recurrence compose directly. Unqualified terms
  from several series use the model's law registry through model fields or
  ``family.expression``; a standalone law has no implicit global registry.

Finite causal equations agree across these representations when they describe
the same history, window, and domains. A scalar recurrence checks its declared
initial history and states needed by a query; a model sequence checks every
member of its declared window. A later invalid state can therefore invalidate
that window while an earlier scalar query remains defined. See
:ref:`sequence-representations` for choosing between them.

Ordinary scalar fields are quantities, not parameterized function definitions.
Keep relative functions inside their law or a bound comprehension. Use
``Index`` and ``Coordinate`` for local coordinates, and ``InstanceDummy`` for
quantities to observe or infer.

Validity and solver results
----------------------------

Declarations retain their identities, coordinate kinds, and state domains
through symbolic reconstruction and renaming. Independently scoped components
remain independent. Named definitions expand in their declaration context
before calculus or parameter observations are applied.

Explicit denominator exclusions are retained through library substitution,
finite reduction, and cancellation. A zero observation cannot turn an undefined
ratio into a valid zero. Inactive Piecewise branches and empty reductions do
not evaluate unused operations. Limits with unresolved parameters, and
parameterized integrals with potential poles in their interval, remain
unevaluated until the required data or a SymPy domain proof is available.

Undefined queries produce ``nan`` and are excluded from materialized solutions.
Symbolic results retain necessary conditions. Exact families preserve joint
alternatives and unresolved conditions, accept exact inputs, and require checked
finite points before materialization. Unsupported operations report an error
or remain symbolic; an empty backend answer alone is not a proof of
inconsistency.

Unsupported model calculations raise ``ModelInputError`` through both solve
APIs. Exceptions raised by the actual solving backend use ``SolverBackendError``;
standalone symbolic operations may raise ``NotImplementedError``.

Construction requires one retained solver candidate. That candidate may be
symbolic and depend on external parameters. Attribute assignment after
construction does not recompute the model; construct another state or add
observations to a solution family.

One-state Itô extension
-----------------------

:class:`~modello.contrib.stochastic.ItoProcess` provides local drift, diffusion,
spatial generators, and quadratic variation for a smooth function of one state
and time. This opt-in extension lives in ``modello.contrib.stochastic``. Its
coefficients may be nonlinear, time-dependent, or adapted to another declared
state for these local calculations.

Nonzero-time exact queries support arithmetic/geometric coefficients and
affine-additive conditional moments. Deterministic affine paths are supported;
a general mean-reverting path requires more information than one Wiener
increment. Zero elapsed time retains its initial-value and zero-variance
identities. Exact realizations check both initial and endpoint state domains.

Known stochastic dependencies are checked before point observations can freeze
them, and remain attached to returned law objects. A one-state transform cannot
silently omit another state's dynamics; an exact moment cannot treat another
process as a fixed coefficient. Time-bound process trajectories require a path
or joint-process model. Standalone unregistered symbols represent fixed
parameters; declare processes in the model when their dynamics matter.
Exact coefficients that still depend on cyclic or multiply defined quantities
require an explicit parametrization; observations at one point do not supply it.

Mathematical and computational scope
------------------------------------

Vectors are one-dimensional and model-vector members require a concrete schema.
Finite windows expand into scalar equations; large or nonlinear systems can be
expensive. A general numerical time-stepper, ragged/tensor solver, coupled SDE
solver, correlated drivers, stochastic integration, and hitting-time or SDE
boundary solvers are outside the implementation.

Finite causal recurrences may be nonlinear. Symbolic-index closed forms are
limited to verified supported cases and SymPy's recurrence solver. Calculus
through cyclic or multiply defined quantities requires an explicit
parametrization.

Symbolic formulas assume their declared domains. Smoothness, adaptedness,
integrability, and pathwise domain invariance are mathematical hypotheses;
initial and endpoint checks do not prove an entire path admissible. Modello
cannot recover information already removed or conflated during native SymPy
expression construction, before it receives a declaration. Retain domain
requirements explicitly when simplification removes them. Upstream-only
regressions record independently reproduced SymPy failures separately.

``tests/scenario/README.rst`` maps independent regression scenarios to these
contracts. See :doc:`series`, :doc:`model-vectors`, and :doc:`solutions` for the
supported workflows.

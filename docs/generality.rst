Generality and solver boundaries
================================

A rendered formula is a declaration, not evidence that every associated problem
can be solved. ``tests/scenario/`` exercises models independently of the gallery,
including equivalent notation, alternative component groupings, inverse
problems, degenerate coefficients, and rejected problem classes. The test
README maps the features to scenario modules.

What the review uncovered
--------------------------

The initial scenario pass found library defects beyond the examples:

* A vector comprehension protected its bound index in its bounds as well as its
  body, so substituting a free endpoint could leave the window unchanged.
* Composed recurrences exposed terms from other series that were never resolved.
  Laws are now resolved in the problem context, including terms produced by
  earlier queries. Cycles must make progress towards an initial boundary.
* A symbolic discrete query lost its integer domain after conversion to a closed
  form. Its validity condition now includes integer membership.
* Current-step model equations worked when split into components but were
  rejected when grouped into one state law. Finite model laws now compile joint
  equations consistently, including implicit current-step equations and
  explicit terminal boundary references.
* Moving sums and sums of recurrence lags depended on their notation. Shared
  finite-index expansion now exposes the same terms for validation and solving.
* Native SymPy variable renaming could erase a series' discrete kind or an Itô
  state's domain, and detach a transformation from its law. Semantic metadata
  now survives reconstruction independently of coordinate spelling. Equivalent
  reconstructions retain their model context when named coefficients expand;
  populated laws preserve those formulas for subsequent queries and copies.
* Standalone Itô calculus and exact coefficient classification differed from
  calculations inside a model for finite operator bodies. They now share the
  same evaluation path. Zero-time identities precede exact classification.
* ``Piecewise`` evaluated unsupported formulas in branches already excluded by
  observations. Conditions now resolve before values; inactive branches are
  skipped while reachable unsupported calculations still report their limits.

These are library regressions covered by passing tests. The separate strict
upstream xfail reproduces a SymPy 1.14 finite-sum derivative that acquires a
spurious singularity at zero. Its docstring contains a standalone program and
source diagnostics. Materializing the finite body first avoids that failure in
Modello's evaluation path.

Choose the appropriate representation
-------------------------------------

:class:`~modello.Vector` contains scalar coordinates with a known or inferable
finite shape. :func:`~modello.model_vector` preserves a typed component at each
position; its finite field equations enter the same scalar solver. Current-step
implicit equations can therefore express discrete schemes such as implicit
Euler. Relative self-references must use fixed covered lags or the current
position. Explicit boundary references follow finite Python indexing.

:class:`~modello.Series` names a scalar family without a finite horizon.
:class:`~modello.Recurrence` evaluates causal laws from contiguous history.
Unlike finite model equations, it requires strictly positive self-lags.
Several declared laws may drive each other when finite queries reach their
boundaries. Finite higher-order queries iterate directly; symbolic queries seek
verified closed forms. An unresolved symbolic query remains an expression,
not a promise of a closed form.
Queries involving several laws need their problem context: use model fields or
``family.expression``. A standalone recurrence cannot discover another series'
law from its name.

An ordinary scalar field is not a parameterized function definition. Keep a
relative expression inside its law, use a bound vector comprehension, or declare
a ``Series`` and its law. A free local ``Index`` cannot be observed like an
``InstanceDummy``; exact families reject unresolved indexing.

Remaining limitations
----------------------

* Vectors are one-dimensional scalar containers. Model-vector elements need a
  concrete member schema; they may contain fixed inner vectors and components.
  Arbitrary tensors, ragged collections, and vector-valued continuous processes
  do not have dedicated modelling or solver support.
* Finite windows expand into scalar equations. Symbolic shape inference needs
  enough independent length or size observations before this expansion. It does
  not solve an unbounded collection of unknowns. Large or nonlinear windows may
  be expensive, and no general numerical time-stepper is provided.
* General symbolic recurrence solutions are limited by the implemented verified
  cases and SymPy. A finite nonlinear recurrence can work while the same law's
  symbolic-index query remains unresolved. Calculus through cyclic or multiply
  defined scalar quantities needs an explicit parametrization first.
* Local Itô drift, diffusion, and generators support one scalar state and one
  standard Wiener driver. Exact nonzero-time queries cover arithmetic/geometric
  coefficients and affine-additive conditional moments, with deterministic
  affine paths when diffusion is zero. Arbitrary SDEs, stochastic integrals,
  correlated drivers, hitting times, and SDE boundary solvers are not implemented.
* Smoothness, adaptedness, integrability, and pathwise domain invariance are
  mathematical premises, not automatic proofs. Checking an initial state and a
  supplied increment does not validate an entire stochastic path. Ordinary SymPy
  simplification can also discard a singularity before Modello receives an
  expression; retain necessary domain conditions explicitly.
* Construction requires one retained symbolic solver candidate. It is not a
  completeness proof. Exact solution families retain alternatives and conditions,
  reject floating observations, and require checked points for materialization.
  Attribute assignment after construction does not recompute a model.

The scenario suite tests these boundaries alongside successful cases. Passing
it establishes regression coverage for those contracts, not universal symbolic
solvability. See :doc:`series`, :doc:`model-vectors`, and :doc:`solutions` for the
supported workflows.

Contributing and releases
=========================

Use uv 0.13.1 or newer. The library and CI support Python 3.10 through 3.15.
Documentation tooling requires Python 3.12 or newer; notebook tooling has its
own optional Python 3.11+ group. All environments share the root ``uv.lock``.
The ``docs`` group includes pandas and Polars for executable DataFrame recipes.
Documentation CI tests those recipes with Sphinx before building HTML; the
ordinary pytest suite excludes that page.

Reading the core
----------------

Follow a model from its class body to its populated instances:

1. ``ModelloMetaNamespace`` in ``modello/_model.py`` collects scalar expressions,
   vectors, components, and inherited replacements. ``compile()`` produces one
   ``_ModelDefinition`` containing the field paths and scoped equations. Each
   ``_Definition`` keeps an equation together with its origin path, including
   separate definitions of the same aliased quantity. ``ModelloMeta`` attaches
   this declaration and the compiled annotations to the class.
2. ``inspect_model()`` calls ``_prepare_problem()`` in ``diagnostics.py`` to bind
   fresh symbols and create a ``ModelProblem`` snapshot. ``_InputCapture`` in
   ``_inputs.py`` appends scalar and component observations in that scope. It
   retains each copied value's supplied or inferred origin and groups all values
   from a component under its constructor argument. Preparation does not solve.
3. ``_algebra()`` in ``_problem.py`` creates an evaluated view of those equations.
   ``_expressions.py`` resolves directed definitions within local binder scopes,
   evaluates SymPy operators, and leaves ordinary algebraic cycles for the joint
   solver. This view is shared by construction, audits, and solution families;
   the original snapshot remains unchanged. Both the resolver and evaluator use
   one argument-rebuilding path to retain denominator domains and pending
   operators. A per-problem memo prevents repeated
   expansion of shared dependency graphs. Limits remain unevaluated until their
   free quantities are substituted, preserving exceptional parameter values
   that generic limit evaluation can discard.
4. ``Modello.__init__()`` prepares the problem, generates joint candidates,
   selects one compatible with the declared assumptions, and calls ``_populate()``
   to materialize the complete component tree. Symbolic construction accepts
   undecided assumptions. Diagnostic audits require a checked, closed witness
   before claiming consistency.
5. ``SolutionFamily._prepare()`` in ``solutions.py`` gives the same snapshot an
   explicit coordinate domain. ``_FamilyReduction`` owns the evolving coordinate
   expressions: first substitute isolated quantities, then eliminate rational
   affine joins. Independent observations and outputs are substituted in batches;
   cyclic joins retain a chart coordinate. Remaining operators are checked after
   observations and conditions have resolved their bounds.
   The family retains all original equations and every field's
   domain as predicates, including conditions on eliminated quantities. Backend
   failures retain the conditions and record a ``SolutionIssue``. ``cases()``
   audits each finite state before populating a freshly bound model tree.

Shared problem records live in ``_problem.py``. Runtime imports point downwards
through records, declarations, input capture, solution families, diagnosis, and
the public namespace; reverse type references use ``TYPE_CHECKING``. The GUI and
numerical example adapters consume the compiled declaration's fields.
``ModelProblem.vector_definitions`` retains complete vector formulas in the same
scope. Both GUI equations and LaTeX exports consume this public view instead of
reconstructing formulas from scalar constraints. ``_printing.py`` adds scoped
LaTeX labels; native vector printing remains in ``vectors.py``.
Text and LaTeX contexts receive the explicit vector schema, including public and
chart aliases. They choose names within the current component without guessing
vector identities from their scalar coordinate paths.

``_values.py`` holds structural input types; ``_symbols.py`` owns independent
quantity identities and local indices. ``vectors.py`` implements immutable
scalar containers and finite comprehensions without importing model or GUI
machinery. Its shared container reader checks native array rank and matrix
orientation for construction and observations. ``_scalar_expression`` sympifies
scalar inputs without evaluating them; ``_rebind`` preserves source denominator
domains while renaming declarations. Scalar and model vectors share size and
integer-position validation. Vector coordinates use normal scalar field
records. ``_paths.py``
shares coordinate traversal and bracketed display paths across consumers.
The GUI separates layout and permissions, MathML printing, bounded input parsing,
and vector summaries under ``_gui/``. Finance and time-series examples stay above
these layers.

``series.py`` declares relative recurrences. ``contrib/stochastic.py`` provides
the opt-in Itô extension; the core resolver uses scoped declaration hooks without
importing domain extensions.
``contrib/validation.py`` compiles model-vector laws into a stateless window
function. Each call checks its final row at an absolute position, returning
evidence and coverage. Domain and equation checks share history handling and
result construction. The stream iterator projects and converts each incoming
row once, retaining at most ``window_size`` rows of observations.
Audits require contiguous results from zero by default; other intervals need an
explicit start. DataFrame projection and windowing live in documentation
recipes. Validation constructs no model instances and performs no joint solve;
core imports load neither this extension nor frame libraries.
``_shapes.py`` solves symbolic finite axes from observation lengths before scalar
coordinates are compiled. Shape specialization produces a cached subclass; it
does not mutate the reusable declaration. The GUI creates a fresh finite context
for each proposed window, preserving original observations and restore behavior.

``_operators.py`` supplies their common scoped-expression protocol: operators
identify local coordinates and the arguments in which those coordinates are
protected while model definitions expand. They produce scalar expressions for
the existing algebra boundary. Series and laws are template declarations in
the compiled model, with fresh function identities in each component and problem.
``ModelProblem.templates`` exposes those scoped originals to printing and GUI
equations. Population and numerical views retain the originals and bind their
scalar parameters together, so a copied law cannot keep obsolete coefficients.

Delegate scalar algebra to SymPy: ``check_assumptions`` checks compatibility,
``Sum`` evaluates geometric weights in affine recurrences, and a shared
derivative-based recognizer identifies affine drift and diffusion. Keep the
recognizer independent of polynomial syntax: parameter-dependent ``Piecewise``
expressions can be affine without being accepted by ``as_poly``. Local state
variables retain the declared domain and the Itô law's real-state premise.

``model_vectors.py`` builds typed collections and compiles relative field laws
into ordinary scalar definitions. ``_model_laws.py`` separates structural
declaration records from immutable source templates: normalize the entire
record on inheritance and keep source formulas distinct from finite expansion.
Preserve equations already declared by member schemas. The compiler stores
ordered field updates; only the inspection template retains complete source
equations. Deferred laws must trigger shape preparation rather than disappearing
from a finite solve.

``_collections.py`` holds deferred locations and typed sparse observations
without importing model construction. Population allocates canonical components
once and connects their views. Structural shape/index binding precedes scalar
solving. The GUI's ``_gui/collections.py`` summarizes nested model members and
observation origins; its labels preserve canonical API paths for navigation.

Keep the four-step feature walkthrough small. Reuse the existing geometry and
portfolio models; ``examples/gallery/models.py`` holds the calibration equations.
The portfolio guide and gallery share ``AccountSequence`` from
``examples/portfolio/models.py``. ``catalog.py`` separates
tutorial steps from domain extensions and supplies one shared question progression.
``extensions.py`` holds specialist models; ``examples/series.py`` supports the
scalar-series guide independently of the GUI. ``app.py``
mounts the existing explorer through Werkzeug, preserving URL prefixes for
assets and API requests. Add an example when it demonstrates a distinct idea,
and derive scenario smoke tests from the catalog. Keep mathematical regressions
in focused library tests, independent of the gallery's menu and wording.

Two sequence evaluation strategies
----------------------------------

The public choice is explained in :ref:`sequence-representations`. Both laws
are immutable SymPy ``Basic`` declarations; their calculated quantities are
scalar ``Expr`` objects. The common scoped-expression protocol retains identity
and lexical bindings. Neither is a wrapper around ``IndexedBase`` or a native
SymPy sequence class.

* ``Series`` names a scalar family; its terms retain their series identity and
  coordinate. ``Recurrence`` stores a causal transition and contiguous history.
  It evaluates required positions while caching each value with its cumulative
  domain condition. Affine formulas and ``rsolve`` results are used only after
  checking boundary values, the transition, and the state domains. ``rsolve``
  receives an ordinary SymPy ``Function`` equation.
* ``ModelVector`` retains typed members. ``model_recurrence`` first records a
  deferred field update; its ``ModelRecurrence`` template retains source
  equations. Once shape is known, the compiler instantiates scalar equations
  across the finite window. Ordinary model solving handles these jointly with
  member equations and observations, including current or future references.

Share position handling rather than these two solvers. In ``_evaluation.py``,
``_relative_expression`` protects local binders and guards relative input
selections, propagating through moving reduction bounds. ``_indexed_value``
binds a position using the common substitution and finite-reduction machinery.
Conditional scalar evaluation keeps the value separate from its domain, so
substitution cannot hide undefined inputs or accumulate nested validity wrappers.
``test_sequence_parity`` runs finite causal problems through both representations;
it also checks the intentional difference between a queried prefix and an entire
finite window.

SymPy's `RecursiveSeq
<https://docs.sympy.org/latest/modules/series/sequences.html#sympy.series.sequences.RecursiveSeq>`_
does not replace the domain-carrying, scoped history required here. The
upstream-only tests also reproduce incorrect nonzero-origin queries in SymPy
1.14.0. Native ``Subs`` applies replacements in order; delegate that ordering
instead of rebuilding it as a dependency graph. Overlapping compound binders
can nevertheless compare equal despite different results, conflating inputs
inside native expression caches before Modello receives them. Strict upstream
xfails isolate these defects; library contracts must pass normally.

Other adapters follow the same rule: preserve exceptional parameter cases
while delegating supported scalar algebra to SymPy. Scoped expansion diagnoses
implicit calculus through cyclic definitions; unknown assumptions remain
distinct from disproven ones. Finite data reductions and explicit zero-rate
kernels avoid slower generic symbolic work without discarding valid cases.

Validation
----------

Install the locked development environment and run the complete test suite,
including doctests:

.. code-block:: console

   $ uv sync --locked --dev --group docs --python 3.12
   $ uv run --locked --dev pytest
   $ uv run --locked --group docs sphinx-build --fail-on-warning --keep-going --builder=doctest docs docs/_build/doctest
   $ uv run --locked --group docs sphinx-build --fail-on-warning --keep-going --builder=html docs docs/_build/html

Use ``.. doctest::`` for executable examples, including imports, setup, and
expected results. Name groups when examples need independent namespaces, as in
the pandas and Polars recipes. Sphinx tests these blocks with the documentation
dependencies installed. For examples that require Python 3.12+, use
``.. testcode::`` and ``.. testoutput::`` blocks; they run in this docs job.
The configured pytest ``--doctest-glob=*.rst`` also executes interactive prompts
outside the frame recipe page.
For a focused check, run ``uv run --locked --dev pytest --no-cov docs/modelling.rst``.

Unset ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` before running pytest if your shell
defines it. CI also checks formatting, lint, types, documentation, and both
distribution formats. The commands are maintained in ``AGENTS.md`` and
``.github/workflows/ci.yml``.

Mirror source packages under ``tests/`` and use ``test_<module>.py``, omitting a
module's leading underscore. Use ``test_init.py`` for package entry points and
keep repository-wide checks at the test root.

When Node.js is available, pytest also exercises the frontend's proposal state
against real Flask response fixtures. It covers bulk vector edits, subsequent
coordinate edits, restoring a vector while retaining a separate target, and
preserving the applied case after rejection. This optional check does not
replace visual browser review and adds no runtime dependency to the library.

Keep regressions focused on distinct behavior and independent mathematical
checks. Refresh dependencies with ``uv lock --upgrade`` and commit ``uv.lock``.
Dependabot proposes grouped weekly dependency and GitHub Actions updates.
Workflows do not schedule rebuilds.

Distribution checks install the wheel outside the checkout, rebuild the source
distribution, and repeat the runtime and typing checks. They also exercise the
packaged examples and fixtures, then install the locked GUI extra and exercise
the gallery and its assets. Builds use the installed, locked backend. Start
with an empty ``dist`` directory so the checks receive exactly one wheel and
one source distribution.

The wheel bundles the ``examples`` modules and fixtures. ``modello[examples]``
selects their optional runtime requirements; currently these are satisfied by
the standard library and the core SymPy dependency, so the extra's requirement
list is empty. Add future example-specific dependencies to this extra. Package
selection and data files belong in the setuptools configuration, independently
of dependency selection.

Read the Docs
-------------

Open ``docs/_build/html/index.html`` to preview the documentation. CI builds
Sphinx with warnings as errors. The root ``.readthedocs.yaml`` configures the same
Sphinx source and documentation dependency group on Python 3.12, following the
`Read the Docs uv configuration reference
<https://docs.readthedocs.com/platform/stable/config-file/v2.html#uv>`_.

The existing ``modello`` project on Read the Docs tracks ``master`` for its
``latest`` version and builds pull request previews. Enable version tags after
they include the documentation configuration. Project connections and version
activation are managed in Read the Docs service settings.

Example guides live under ``docs/examples/<example>/`` beneath the Examples
section. Keep each guide focused on the modelling idea it demonstrates.
Portfolio modules and fixtures are grouped in ``examples/portfolio/``;
``examples/data.py``, ``examples/named_members.py``, and ``examples/symbolic.py``
hold experimental data, named-projection, and numerical-evaluation helpers.
Keep example-specific behavior in those layers.

Diagrams use the ``.. mermaid::`` directive from
`sphinxcontrib-mermaid <https://sphinxcontrib-mermaid-demo.readthedocs.io/en/stable/>`_
in reStructuredText pages. The documentation dependency group enables the
extension for local, CI, and Read the Docs HTML builds.

Publishing
----------

Publish a GitHub release for a version tag such as ``v0.2.0``. The release
workflow runs the full CI matrix, documentation build, and distribution checks
on that release commit. It verifies that the tag still points to the commit,
normalizes the tag as a package version, and rejects development or local
versions. The installed wheel and rebuilt source distribution must both match
the release version.

Validated distributions pass through an immutable workflow artifact to a
separate publishing job. Only that job receives ``id-token: write`` and the
``pypi`` environment. Configure PyPI Trusted Publishing for the
``Code0x58/modello`` repository, ``publish.yml`` workflow, and ``pypi`` environment.
Active publishing runs for the same release tag are allowed to finish.

When ``CODECOV_TOKEN`` is configured, CI uploads coverage and test results to
Codecov. The uploads are skipped without the token.

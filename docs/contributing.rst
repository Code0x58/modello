Contributing and releases
=========================

Use uv 0.13.1 or newer. The library and CI support Python 3.10 through 3.15.
Documentation tooling requires Python 3.12 or newer; notebook tooling has its
own optional Python 3.11+ group. All environments share the root ``uv.lock``.

Reading the core
----------------

Follow a model from its class body to its populated instances:

1. ``ModelloMetaNamespace`` in ``modello/_model.py`` collects scalar expressions,
   components, and inherited replacements. ``compile()`` produces one
   ``_ModelDefinition`` containing the field paths and scoped equations. Each
   ``_Definition`` keeps an equation together with its origin path, including
   separate definitions of the same aliased quantity. ``ModelloMeta`` attaches
   this declaration and the compiled annotations to the class.
2. ``inspect_model()`` calls ``_prepare_problem()`` in ``diagnostics.py`` to bind
   fresh symbols and create a ``ModelProblem`` snapshot. ``_InputCapture`` in
   ``_inputs.py`` appends scalar and component observations in that scope. It
   retains each copied value's supplied or inferred origin and groups all values
   from a component under its constructor argument. Preparation does not solve.
3. ``Modello.__init__()`` prepares the problem, generates joint candidates,
   selects one compatible with the declared assumptions, and calls ``_populate()``
   to materialize the complete component tree. Symbolic construction accepts
   undecided assumptions. Diagnostic audits require a checked, closed witness
   before claiming consistency.
4. ``SolutionFamily._prepare()`` in ``solutions.py`` gives the same snapshot an
   explicit coordinate domain. ``_FamilyReduction`` owns the evolving coordinate
   expressions: first substitute isolated quantities, then eliminate rational
   affine joins. The family retains all original equations and every field's
   domain as predicates, including conditions on eliminated quantities. Backend
   failures retain the conditions and record a ``SolutionIssue``. ``cases()``
   audits each finite state before populating a freshly bound model tree.

Shared problem records live in ``_problem.py``. Runtime imports point downwards
through records, declarations, input capture, solution families, diagnosis, and
the public namespace; reverse type references use ``TYPE_CHECKING``. The GUI and
numerical example adapters consume the compiled declaration's fields.

Validation
----------

Install the locked development environment and run the complete test suite,
including doctests:

.. code-block:: console

   $ uv sync --locked --dev --group docs --python 3.12
   $ uv run --locked --dev pytest

Unset ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` before running pytest if your shell
defines it. CI also checks formatting, lint, types, documentation, and both
distribution formats. The commands are maintained in ``AGENTS.md`` and
``.github/workflows/ci.yml``.

Keep regressions focused on distinct behavior and independent mathematical
checks. Refresh dependencies with ``uv lock --upgrade`` and commit ``uv.lock``.
Dependabot proposes grouped weekly dependency and GitHub Actions updates.
Workflows do not schedule rebuilds.

Distribution checks install the wheel outside the checkout, rebuild the source
distribution, and repeat the runtime and typing checks. They also exercise the
packaged examples and fixtures. Builds use the installed, locked backend. Start
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

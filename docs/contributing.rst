Contributing and releases
=========================

Use uv 0.13.1 or newer. The library and CI support Python 3.10 through 3.15.
Documentation tooling requires Python 3.12 or newer; notebook tooling has its
own optional Python 3.11+ group. All environments share the root ``uv.lock``.

Validation
----------

Install the locked environment and run the same checks as CI:

.. code-block:: console

   $ uv sync --locked --dev --group docs --python 3.12
   $ uv run --locked --dev ruff format --check --diff .
   $ uv run --locked --dev ruff check .
   $ uv run --locked --dev mypy .
   $ uv run --locked --dev pytest --verbosity=2 --cov=modello --cov-report=xml --junit-xml=test-results.xml
   $ uv run --locked --group docs sphinx-build --fail-on-warning --keep-going --builder=html docs docs/_build/html

Unset ``PYTEST_DISABLE_PLUGIN_AUTOLOAD`` before running pytest if your shell
defines it. Format code with ``uv run --dev ruff format .``. Refresh dependencies
with ``uv lock --upgrade`` and commit ``uv.lock``. Dependabot proposes grouped
weekly dependency and GitHub Actions updates. Workflows do not schedule rebuilds.

Tests follow the source layers. Core model tests and public typing contracts
live in ``test_modello.py`` and ``typing_checks.py``. Example tests
live under ``tests/examples/``: ``test_data.py`` covers the independent data
helpers, ``test_symbolic.py`` covers numerical views, and
``test_named_members.py`` covers bound declarations and projections.
Financial tests live under ``portfolio/`` and follow its equation, valuation,
source, account, decision, analytics, and reporting layers. Their ``conftest.py``
shares the synthetic scenario; ``_data.py`` locates fixtures through the example
package, independently of the tests' location. Cross-layer import checks live
in ``tests/test_layer_boundaries.py``.

Core imports also follow layers: shared records in ``modello/_problem.py``,
declarations and population in ``_model.py``, solution families, then problem
preparation and diagnosis. The public namespace exposes these APIs and keeps
the constructor's solving path. Runtime imports point downwards; reverse type
references use ``TYPE_CHECKING``. Problem snapshots own constructor-input
capture, shared by construction and immutable family observations.

Executable walkthroughs cover the examples' ordinary usage. Focus additional
tests on distinct regressions, accounting and chronology invariants, and
independent mathematical checks. Test shared validation and compilation behavior
at their owning layer. Portfolio typing contracts are collected in
``tests/examples/portfolio/typing_contracts.py`` and checked against installed
packages as well as the checkout.

Existing jobs and streaming checks retain their original locations alongside
those examples. Example source modules retain their executable doctests.

Run a subtree while working on one layer:

.. code-block:: console

   $ uv run --locked --dev pytest tests/examples/portfolio
   $ uv run --locked --dev pytest tests/examples/test_data.py
   $ uv run --locked --dev pytest test_modello.py

Build and validate both distribution formats:

.. code-block:: console

   $ uv build --no-build-isolation --python .venv/bin/python
   $ uv run --locked --dev twine check --strict dist/*
   $ uv run --locked --dev python scripts/check_distributions.py dist

Start with an empty ``dist`` directory so that it contains exactly one wheel
and one source distribution. Builds use the installed, locked backend. The
distribution check installs locked runtime dependencies and the wheel in a fresh
environment outside the checkout, checks metadata, forward/inverse solving, and
the installed type annotations, then rebuilds the source distribution and repeats
the checks. It also selects the ``examples`` extra on both installed wheels,
checks the packaged CSV fixtures, and runs the portfolio entry points,
including HTML and JSON output. The core and generic-example typing contracts
include valid calls and expected type errors; unused-ignore checks ensure mypy
continues to catch those errors against the installed packages.

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

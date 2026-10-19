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
the checks. ``typing_checks.py`` includes valid calls and expected type errors;
unused-ignore checks ensure mypy continues to catch those errors.

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

Example guides live under ``docs/examples/<example>/`` and have their own
navigation beneath the Examples section. Portfolio modules and fixtures are
grouped in ``examples/portfolio/``; ``examples/compute/`` holds experimental
data and expression-execution helpers. Keep example-specific behavior in those
layers. The :doc:`named-solving design note <design/named_solving>` records the
candidate general modeling feature.

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

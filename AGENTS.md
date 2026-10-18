Use uv 0.13.1 or newer. Before committing, run the relevant checks from CI:

* `uv sync --locked --dev --group docs` (Python 3.12+ for documentation tooling)
* `uv run --locked --dev ruff format --check --diff .`
* `uv run --locked --dev ruff check .`
* `uv run --locked --dev mypy .`
* `uv run --locked --dev pytest --verbosity=2 --cov=modello --cov-report=xml --junit-xml=test-results.xml`
* `uv run --locked --group docs sphinx-build --fail-on-warning --keep-going --builder=html docs docs/_build/html`
* `uv build --no-build-isolation --python .venv/bin/python`
* `uv run --locked --dev twine check --strict dist/*`
* `uv run --locked --dev python scripts/check_distributions.py dist`

Format code with `uv run --dev ruff format .`. Keep `docs/` up to date with
user-facing behavior, API, and workflow changes. Test and document new behavior;
documentation must compile without warnings. The library supports Python 3.10+.

Commit `uv.lock` after dependency updates. Release publishing must validate the
release commit and tag, then publish only the validated distribution artifact
from the separate job with PyPI Trusted Publishing permissions.

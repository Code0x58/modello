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

Before pushing changes for a PR or creating/updating a PR, spawn a fresh
`pre_pr_review` subagent defined in
[its agent prompt](.codex/agents/pre_pr_review.toml). Give it the worktree path,
actual target ref/commit, `HEAD`, proposed PR title/body, and validation results.
Use the user's target or existing PR base; for a new PR, use the remote default
branch unless directed otherwise. Refresh the target ref when possible and
report if it could not be refreshed. The feature branch's tracking upstream is
not its PR target.

Use the named custom agent when supported; otherwise spawn a general subagent
with that file's `developer_instructions`. Request fresh context when supported.
Review the full branch and relevant local changes, loading the linked review
references relevant to the changes and finishing with PR wording. Wait for
findings, address actionable issues, and reassess affected areas after changes
to the target, branch, local files, or PR draft. When acting as the review
subagent, review directly and return findings; do not spawn another reviewer.

At commit time, check whether the changes need that review. The versioned
`.githooks/pre-commit` and `.githooks/pre-push` print reminders for the coding
agent; they do not run a model or certify a review. Install them per checkout
as described in `docs/contributing.rst`.

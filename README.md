# Modello
This project aims to explore symbolic modeling with object orientated programming. The heavy lifting is done by the [sympy](https://www.sympy.org/en/index.html) library. This module just provides a class to derive models from.

An example based on `examples/geometry.py`:
```python
from modello import InstanceDummy, Modello
from sympy import sqrt


class RightAngleTriangle(Modello):
    a = InstanceDummy("a", real=True, positive=True)
    b = InstanceDummy("b", real=True, positive=True)
    c = sqrt(a**2 + b**2)


T = RightAngleTriangle("T", a=3, b=4)
assert T.c == 5
T = RightAngleTriangle("T", b=4, c=5)
assert T.a == 3
```

The best place to see how this can be used is to look in the examples directory, which still needs padding out.

The functionality is covered by tests in both `test_modello.py` and doctests+tests in the examples.


## Installation
This can be installed using one of:
```sh
# using pip
pip install --user modello

# using git+pip
pip install --user git+https://github.com/Code0x58/modello.git#egg=modello
```

Requires Python 3.10 or newer and SymPy 1.14 or newer.


## Development
Use uv 0.13.1 or newer to install the locked development environment and run the checks:

```sh
uv sync --locked --dev
uv run --locked ruff format --check --diff .
uv run --locked ruff check .
uv run --locked mypy .
uv run --locked pytest
```

Format changes with `uv run ruff format .`. Refresh all Python dependencies with
`uv lock --upgrade` and commit `uv.lock`. The optional notebook environment is
documented in [notebooks/README.md](notebooks/README.md).

You can also install test dependencies with `pip install --editable '.[test]'`.
Ruff and mypy run separately from pytest, avoiding the old pytest lint plugins.

If you want to run a subset of tests or otherwise pass arguments to pytest, just invoke `pytest` directly, e.g.:
```sh
pytest examples/jobs.py::jobs.Job
```

## TODO:
 * elaborate on tests/examples
 * think about extending the functionality to allow for a more complete system of constraints, use the RightAngleTriangle example as a base
 * work out patterns for labels/names on symbols so they render nicely
 * for mypy, consider getting value from or removing it
 * implement a first attempt at nested models (named, rather than lists, which would be nice)
 * think about the possibility of symbolic (instance) dummies to allow live updates to system rather than modello instantiation being final
 * think about defaults (e.g expenses=0)

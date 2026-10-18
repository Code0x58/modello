[![PyPI Package](https://badge.fury.io/py/modello.svg)](https://pypi.org/project/modello/) [![Build Status](https://github.com/Code0x58/modello/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Code0x58/modello/actions/workflows/ci.yml) [![Coverage Report](https://codecov.io/gh/Code0x58/modello/branch/master/graph/badge.svg)](https://codecov.io/gh/Code0x58/modello) [![Documentation Status](https://readthedocs.org/projects/modello/badge/?version=latest)](https://modello.readthedocs.io/en/latest/?badge=latest)

# Modello

This project aims to explore symbolic modeling with object orientated programming. The heavy lifting is done by the [sympy](https://www.sympy.org/en/index.html) library. This module just provides a class to derive models from.

[Documentation](https://modello.readthedocs.io/) covers the API, development checks, and release workflow.

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

Models can contain named components. Using the triangle above:

```python
from modello import component


class Box(Modello):
    base = component(RightAngleTriangle)
    height = InstanceDummy("height", positive=True)
    diagonal = sqrt(base.c**2 + height**2)


box = Box("B", base=RightAngleTriangle("base", b=4), height=12, diagonal=13)
assert box.base.a == 3
```

Pass instances of the declared component classes at any depth; dictionaries are
not accepted. The whole tree is solved together; omitted inputs stay symbolic
unless constraints determine them. Supplied children are copied, so the parent
can complete their missing inputs without mutating the originals. See the
[nested-model guide](https://modello.readthedocs.io/en/latest/nesting.html) for
snapshot, inheritance, and construction semantics. `component(Model)` also
gives child attributes their model instance type for static checking.


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

Build the documentation with Python 3.12 or newer:

```sh
uv sync --locked --group docs --python 3.12
uv run --locked --group docs sphinx-build --fail-on-warning --keep-going --builder=html docs docs/_build/html
```

See [Contributing and releases](docs/contributing.rst) for distribution checks,
Read the Docs setup, and the validated PyPI Trusted Publishing workflow.

If you want to run a subset of tests or otherwise pass arguments to pytest, just invoke `pytest` directly, e.g.:
```sh
pytest examples/jobs.py::jobs.Job
```

## TODO:
 * elaborate on tests/examples
 * think about extending the functionality to allow for a more complete system of constraints, use the RightAngleTriangle example as a base
 * work out patterns for labels/names on symbols so they render nicely
 * think about the possibility of symbolic (instance) dummies to allow live updates to system rather than modello instantiation being final
 * think about defaults (e.g expenses=0)

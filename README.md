[![PyPI Package](https://badge.fury.io/py/modello.svg)](https://pypi.org/project/modello/) [![Build Status](https://github.com/Code0x58/modello/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Code0x58/modello/actions/workflows/ci.yml) [![Coverage Report](https://codecov.io/gh/Code0x58/modello/branch/master/graph/badge.svg)](https://codecov.io/gh/Code0x58/modello) [![Documentation Status](https://readthedocs.org/projects/modello/badge/?version=latest)](https://modello.readthedocs.io/en/latest/?badge=latest)

# Modello

Modello describes symbolic models with Python classes. Define relationships
between quantities, supply the values you know, and let
[SymPy](https://www.sympy.org/en/index.html) solve for the rest.

## Installation

```sh
pip install modello
```

Requires Python 3.10 or newer and SymPy 1.14 or newer.

## Define a model

Declare independent quantities with `InstanceDummy` and write expressions for
the relationships between them:

```python
from sympy import sqrt

from modello import InstanceDummy, Modello, component


class RightAngleTriangle(Modello):
    a = InstanceDummy("a", real=True, positive=True)
    b = InstanceDummy("b", real=True, positive=True)
    c = sqrt(a**2 + b**2)


triangle = RightAngleTriangle("T", a=3, b=4)
assert triangle.c == 5

triangle = RightAngleTriangle("T", b=4, c=5)
assert triangle.a == 3
```

The same model works in either direction. Positive assumptions select the
positive side length when solving for `a`.

## Compose models

Use `component()` to give a model named parts. A parent's equations can determine
unknown values inside its children:

```python
class Box(Modello):
    base = component(RightAngleTriangle)
    height = InstanceDummy("height", positive=True)
    diagonal = sqrt(base.c**2 + height**2)


base = RightAngleTriangle("base", b=4)
box = Box("B", base=base, height=12, diagonal=13)
assert box.base.a == 3
```

Here the box's height and diagonal determine the missing side of its base.
The original `base` stays symbolic; `box.base` contains the completed values.
`component(RightAngleTriangle)` also gives `box.base` its model instance type
for editors and type checkers.

Follow the [nested-model guide](https://modello.readthedocs.io/en/latest/nesting.html)
for more examples of composition and symbolic inputs. The
[examples directory](examples/) includes geometry, flow, and tax models, and the
[API reference](https://modello.readthedocs.io/en/latest/api.html) describes the
available classes and functions.

## Development

See [Contributing and releases](docs/contributing.rst) for environment setup,
tests, documentation builds, and publishing. The optional notebook environment
is described in [notebooks/README.md](notebooks/README.md).

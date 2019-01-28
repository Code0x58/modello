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
# using pipenv
pipenv install modello

# using pip
pip install --user modello

# using git+pipenv
pipenv install git+https://github.com/Code0x58/modello.git#egg=modello

# using git+pip
pip install --user git+https://github.com/Code0x58/modello.git#egg=modello
```

Currently this requires Python 3.6+ but the version requirements can drop a couple of minor versions easily if there is interest. Python 2.7 isn't planned to be supported as the Modello class relies on [PEP-3115](https://www.python.org/dev/peps/pep-3115/).


## Development
Run the tests and linting with `python setup.py test`. Pushes have the test suite run against them, and will also publish a release if tagged thanks to GitHub Actions. You can reproduce the Actions locally using [act](https://github.com/nektos/act), e.g. `TWINE_USERNAME= TWINE_PASSWORD= act`.

## TODO:
 * elaborate on tests/examples
 * think about extending the functionality to allow for a more complete system of constraints, use the RightAngleTriangle example as a base
 * work out patterns for labels/names on symbols so they render nicely
 * for mypy, consider getting value from or removing it
 * implement a first attempt at nested models (named, rather than lists, which would be nice)
 * think about the possibility of symbolic (instance) dummies to allow live updates to system rather than modello instantiation being final
 * think about defaults (e.g expenses=0)

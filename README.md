[![PyPI Package](https://badge.fury.io/py/modello.svg)](https://pypi.org/project/modello/) [![Build Status](https://github.com/Code0x58/modello/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Code0x58/modello/actions/workflows/ci.yml) [![Coverage Report](https://codecov.io/gh/Code0x58/modello/branch/master/graph/badge.svg)](https://codecov.io/gh/Code0x58/modello) [![Documentation Status](https://readthedocs.org/projects/modello/badge/?version=latest)](https://modello.readthedocs.io/en/latest/?badge=latest)

# Modello

Express a modelling question as quantities and relationships, then supply what
you know. The same model can calculate a result, infer missing inputs, or retain
a symbolic formula. [SymPy](https://www.sympy.org/) handles the algebra.

```sh
pip install modello
```

Requires Python 3.10+ and SymPy 1.14+.

## Start with a relationship

A brace spans a right-angle opening. Declare positive side lengths and their
relationship; choose the solving direction through the observations:

```python
from sympy import sqrt
from modello import InstanceDummy, Modello


class Triangle(Modello):
    a = InstanceDummy("a", positive=True)
    b = InstanceDummy("b", positive=True)
    c = sqrt(a**2 + b**2)


assert Triangle("brace", a=3, b=4).c == 5
assert Triangle("run", b=8, c=10).a == 6
```

Omit a quantity to leave it unknown. Construct another case to try different
observations; changing an attribute does not recalculate the model.

## Build up your problem

The [getting-started guide](docs/getting-started.rst) develops four concepts in order:

| Concept | How it helps express a question |
| --- | --- |
| Scalar models | Relate quantities, declare domains, and solve in either direction. |
| Component models | Reuse smaller models; parent observations can determine child quantities. |
| Vector models | Express predictions across observations and fit parameters with an objective. |
| Recurrent models | Relate typed states over time and infer missing events from boundary observations. |

The examples progress from a brace and parcel to sensor calibration and account
reconciliation. Each uses the same questions: calculate results, infer inputs,
try different observations, leave unknowns, and check conflicting claims.

Use [solution families](docs/solutions.rst) to retain alternatives or add bounds.
Use [inspection and diagnostics](docs/debugging.rst) when observations disagree
or a calculation remains unresolved. The [solver boundaries](docs/generality.rst)
record what the current implementation supports.

## Explore interactively

The optional GUI follows the same four-step progression:

```sh
pip install 'modello[gui]'
python -m examples.gallery
```

From a checkout, run `uv run --extra gui python -m examples.gallery`.
Open <http://127.0.0.1:5070> to inspect formulas and try the guided questions.
The [gallery guide](docs/examples/gallery/index.rst) explains the controls;
the [GUI quickstart](docs/gui.rst) shows how to explore your own model.

Domain extensions are opt-in. For example,
[`modello.contrib.stochastic`](docs/contrib/stochastic.rst) adds one-state Itô
formulas while using the same observations and modelling interface.

## Further reading

- [Documentation](https://modello.readthedocs.io/) and [API reference](docs/api.rst).
- [Practical applications](docs/examples/applications/index.rst): invoices, production, and circuits.
- [Research examples](docs/examples/portfolio/index.rst): portfolio reporting and calibration.
- [Reconciliation](docs/examples/reconciliation/index.rst): compare source data and infer suspect inputs.
- [Observed data validation](docs/contrib/validation.rst): generic window checks with explicit coverage and tested [pandas/Polars recipes](docs/contrib/validation-frames.rst).
- [Contributing](docs/contributing.rst): setup, checks, documentation, and releases.

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

## Build a system from its parts

Define a holding's state and an execution, then connect them with balance
equations. Each part is an ordinary model with typed, named quantities:

```python
from modello import InstanceDummy, Modello, component


class HoldingState(Modello):
    quantity = InstanceDummy("quantity", real=True)
    cash = InstanceDummy("cash", real=True)


class Execution(Modello):
    quantity = InstanceDummy("quantity", real=True)
    price = InstanceDummy("price", positive=True)
    fee = InstanceDummy("fee", nonnegative=True)
    notional = quantity * price


class FillPosting(HoldingState):
    before = component(HoldingState)
    fill = component(Execution)
    after = component(HoldingState)

    quantity = after.quantity
    quantity = before.quantity + fill.quantity
    cash = after.cash
    cash = before.cash - fill.notional - fill.fee
```

The first `quantity` assignment binds the name to `after.quantity`; the second
defines its relationship to the other components. The cash assignments do the
same for settlement. Together, these equations describe the whole posting.

Supply an opening state and a fill to find the closing balances:

```python
opening = HoldingState("opening", quantity=3, cash=1000)
posted = FillPosting(
    "buy",
    before=opening,
    fill=Execution("execution", quantity=2, price=100, fee=1),
)
assert posted.after.quantity == 5
assert posted.after.cash == 799
```

Or observe the closing cash and let the **same model** infer the execution price:

```python
observed = FillPosting(
    "infer_price",
    before=opening,
    fill=Execution("execution", quantity=2, fee=1),
    after=HoldingState("observed", cash=799),
)
assert observed.fill.price == 100
assert opening.cash == 1000
```

Components are solved together, so known values can determine unknowns anywhere
in the hierarchy. `before` and `after` describe economic order; the values you
supply determine the solving direction. The original instances stay unchanged,
and the completed components retain their model types for editors, traversal,
and further composition. A posting is itself a holding state: its `quantity`
and `cash` describe the result. Declare another posting as its `before`
component to nest the operations and retain their complete equations.

Follow the [nested-model guide](https://modello.readthedocs.io/en/latest/nesting.html)
for more examples of composition and symbolic inputs. The
[examples directory](examples/) includes geometry, flow, portfolio analytics, and
tax models. The [API reference](https://modello.readthedocs.io/en/latest/api.html)
describes the available classes and functions.

The [portfolio research example](https://modello.readthedocs.io/en/latest/examples/portfolio/index.html)
explores named holdings, market data, account replay, signals, analytics, and
symbolic reports. Its code and fixtures live in [examples/portfolio](examples/portfolio/),
with a separate documentation section. The
[named-member solving note](https://modello.readthedocs.io/en/latest/examples/portfolio/index.html#named-member-solving-as-a-general-feature)
describes the general modeling feature being explored.

Install and run the examples with their bundled synthetic data:

```sh
pip install "modello[examples]"
python -m examples.portfolio --html account.html --json account.json
```

## Development

See [Contributing and releases](docs/contributing.rst) for environment setup,
tests, documentation builds, and publishing. The optional notebook environment
is described in [notebooks/README.md](notebooks/README.md).

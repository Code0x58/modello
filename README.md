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

## Inspect a failed solve

Conflicting observations, ambiguous candidates, and backend failures raise
`ModelSolveError` with a snapshot of the problem. Match the failure to decide
when to request a detailed report:

```python
from modello import ModelSolveError, SolutionCountError, SolverBackendError

try:
    FillPosting(
        "reconcile",
        before=opening,
        fill=Execution("execution", quantity=2, price=100, fee=1),
        after=HoldingState("closing", cash=800),
    )
except ModelSolveError as error:
    match error:
        case SolutionCountError(solution_count=0):
            print(error.problem.analyse(conflicts=True))
        case SolverBackendError(backend_error=cause):
            print(f"Solver failed: {cause}")
        case _:
            print(error)
```

`SolutionCountError` exposes `solution_count` for retained candidates and
`returned_count` before domain checks. Counts can describe symbolic families.
A zero count alone does not prove a contradiction. Analysis looks for checked
evidence and runs only when you ask
for it. `print(error.problem)` shows the nested equations without running analysis.

Catch `ModelloError` to handle Modello's validation and solve failures together,
or use the specific classes in the
[error reference](https://modello.readthedocs.io/en/latest/api.html#errors).

Follow the [nested-model guide](https://modello.readthedocs.io/en/latest/nesting.html)
for more examples of composition and symbolic inputs. The
[examples directory](examples/) includes geometry, flow, portfolio analytics, and
tax models. The [API reference](https://modello.readthedocs.io/en/latest/api.html)
describes the available classes and functions.
The [debugging guide](https://modello.readthedocs.io/en/latest/debugging.html)
explains symbolic results, conflicting observations, and ambiguous solutions.

The [portfolio research example](https://modello.readthedocs.io/en/latest/examples/portfolio/index.html)
develops this pattern in one short walkthrough: nest transitions, join a
reporting layer, then refine an execution model and fit it through an outer
constraint. A richer exploratory implementation and its synthetic fixtures live
in [examples/portfolio](examples/portfolio/).

Install and run the examples with their bundled synthetic data:

```sh
pip install "modello[examples]"
python -m examples.portfolio --html account.html --json account.json
```

## Development

See [Contributing and releases](docs/contributing.rst) for environment setup,
tests, documentation builds, and publishing. The optional notebook environment
is described in [notebooks/README.md](notebooks/README.md).

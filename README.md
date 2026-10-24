[![PyPI Package](https://badge.fury.io/py/modello.svg)](https://pypi.org/project/modello/) [![Build Status](https://github.com/Code0x58/modello/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/Code0x58/modello/actions/workflows/ci.yml) [![Coverage Report](https://codecov.io/gh/Code0x58/modello/branch/master/graph/badge.svg)](https://codecov.io/gh/Code0x58/modello) [![Documentation Status](https://readthedocs.org/projects/modello/badge/?version=latest)](https://modello.readthedocs.io/en/latest/?badge=latest)

# Modello

Modello describes symbolic models with Python classes. Define relationships
between quantities, supply the values you know, and let
[SymPy](https://www.sympy.org/en/index.html) solve for the rest.

Compose small models into typed components, use the same equations in either
direction, and inspect the formulas and observations behind a result.

## Installation

```sh
pip install modello
```

Requires Python 3.10 or newer and SymPy 1.14 or newer.

## Build a system from its parts

Define a holding's state and an execution, then connect them with balance
equations. Each part has named quantities, and components retain their Python
model types:

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

Construction requires one retained solver candidate. Zero or several candidates
raise `SolutionCountError`; a solver exception raises `SolverBackendError`.
Both inherit from `ModelSolveError` and retain a snapshot of the problem. Match
the failure to decide when to request a detailed report:

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
`returned_count` for captured candidates before domain checks. A candidate may
represent a symbolic family of answers; a zero count alone does not prove a
contradiction. Analysis looks for checked evidence and runs only when you ask
for it. `print(error.problem)` shows the nested equations without running analysis.

Catch `ModelloError` to handle Modello's validation and solve failures together,
or use the specific classes in the
[error reference](https://modello.readthedocs.io/en/latest/api.html#errors).

The [nested-model guide](https://modello.readthedocs.io/en/latest/nesting.html)
develops composition and symbolic inputs; the
[API reference](https://modello.readthedocs.io/en/latest/api.html) covers classes
and functions. The [examples directory](examples/) also includes geometry, flow,
portfolio analytics, and historical tax models.

The [debugging guide](https://modello.readthedocs.io/en/latest/debugging.html)
explains conflicts and ambiguous solutions. Use
[solution families](https://modello.readthedocs.io/en/latest/solutions.html)
to explore alternative values and bounded observations before selecting a model.

The [practical applications](docs/examples/applications/index.rst) demonstrate
invoice reconciliation, production planning, and circuit measurement checks.

One symbolic model can support both pricing and reconciliation, saving time
spent manually substituting parameters or rearranging formulas for each check.
The [credit default swap (CDS) example](docs/examples/reconciliation/index.rst)
holds trusted profit and loss (PnL) fixed while comparing source components and
investigating suspect inputs. It also derives a readable pricing formula from
the same model. The reusable
[reconciliation framework](docs/examples/reconciliation/framework.rst) tests
decimal shifts and configured quote-unit, lot-size, or sign corrections before
solving backwards for individual inputs.

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

## Vectors and full formulas

Retain indexed sums and vector comprehensions while solving individual entries:

```python
from sympy import Sum
from modello import Index, Modello, Vector, vector


class Window(Modello):
    samples = Vector("x", size=3, real=True)
    i = Index("i")
    total = Sum(samples[i], (i, 0, len(samples) - 1))
    changes = vector(samples[i] - samples[i - 1], (i, 1, len(samples) - 1))


window = Window("observed", samples=[1, None, 3], total=6)
assert window.samples == Vector(1, 2, 3)
assert window.changes == Vector(1, 1)
```

Vectors share scalar domains, aliases, component scopes, and solution families.
Use SymPy `Product`, `Integral`, `Derivative`, and `Limit` directly; the explorer
keeps their mathematical notation and gives vector symbols an arrow. The
[vector guide](docs/vectors.rst) explains indices, observations, and symbolic
vector sizes such as `size=N`, where supplied lengths determine the finite horizon. The
[example gallery](docs/examples/gallery/index.rst) demonstrates vector summaries,
relative sequences, typed account states, integrals, calibration, and Itô processes.

Declare horizon-independent relative laws with `Series` and `Recurrence`, such
as `T[n] = a*T[n-1]`, then query a position or a finite observation window.
`ItoProcess` adds one-state Itô transformations and supported exact diffusion
values and conditional moments through the same flat model interface. The
[series and process guide](docs/series.rst) includes discrete and continuous
examples, boundary conditions, and solver scope.

## Development

See [Contributing and releases](docs/contributing.rst) for environment setup,
tests, documentation builds, and publishing. The optional notebook environment
is described in [notebooks/README.md](notebooks/README.md).

### Interactive exploration (WIP)

Install `modello[gui]` and run `python -m examples.gallery`, or from a checkout:

```console
uv run --extra gui python -m examples.gallery
```

Open <http://127.0.0.1:5070> for a four-step feature walkthrough: scalar models,
components, vectors, and recurrent model states. Each offers the same practical
questions: calculate, infer, change inputs, retain unknowns, and check observations.
The separate Extensions page demonstrates `modello.contrib.stochastic`.
Inspect the MathML equations, edit or release inputs, and use a calculated result as a target.
The [gallery guide](docs/examples/gallery/index.rst) explains each model's idea;
the [GUI quickstart](docs/gui.rst#quickstart-your-own-explorer) shows how to build
your own explorer. Presentation metadata uses the core annotation registry,
and Python snapshots work without Flask. The base install remains SymPy only.

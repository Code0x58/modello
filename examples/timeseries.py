"""Finite paths: state transitions, compounding, rolling windows, and exposure.

The model factories choose a finite shape. Acquisition, timestamps, missing-data
policies, and plotting belong to their callers; all relationships here remain
ordinary Modello equations with SymPy operators.
"""

from __future__ import annotations

from functools import lru_cache

from sympy import Abs, Integral, Product, Sum, Symbol

from examples.symbolic import SymbolicModel
from modello import Index, InstanceDummy, Modello, Vector, component, vector


class HoldingState(Modello):
    """Signed inventory and settlement cash at an event boundary."""

    quantity = InstanceDummy("quantity", real=True)
    cash = InstanceDummy("cash", real=True)


class InventoryPath(HoldingState, SymbolicModel):
    """A resulting holding state with the complete ordered fill path retained."""

    before: HoldingState
    after: HoldingState
    fills: Vector
    prices: Vector
    quantities: Vector


@lru_cache(maxsize=None, typed=True)
def inventory_path(periods: int) -> type[InventoryPath]:
    """Declare one signed fill and execution price per period; fees are excluded."""
    _positive_size(periods)

    class Path(InventoryPath):
        before = component(HoldingState)
        after = component(HoldingState)
        fills = Vector("f", size=periods, real=True)
        prices = Vector("p", size=periods, positive=True)
        i = Index("i")
        j = Index("j")
        quantities = vector(before.quantity + Sum(fills[j], (j, 0, i - 1)), (i, 0, periods))
        notional = fills.dot(prices, i)
        turnover = Sum(Abs(fills[i]), (i, 0, periods - 1))
        quantity = after.quantity
        quantity = before.quantity + fills.sum(i)
        cash = after.cash
        cash = before.cash - notional

    return Path


class ReturnPath(SymbolicModel):
    """A finite return path with positive gross factors and explicit capital."""

    before = InstanceDummy("before", positive=True)
    returns: Vector
    factors: Vector
    wealth: Vector


@lru_cache(maxsize=None, typed=True)
def return_path(periods: int) -> type[ReturnPath]:
    """Compound simple period returns without an implicit annualization rule."""
    _positive_size(periods)

    class Path(ReturnPath):
        returns = Vector("r", size=periods, real=True)
        factors = Vector("g", size=periods, positive=True)
        i = Index("i")
        j = Index("j")
        factors = vector(1 + returns[i], (i, 0, periods - 1))
        growth = factors.product(i)
        total_return = growth - 1
        after = ReturnPath.before * growth
        wealth = vector(ReturnPath.before * Product(factors[j], (j, 0, i - 1)), (i, 0, periods))

    return Path


class RollingWindow(SymbolicModel):
    """A complete observation vector and its explicitly sized trailing means."""

    samples: Vector
    means: Vector


@lru_cache(maxsize=None, typed=True)
def rolling_window(size: int, width: int) -> type[RollingWindow]:
    """Use complete, equally weighted windows, without padding or lookahead."""
    _positive_size(size)
    _positive_size(width)
    if width > size:
        raise ValueError("Window width cannot exceed the sample count")

    class Window(RollingWindow):
        samples = Vector("x", size=size, real=True)
        i = Index("i")
        j = Index("j")
        means = vector(Sum(samples[j], (j, i - width + 1, i)) / width, (i, width - 1, size - 1))

    return Window


_time = Symbol("t", real=True)


class LinearExposure(SymbolicModel):
    """Integrate a stated linear interpolation between two observed levels."""

    opening = InstanceDummy("opening", real=True)
    closing = InstanceDummy("closing", real=True)
    duration = InstanceDummy("duration", positive=True)
    exposure = Integral(opening + (closing - opening) * _time / duration, (_time, 0, duration))


def _positive_size(size: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("Path and window sizes must be positive integers")


if __name__ == "__main__":
    Inventory = inventory_path(4)
    opening = HoldingState("opening", quantity=5, cash=1000)
    result = Inventory("fills", before=opening, fills=[2, -1, None, -2], prices=[100, 110, 105, 120], quantity=7)
    print("Inferred fills:", result.fills)
    print("Inventory path:", result.quantities)
    print("Closing cash:", result.cash)
    print("Rolling means:", rolling_window(5, 3)("window", samples=[1, 2, 3, 4, 5]).means)
    print("Compounded wealth:", return_path(3)("returns", before=100, returns=["1/10", "-1/20", "1/5"]).wealth)
    print("Linear exposure:", LinearExposure("area", opening=2, closing=4, duration=3).exposure)

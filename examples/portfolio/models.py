"""Shared symbolic instruments and holdings, independent of tutorial runners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
from typing import ClassVar

from examples.compute.series import Series
from modello import InstanceDummy, Modello, component


@dataclass(frozen=True)
class Equity:
    """Serializable quote identity, separate from a configured data request."""

    symbol: str
    provider: str
    currency: str = "USD"

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip() for value in (self.symbol, self.provider, self.currency)
        ):
            raise ValueError("An equity needs a symbol, provider, and quote currency")


AAPL = Equity("AAPL", provider="synthetic")
MSFT = Equity("MSFT", provider="synthetic")
GOOGL = Equity("GOOGL", provider="synthetic")


class Quote(Modello):
    """A symbolic price; provider identity comes from the holding declaration."""

    price = InstanceDummy("price", positive=True)


class PriceChange(Modello):
    """Two marks of the same instrument, in currency and proportional terms."""

    before = component(Quote)
    after = component(Quote)
    change = after.price - before.price
    simple_return = after.price / before.price - 1


class Position(Modello):
    """A marked position, including signed virtual attribution sleeves."""

    instrument = component(Quote)
    quantity = InstanceDummy("quantity", real=True)
    market_value = quantity * instrument.price


class EquityPosition(Position):
    """A long position in an equity, separate from the instrument itself."""

    quantity = InstanceDummy("quantity", nonnegative=True)


class AccountValue(Modello):
    """The net asset value at an account boundary."""

    nav = InstanceDummy("nav", positive=True)


class Trade(Modello):
    """A signed share exchange at a price, before settlement charges."""

    quantity = InstanceDummy("quantity", real=True)
    price = InstanceDummy("price", positive=True)
    notional = quantity * price


class Allocation(Modello):
    """A share of capital expressed as an amount and a weight."""

    capital = InstanceDummy("capital", positive=True)
    weight = InstanceDummy("weight", nonnegative=True)
    value = capital * weight


class USDCash(Modello):
    """A nonnegative USD cash balance with no interest accrual."""

    currency: ClassVar[str] = "USD"
    amount = InstanceDummy("amount", nonnegative=True)


@dataclass(frozen=True)
class EquityPrice:
    """An identified positive scalar quote, ready for portfolio evaluation."""

    equity: Equity
    price: float

    def __post_init__(self) -> None:
        if not isfinite(self.price) or self.price <= 0:
            raise ValueError("An equity quote must be finite and strictly positive")


@dataclass(frozen=True)
class EquityPrices:
    """An instrument's positive closing prices on a dated observation axis."""

    equity: Equity
    close: Series[date]

    def __post_init__(self) -> None:
        """Keep invalid quotes out of valuation and return calculations."""
        if any(value <= 0 for value in self.close.values):
            raise ValueError("Equity prices must be strictly positive")

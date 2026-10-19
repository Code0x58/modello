"""Shared symbolic instruments and holdings, independent of tutorial runners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
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
    """A symbolic price; provider identity is supplied by the basket binding."""

    price = InstanceDummy("price", positive=True)


class EquityPosition(Modello):
    """A long position in an equity, separate from the instrument itself."""

    instrument = component(Quote)
    quantity = InstanceDummy("quantity", nonnegative=True)
    market_value = quantity * instrument.price


class USDCash(Modello):
    """A nonnegative USD cash balance with no interest accrual."""

    currency: ClassVar[str] = "USD"
    amount = InstanceDummy("amount", nonnegative=True)


@dataclass(frozen=True)
class EquityPrices:
    """An instrument's positive closing prices on a dated observation axis."""

    equity: Equity
    close: Series[date]

    def __post_init__(self) -> None:
        """Keep invalid quotes out of valuation and return calculations."""
        if any(value <= 0 for value in self.close.values):
            raise ValueError("Equity prices must be strictly positive")

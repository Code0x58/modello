"""Shared symbolic instruments and holdings, independent of tutorial runners."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Generic, TypeVar

from examples.compute.series import Series
from modello import InstanceDummy, Modello, component


class Equity(Modello):
    """A quoted equity, with identity and currency attached to its type."""

    ticker: ClassVar[str]
    currency: ClassVar[str] = "USD"
    price = InstanceDummy("price", positive=True)


class AAPL(Equity):
    """Apple equity; example observations are synthetic."""

    ticker = "AAPL"


class MSFT(Equity):
    """Microsoft equity; example observations are synthetic."""

    ticker = "MSFT"


class EquityPosition(Modello):
    """A long position in an equity, separate from the instrument itself."""

    instrument = component(Equity)
    quantity = InstanceDummy("quantity", nonnegative=True)
    market_value = quantity * instrument.price


class ApplePosition(EquityPosition):
    """An explicitly typed position in AAPL."""

    instrument = component(AAPL)


class MicrosoftPosition(EquityPosition):
    """An explicitly typed position in MSFT."""

    instrument = component(MSFT)


class USDCash(Modello):
    """A nonnegative USD cash balance with no interest accrual."""

    currency: ClassVar[str] = "USD"
    amount = InstanceDummy("amount", nonnegative=True)


_EquityT = TypeVar("_EquityT", bound=Equity, covariant=True)


@dataclass(frozen=True)
class EquityPrices(Generic[_EquityT]):
    """An instrument's positive closing prices on a dated observation axis."""

    equity: type[_EquityT]
    close: Series[date]

    def __post_init__(self) -> None:
        """Keep invalid quotes out of valuation and return calculations."""
        if any(value <= 0 for value in self.close.values):
            raise ValueError("Equity prices must be strictly positive")

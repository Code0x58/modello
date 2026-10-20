"""Typed portfolio cases and paths evaluated from their existing model structure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Generic, Protocol, TypeVar

from sympy import Expr

from examples.compute.named_map import NamedMap
from examples.compute.providers import load_providers
from examples.compute.series import Series, utc_time
from examples.model_plan import Known, ModelPlan
from examples.named_members import cached_definition
from examples.portfolio.basket import EquityBasket, EquityHolding, EquityHoldings
from examples.portfolio.models import Equity, EquityPrice, EquityPrices, USDCash
from examples.portfolio.providers import EquityProvider
from examples.portfolio.sources import PriceHistory
from examples.symbolic import SymbolicModel
from modello import component

_Members = TypeVar("_Members", bound=EquityHoldings)
_Declared = TypeVar("_Declared", bound=EquityHoldings)
_Portfolio = TypeVar("_Portfolio", bound="NamedPortfolio[Any]")


class _EquityData(Protocol):
    @property
    def equity(self) -> Equity: ...


_Data = TypeVar("_Data", bound=_EquityData)


class NamedPortfolio(SymbolicModel, Generic[_Members]):
    """Common queries for portfolios built from an equity member declaration."""

    basket: EquityBasket[_Members]
    cash: USDCash
    nav: Expr

    @property
    def positions(self) -> NamedMap[EquityHolding]:
        return self.basket.positions

    @property
    def values(self) -> NamedMap[Expr]:
        return self.basket.values

    @property
    def weights(self) -> NamedMap[Expr]:
        return self.basket.weights

    @property
    def instruments(self) -> NamedMap[Equity]:
        return self.basket.instruments

    @property
    def providers(self) -> NamedMap[EquityProvider]:
        return self.basket.providers

    @property
    def cash_weight(self) -> Expr:
        return self.cash.amount / self.nav

    @staticmethod
    @cached_definition
    def define(basket: type[EquityBasket[_Declared]]) -> type[NamedPortfolio[_Declared]]:
        basket_type = basket

        class Portfolio(NamedPortfolio[_Declared]):
            basket = component(basket_type)
            cash = component(USDCash)
            nav = basket.capital
            nav = basket.market_value + cash.amount

        return Portfolio

    def pricer(self: _Portfolio) -> ModelPlan[_Portfolio]:
        """Select price inputs for fixed holdings using the ordinary model compiler."""
        if any(instrument.currency != "USD" for instrument in self.instruments.values()):
            raise ValueError("The example requires USD equities and cash")
        return self.compile(
            inputs=tuple(position.price for position in self.positions.values()),
            properties=(*self.weights.values(), self.cash_weight),
        )

    def match_market(self, data: NamedMap[_Data]) -> NamedMap[_Data]:
        """Match holding names and full instrument identities before using observations."""

        def checked(instrument: Equity, observation: _Data) -> _Data:
            if instrument.currency != "USD":
                raise ValueError("The example requires USD equities and cash")
            if observation.equity != instrument:
                raise ValueError(f"Price instrument must be {instrument.symbol}")
            return observation

        return self.instruments.combine(data, checked)

    def load_market(self) -> NamedMap[PriceHistory]:
        """Explicitly acquire the configured market data for this portfolio."""
        return load_providers(self.providers)


@dataclass(frozen=True)
class MarketSnapshot:
    """One valuation time and identified quotes, keyed only by declared holding name."""

    at: datetime
    prices: NamedMap[EquityPrice]

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", utc_time(self.at))

    @classmethod
    def sample(cls, market: NamedMap[PriceHistory], *, at: datetime, max_price_age: timedelta) -> MarketSnapshot:
        """Sample causal histories while retaining their provider instrument identities."""
        at = utc_time(at)
        if max_price_age < timedelta(0):
            raise ValueError("Price age must be nonnegative")
        return cls(at, market.map(lambda history: EquityPrice(history.equity, history.mark(at, max_age=max_price_age))))

    def knowns(self, model: NamedPortfolio[Any]) -> tuple[Known[float], ...]:
        """Bind observed quotes to the supplied case's declared position prices."""
        quotes = model.match_market(self.prices)
        return tuple(
            model.positions.combine(quotes, lambda position, quote: Known(position.price, quote.price)).values()
        )


@dataclass(frozen=True)
class NamedMarketData:
    """An observation axis and identified price vectors on the holding domain."""

    times: tuple[date, ...]
    prices: NamedMap[EquityPrices]

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", tuple(self.times))
        if len(self.times) < 2 or any(left >= right for left, right in zip(self.times, self.times[1:])):
            raise ValueError("Portfolio paths need at least two strictly increasing observation dates")
        if any(prices.close.times != self.times for prices in self.prices.values()):
            raise ValueError("Every price vector must match the observation dates")

    def knowns(self, model: NamedPortfolio[Any]) -> tuple[Known[Series[date]], ...]:
        """Bind aligned price columns using the same domain and identity checks."""
        quotes = model.match_market(self.prices)
        return tuple(
            model.positions.combine(quotes, lambda position, quote: Known(position.price, quote.close)).values()
        )

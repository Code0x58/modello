"""Typed holdings, named portfolios, and observation bindings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Generic, Protocol, TypeVar

from sympy import Expr, S

from examples.data import NamedMap, Series, Time, load_providers, utc_time
from examples.named_members import (
    BoundMembers,
    ModelFactory,
    cached_definition,
    component_members,
    member_bindings,
    member_name,
)
from examples.portfolio.models import Equity, EquityPosition, EquityPrice, EquityPrices, USDCash
from examples.portfolio.sources import EquityProvider, PriceHistory
from examples.symbolic import Known, ModelPlan, SymbolicModel
from modello import InstanceDummy, Modello, component

equity = ModelFactory(EquityPosition, EquityProvider)  # type: ignore[type-abstract]


class EquityHoldings(BoundMembers[EquityProvider]):
    """Equity components whose provider travels with the member declaration."""

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        component_members(cls, EquityPosition)
        if any(not isinstance(provider, EquityProvider) for provider in member_bindings(cls).values()):
            raise TypeError("Equity holdings need EquityProvider bindings")


_Members = TypeVar("_Members", bound=EquityHoldings)


_Declared = TypeVar("_Declared", bound=EquityHoldings)


@dataclass(frozen=True)
class EquityHolding:
    """A typed position view with instrument identity and its allocation context."""

    model: EquityPosition
    provider: EquityProvider
    capital: Expr

    @property
    def instrument(self) -> Equity:
        return self.provider.instrument

    @property
    def quantity(self) -> Expr:
        return self.model.quantity

    @property
    def price(self) -> Expr:
        return self.model.instrument.price

    @property
    def market_value(self) -> Expr:
        return self.model.market_value

    @property
    def weight(self) -> Expr:
        return self.market_value / self.capital


class EquityBasket(Modello, Generic[_Members]):
    """Named holdings, their provider bindings, and weights in a common capital base."""

    members: _Members
    capital = InstanceDummy("capital", positive=True)
    market_value = InstanceDummy("market_value", nonnegative=True)
    weight = market_value / capital

    @property
    def providers(self) -> NamedMap[EquityProvider]:
        """Project acquisition requests from the holding declarations."""
        return member_bindings(self.members)

    @property
    def instruments(self) -> NamedMap[Equity]:
        """Derive quote identities from the configured providers."""
        return self.providers.map(lambda provider: provider.instrument)

    @property
    def positions(self) -> NamedMap[EquityHolding]:
        """Associate each symbolic holding with its instrument and capital base."""
        return component_members(self.members, EquityPosition).combine(
            self.providers, lambda model, provider: EquityHolding(model, provider, self.capital)
        )

    def holding(self, member: EquityPosition) -> EquityHolding:
        """Select a contextual holding by its declared, typed component instance."""
        return self.positions[member_name(component_members(self.members, EquityPosition), member)]

    @property
    def values(self) -> NamedMap[Expr]:
        return self.positions.map(lambda position: position.market_value)

    @property
    def weights(self) -> NamedMap[Expr]:
        """Member weights in the same capital base as the basket's weight."""
        return self.positions.map(lambda position: position.weight)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if "providers" in cls.__dict__:
            raise TypeError("Basket providers derive from the holding declarations")
        component_members(cls.members, EquityPosition)
        member_bindings(cls.members)

    @staticmethod
    def total(members: Modello) -> Expr:
        """Reduce declared positions into a symbolic value equation."""
        return sum((position.market_value for position in component_members(members, EquityPosition).values()), S.Zero)

    @staticmethod
    @cached_definition
    def define(members: type[_Declared]) -> type[EquityBasket[_Declared]]:
        """Reuse the basket definition for a runtime-supplied holding domain."""
        member_type = members

        class Basket(EquityBasket[_Declared]):
            members = component(member_type)
            market_value = EquityBasket.total(members)

        return Basket


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
class NamedMarketData(Generic[Time]):
    """An observation axis and identified price vectors on the holding domain."""

    times: tuple[Time, ...]
    prices: NamedMap[EquityPrices[Time]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "times", tuple(self.times))
        if len(self.times) < 2 or any(left >= right for left, right in zip(self.times, self.times[1:])):
            raise ValueError("Portfolio paths need at least two strictly increasing observation dates")
        if any(prices.close.times != self.times for prices in self.prices.values()):
            raise ValueError("Every price vector must match the observation dates")

    def knowns(self, model: NamedPortfolio[Any]) -> tuple[Known[Series[Time]], ...]:
        """Bind aligned price columns using the same domain and identity checks."""
        quotes = model.match_market(self.prices)
        return tuple(
            model.positions.combine(quotes, lambda position, quote: Known(position.price, quote.close)).values()
        )

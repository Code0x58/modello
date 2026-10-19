"""A symbolic basket with one named instrument binding and allocation base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from sympy import Expr, S

from examples.compute.named_map import NamedMap
from examples.named_members import component_members
from examples.portfolio.models import Equity, EquityPosition
from examples.portfolio.providers import EquityProvider
from modello import InstanceDummy, Modello, component

_Members = TypeVar("_Members", bound=Modello)


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
    providers: NamedMap[EquityProvider]
    capital: Expr
    market_value: Expr
    weight: Expr

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

    @property
    def values(self) -> NamedMap[Expr]:
        return self.positions.map(lambda position: position.market_value)

    @property
    def weights(self) -> NamedMap[Expr]:
        """Member weights in the same capital base as the basket's weight."""
        return self.positions.map(lambda position: position.weight)


def basket_model(members: type[_Members], providers: NamedMap[EquityProvider]) -> type[EquityBasket[_Members]]:
    """Bind explicitly declared positions once, then derive the basket's formulas."""
    member_type = members
    bindings = providers
    # Validate before compiling, including when every holding is still symbolic.
    declared = component_members(member_type, EquityPosition)
    bindings = declared.combine(bindings, lambda position, provider: provider)
    if any(not isinstance(provider, EquityProvider) for provider in bindings.values()):
        raise TypeError("Basket providers must be EquityProvider instances")

    class Basket(EquityBasket[_Members]):
        members = component(member_type)
        providers = bindings
        capital = InstanceDummy("capital", positive=True)
        market_value = sum(
            (position.market_value for position in component_members(members, EquityPosition).values()), S.Zero
        )
        weight = market_value / capital

    return Basket

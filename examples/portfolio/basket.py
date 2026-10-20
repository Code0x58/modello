"""A symbolic basket with one named instrument binding and allocation base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from sympy import Expr, S

from examples.compute.named_map import NamedMap
from examples.named_members import BoundMembers, ModelFactory, cached_definition, component_members, member_bindings
from examples.portfolio.models import Equity, EquityPosition
from examples.portfolio.providers import EquityProvider
from modello import InstanceDummy, Modello, component

# Binding interfaces are checked with isinstance, never instantiated.
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
        for holding in self.positions.values():
            if holding.model is member:
                return holding
        raise ValueError("Select a member of this basket's model case")

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

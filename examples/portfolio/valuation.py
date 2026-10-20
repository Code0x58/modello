"""Typed portfolio cases and paths evaluated from their existing model structure."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Generic, TypeVar

from sympy import Expr, Symbol

from examples.compute.named_map import NamedMap
from examples.compute.providers import load_providers
from examples.compute.series import Series, utc_time
from examples.model_plan import Known, ModelPath, ModelPlan
from examples.named_members import cached_definition
from examples.portfolio.analytics import NamedPortfolioPath, PositionPath
from examples.portfolio.basket import EquityBasket, EquityHolding, EquityHoldings
from examples.portfolio.models import Equity, EquityPrice, EquityPrices, USDCash
from examples.portfolio.providers import EquityProvider
from examples.portfolio.sources import PriceHistory
from examples.symbolic import SymbolicModel
from modello import SymbolicScalar, component

_Members = TypeVar("_Members", bound=EquityHoldings)
_Declared = TypeVar("_Declared", bound=EquityHoldings)
_Portfolio = TypeVar("_Portfolio", bound="NamedPortfolio[Any]")


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

    def compile(self: _Portfolio, *, inputs: tuple[SymbolicScalar, ...] | None = None) -> ModelPlan[_Portfolio]:
        """Compile a complete typed model case, using its remaining symbolic fields."""
        return ModelPlan(self, inputs=inputs, properties=(*self.weights.values(), self.cash_weight))

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

    def pricer(self: _Portfolio) -> NamedPortfolioPricer[_Portfolio]:
        """Bind a fixed portfolio's prices to identified market observations."""
        return NamedPortfolioPricer(self)

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


class NamedPortfolioPricer(Generic[_Portfolio]):
    """Evaluate into the same concrete portfolio type, including its child models."""

    def __init__(self, portfolio: _Portfolio) -> None:
        self._positions = portfolio.positions
        self._instruments = portfolio.instruments
        if any(instrument.currency != "USD" for instrument in self._instruments.values()):
            raise ValueError("The example requires USD equities and cash")
        prices = self._positions.map(lambda position: position.price)
        if not all(isinstance(price, Symbol) for price in prices.values()) or len(set(prices.values())) != len(prices):
            raise ValueError("Leave each instrument price as an independent symbolic input")
        if (
            any(position.quantity.free_symbols for position in self._positions.values())
            or portfolio.cash.amount.free_symbols
        ):
            raise ValueError("Specify all position quantities and the cash balance before compiling")
        self.view = portfolio.compile(inputs=tuple(prices.values()))
        self._positions = self.view.source.positions
        self._instruments = self.view.source.instruments

    def snapshot(self, market: NamedMap[PriceHistory], *, at: datetime, max_price_age: timedelta) -> MarketSnapshot:
        """Sample causal, identified quotes on this portfolio's declared domain."""
        at = utc_time(at)
        if max_price_age < timedelta(0):
            raise ValueError("Price age must be nonnegative")

        def quote(instrument: Equity, history: PriceHistory) -> EquityPrice:
            if history.equity != instrument:
                raise ValueError(f"Price instrument must be {instrument.symbol}")
            return EquityPrice(instrument, history.mark(at, max_age=max_price_age))

        return MarketSnapshot(at, self._instruments.combine(market, quote))

    def evaluate(self, snapshot: MarketSnapshot) -> _Portfolio:
        """Return the original model shape, with numeric scalar fields."""
        if not isinstance(snapshot, MarketSnapshot):
            raise TypeError("Supply a MarketSnapshot with identified quotes")

        def price(position: EquityHolding, quote: EquityPrice) -> Known[float]:
            if quote.equity != position.instrument:
                raise ValueError(f"Price instrument must be {position.instrument.symbol}")
            return Known(position.price, quote.price)

        knowns = self._positions.combine(snapshot.prices, price)
        result = self.view.evaluate(*knowns.values())
        if float(result.nav) <= 0:
            raise ValueError("Portfolio value must stay strictly positive")
        return result

    def at(self, market: NamedMap[PriceHistory], *, at: datetime, max_price_age: timedelta) -> _Portfolio:
        return self.evaluate(self.snapshot(market, at=at, max_price_age=max_price_age))

    def evaluate_over(self, data: NamedMarketData) -> ModelPath[_Portfolio, date]:
        """Keep the model schema for column selection, traversal, and numeric cases."""

        def checked_prices(position: EquityHolding, prices: EquityPrices) -> Known[Series[date]]:
            if position.instrument != prices.equity:
                raise ValueError(f"Price instrument must be {position.instrument.symbol}")
            return Known(position.price, prices.close)

        return self.view.over(data.times, *self._positions.combine(data.prices, checked_prices).values())

    def over(self, data: NamedMarketData) -> NamedPortfolioPath[date]:
        """Add interval return attribution to the same structured valuation columns."""
        columns = self.evaluate_over(data)
        model = self.view.source
        nav = columns.column(model.nav)
        if any(value <= 0 for value in nav.values):
            raise ValueError("Portfolio value must stay strictly positive")

        def position(name: str) -> PositionPath[date]:
            holding = model.positions[name]
            value = columns.column(holding.market_value)
            weight = columns.column(holding.weight)
            returns = columns.column(holding.price).simple_returns()
            contribution = Series(
                returns.times, tuple(w * r for w, r in zip(weight.values[:-1], returns.values, strict=True))
            )
            return PositionPath(value, weight, returns, contribution)

        return NamedPortfolioPath(
            NamedMap((name, position(name)) for name in model.positions),
            columns.column(model.cash.amount),
            columns.column(model.cash_weight),
            nav,
        )

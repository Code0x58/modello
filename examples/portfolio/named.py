"""Portfolio computation over named members, using synthetic observations.

Run ``python -m examples.portfolio.named`` for the worked example.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cache
from typing import Generic, TypeVar

from sympy import Expr, Symbol

from examples.compute.named_map import NamedMap
from examples.compute.providers import load_providers
from examples.compute.scalar_plan import ScalarPlan
from examples.compute.series import Series
from examples.portfolio.analytics import NamedPortfolioPath, PositionPath
from examples.portfolio.basket import EquityBasket, EquityHolding
from examples.portfolio.models import GOOGL, Equity, EquityPosition, EquityPrices, USDCash
from examples.portfolio.providers import DummyEquityProvider, EquityProvider
from examples.portfolio.sources import PriceHistory
from examples.symbolic import SymbolicModel, ViewFields
from modello import Modello, component

_Members = TypeVar("_Members", bound=Modello)
_Declared = TypeVar("_Declared", bound=Modello)


class EquityMembers(Modello):
    """A declared domain of two named equity positions."""

    aapl = component(EquityPosition)
    msft = component(EquityPosition)


class NamedPortfolio(SymbolicModel, Generic[_Members]):
    """Common queries for portfolios built from an equity member declaration."""

    basket: EquityBasket[_Members]
    cash: USDCash
    nav: Expr

    @property
    def positions(self) -> NamedMap[EquityHolding]:
        """Expose positions with a shared value type and their declared names."""
        return self.basket.positions

    @property
    def values(self) -> NamedMap[Expr]:
        """Project each position to its market value."""
        return self.basket.values

    @property
    def weights(self) -> NamedMap[Expr]:
        """Project market values to equity weights on the same named domain."""
        return self.basket.weights

    @property
    def instruments(self) -> NamedMap[Equity]:
        """Bind each holding name to its instrument's provider and symbol."""
        return self.basket.instruments

    @property
    def providers(self) -> NamedMap[EquityProvider]:
        return self.basket.providers

    @property
    def cash_weight(self) -> Expr:
        return self.cash.amount / self.nav

    @property
    def valuation(self) -> NamedMap[Expr]:
        """Use namespaced outputs so member names cannot collide with totals."""
        return NamedMap(
            [
                *((f"value:{name}", value) for name, value in self.values.items()),
                *((f"weight:{name}", weight) for name, weight in self.weights.items()),
                ("cash", self.cash.amount),
                ("cash_weight", self.cash_weight),
                ("nav", self.nav),
            ]
        )

    def compile(self, *, inputs: ViewFields | None = None, outputs: ViewFields | None = None) -> ScalarPlan:
        """Compile this case's valuation expressions for selected remaining inputs."""
        return super().compile(inputs=inputs, outputs=self.valuation if outputs is None else outputs)

    @classmethod
    @cache
    def plan(cls) -> ScalarPlan:
        """The account view varies quantities, prices, and cash in the same model."""
        model = cls("valuation")
        return model.compile(
            inputs=NamedMap(
                [
                    *((f"price:{name}", position.price) for name, position in model.positions.items()),
                    *((f"quantity:{name}", position.quantity) for name, position in model.positions.items()),
                    ("cash", model.cash.amount),
                ]
            )
        )

    @staticmethod
    def define(basket: type[EquityBasket[_Declared]]) -> type[NamedPortfolio[_Declared]]:
        """Compose a supplied basket with cash; account NAV is its capital base."""
        basket_type = basket

        class Portfolio(NamedPortfolio[_Declared]):
            basket = component(basket_type)
            cash = component(USDCash)
            nav = basket.capital
            nav = basket.market_value + cash.amount

        return Portfolio

    def pricer(self) -> NamedPortfolioPricer[_Members]:
        """Compile a valuation view with known quantities and remaining price inputs."""
        return NamedPortfolioPricer(self)

    def load_market(self) -> NamedMap[PriceHistory]:
        """Explicitly acquire the configured market data for this portfolio."""
        return load_providers(self.providers)


@dataclass(frozen=True)
class NamedMarketData:
    """An observation axis and price vectors identified by member name."""

    times: tuple[date, ...]
    prices: NamedMap[EquityPrices]

    def __post_init__(self) -> None:
        """Keep a common axis, including when the equity domain is empty."""
        object.__setattr__(self, "times", tuple(self.times))
        if len(self.times) < 2 or any(left >= right for left, right in zip(self.times, self.times[1:])):
            raise ValueError("Portfolio paths need at least two strictly increasing observation dates")
        if any(prices.close.times != self.times for prices in self.prices.values()):
            raise ValueError("Every price vector must match the observation dates")


class NamedPortfolioPricer(Generic[_Members]):
    """Compile all named equity values and reuse the plan on aligned price data."""

    def __init__(self, portfolio: NamedPortfolio[_Members]) -> None:
        """Bind known quantities, cash, member names, and instrument identities."""
        self._instruments = portfolio.instruments
        if any(instrument.currency != "USD" for instrument in self._instruments.values()):
            raise ValueError("The example requires USD equities and cash")
        prices = portfolio.positions.map(lambda position: position.price)
        if not all(isinstance(price, Symbol) for price in prices.values()) or len(set(prices.values())) != len(prices):
            raise ValueError("Leave each instrument price as an independent symbolic input")
        outputs = (*portfolio.values.values(), portfolio.cash.amount, portfolio.nav)
        if set().union(*(expression.free_symbols for expression in outputs)) - set(prices.values()):
            raise ValueError("Specify all position quantities and the cash balance before compiling")
        self._plan = portfolio.compile(inputs=prices)

    def over(self, data: NamedMarketData) -> NamedPortfolioPath[date]:
        """Align data by name before constructing the numerical input layout."""

        def checked_prices(instrument: Equity, prices: EquityPrices) -> Series[date]:
            if instrument != prices.equity:
                raise ValueError(f"Price instrument must be {instrument.symbol}")
            return prices.close

        prices = self._instruments.combine(data.prices, checked_prices)
        outputs = self._plan.over(data.times, prices)

        nav = outputs["nav"]
        if any(value <= 0 for value in nav.values):
            raise ValueError("Portfolio value must stay strictly positive")

        def position(name: str) -> PositionPath[date]:
            value = outputs[f"value:{name}"]
            weight = outputs[f"weight:{name}"]
            returns = prices[name].simple_returns()
            contribution = Series(
                returns.times,
                tuple(w * r for w, r in zip(weight.values[:-1], returns.values, strict=True)),
            )
            return PositionPath(value, weight, returns, contribution)

        cash = outputs["cash"]
        return NamedPortfolioPath(
            NamedMap((name, position(name)) for name in prices),
            cash,
            outputs["cash_weight"],
            nav,
        )


class ThreeEquities(EquityMembers):
    """Add one generic position; aggregates and projected maps follow automatically."""

    googl = component(EquityPosition)


EQUITY_PROVIDERS = NamedMap({"aapl": DummyEquityProvider(symbol="AAPL"), "msft": DummyEquityProvider(symbol="MSFT")})


class Basket(EquityBasket[EquityMembers]):
    """A declared equity basket with provider bindings and an explicit reduction."""

    members = component(EquityMembers)
    providers = EQUITY_PROVIDERS
    market_value = EquityBasket.total(members)


class ThreeEquityBasket(EquityBasket[ThreeEquities]):
    """The same basket equations over an extended named holding domain."""

    members = component(ThreeEquities)
    providers = NamedMap({**EQUITY_PROVIDERS, "googl": DummyEquityProvider(symbol="GOOGL")})
    market_value = EquityBasket.total(members)


def example_named_portfolio() -> NamedPortfolio[ThreeEquities]:
    """Three declared equities with fixed quantities and USD cash."""
    return NamedPortfolio.define(ThreeEquityBasket)(
        "holdings",
        basket=ThreeEquityBasket(
            "basket",
            members=ThreeEquities(
                "equities",
                aapl=EquityPosition("apple", quantity=10),
                msft=EquityPosition("microsoft", quantity=20),
                googl=EquityPosition("alphabet", quantity=1),
            ),
        ),
        cash=USDCash("cash", amount=100),
    )


def example_named_market_data() -> NamedMarketData:
    """Synthetic price vectors for all three declared members."""
    from examples.portfolio.explicit import MarketData

    data = MarketData.example()
    return NamedMarketData(
        data.aapl.close.times,
        NamedMap(
            {
                "aapl": data.aapl,
                "msft": data.msft,
                "googl": EquityPrices(GOOGL, Series(data.aapl.close.times, (1000, 1005, 990, 1010, 1020))),
            }
        ),
    )


def main() -> None:
    """Show the same named domain carried through valuation and risk."""
    path = example_named_portfolio().pricer().over(example_named_market_data())
    result = path.analytics(periods_per_year=252)
    print("Named equity members:", ", ".join(path.positions))
    print("NAV:", path.nav.values)
    print("Latest equity weights:", dict(path.weights.map(lambda series: series.values[-1])))
    print("Latest cash weight:", path.cash_weight.values[-1])
    print("Total price return:", f"{result.performance.total_return:.4%}")
    print("Latest-weights volatility:", f"{result.latest_weights_risk.volatility:.4%}")
    print("Volatility contributions:", dict(result.latest_weights_risk.contributions))


if __name__ == "__main__":
    main()

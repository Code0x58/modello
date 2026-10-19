"""Portfolio computation over named members, using synthetic observations.

Run ``python -m examples.named_portfolio`` from a checkout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Generic, Protocol, TypeVar

from sympy import Expr, Symbol

from examples.named_members import NamedMap, component_members
from examples.portfolio import (
    ApplePosition,
    Equity,
    EquityPosition,
    EquityPrices,
    MicrosoftPosition,
    Series,
    USDCash,
    example_market_data,
)
from examples.portfolio_analytics import NamedAnalytics as NamedAnalytics
from examples.portfolio_analytics import NamedPortfolioPath, PositionPath
from examples.portfolio_analytics import NamedRisk as NamedRisk
from examples.scalar_plan import ScalarPlan
from modello import InstanceDummy, Modello, component

_Members = TypeVar("_Members", bound=Modello)


class EquityMembers(Modello):
    """A declared domain of two named equity positions."""

    aapl = component(ApplePosition)
    msft = component(MicrosoftPosition)


class NamedPortfolio(Modello, Generic[_Members]):
    """Common queries for portfolios built from an equity member declaration."""

    equities: _Members
    cash: USDCash
    nav: Expr

    @property
    def positions(self) -> NamedMap[EquityPosition]:
        """Expose positions with a shared value type and their declared names."""
        return component_members(self.equities, EquityPosition)

    @property
    def values(self) -> NamedMap[Expr]:
        """Project each position to its market value."""
        return self.positions.map(lambda position: position.market_value)

    @property
    def weights(self) -> NamedMap[Expr]:
        """Project market values to equity weights on the same named domain."""
        return self.values.map(lambda value: value / self.nav)


def portfolio_model(equity_members: type[_Members]) -> type[NamedPortfolio[_Members]]:
    """Compile a member declaration into a portfolio's scalar equations."""

    class Portfolio(NamedPortfolio[_Members]):
        equities = component(equity_members)
        cash = component(USDCash)
        nav = InstanceDummy("nav", positive=True)
        nav = (
            sum(position.market_value for position in component_members(equities, EquityPosition).values())
            + cash.amount
        )

    return Portfolio


@dataclass(frozen=True)
class NamedMarketData:
    """An observation axis and price vectors identified by member name."""

    dates: tuple[date, ...]
    prices: NamedMap[EquityPrices[Equity]]

    def __post_init__(self) -> None:
        """Keep a common axis, including when the equity domain is empty."""
        object.__setattr__(self, "dates", tuple(self.dates))
        if len(self.dates) < 2 or any(left >= right for left, right in zip(self.dates, self.dates[1:])):
            raise ValueError("Portfolio paths need at least two strictly increasing observation dates")
        if any(prices.close.dates != self.dates for prices in self.prices.values()):
            raise ValueError("Every price vector must match the observation dates")


class PortfolioView(Protocol):
    """The symbolic inputs required by the compiled portfolio calculation."""

    @property
    def positions(self) -> NamedMap[EquityPosition]: ...

    @property
    def values(self) -> NamedMap[Expr]: ...

    @property
    def cash(self) -> USDCash: ...

    @property
    def nav(self) -> Expr: ...


def valuation_outputs(portfolio: PortfolioView) -> NamedMap[Expr]:
    """Use namespaced outputs so member names cannot collide with totals."""
    return NamedMap(
        [
            *((f"value:{name}", value) for name, value in portfolio.values.items()),
            ("cash", portfolio.cash.amount),
            ("nav", portfolio.nav),
        ]
    )


class NamedPortfolioPricer:
    """Compile all named equity values and reuse the plan on aligned price data."""

    def __init__(self, portfolio: PortfolioView) -> None:
        """Bind known quantities, cash, member names, and instrument identities."""
        self._instruments = portfolio.positions.map(lambda position: type(position.instrument))
        if any(instrument.currency != "USD" for instrument in self._instruments.values()):
            raise ValueError("The example requires USD equities and cash")
        prices = portfolio.positions.map(lambda position: position.instrument.price)
        if not all(isinstance(price, Symbol) for price in prices.values()) or len(set(prices.values())) != len(prices):
            raise ValueError("Leave each instrument price as an independent symbolic input")
        outputs = (*portfolio.values.values(), portfolio.cash.amount, portfolio.nav)
        if set().union(*(expression.free_symbols for expression in outputs)) - set(prices.values()):
            raise ValueError("Specify all position quantities and the cash balance before compiling")
        self._plan = ScalarPlan(prices, valuation_outputs(portfolio))

    def __call__(self, data: NamedMarketData) -> NamedPortfolioPath:
        """Align data by name before constructing the numerical input layout."""

        def checked_prices(instrument: type[Equity], prices: EquityPrices[Equity]) -> Series:
            if instrument is not prices.equity:
                raise ValueError(f"Price instrument must be {instrument.ticker}")
            return prices.close

        prices = self._instruments.combine(data.prices, checked_prices)
        outputs = self._plan.over(data.dates, prices)

        nav = outputs["nav"]
        if any(value <= 0 for value in nav.values):
            raise ValueError("Portfolio value must stay strictly positive")

        def position(name: str) -> PositionPath:
            value = outputs[f"value:{name}"]
            weight = Series(data.dates, tuple(v / n for v, n in zip(value.values, nav.values, strict=True)))
            returns = prices[name].simple_returns()
            contribution = Series(
                returns.dates,
                tuple(w * r for w, r in zip(weight.values[:-1], returns.values, strict=True)),
            )
            return PositionPath(value, weight, returns, contribution)

        cash = outputs["cash"]
        return NamedPortfolioPath(
            NamedMap((name, position(name)) for name in prices),
            cash,
            Series(data.dates, tuple(c / n for c, n in zip(cash.values, nav.values, strict=True))),
            nav,
        )


class Alphabet(Equity):
    """A third equity type for extending the declared member domain."""

    ticker = "GOOGL"


class AlphabetPosition(EquityPosition):
    """A named holding in Alphabet equity."""

    instrument = component(Alphabet)


class ThreeEquities(EquityMembers):
    """Add one declaration; aggregates and projected maps follow automatically."""

    googl = component(AlphabetPosition)


def example_named_portfolio() -> NamedPortfolio[ThreeEquities]:
    """Three declared equities with fixed quantities and USD cash."""
    return portfolio_model(ThreeEquities)(
        "holdings",
        equities=ThreeEquities(
            "equities",
            aapl=ApplePosition("apple", quantity=10),
            msft=MicrosoftPosition("microsoft", quantity=20),
            googl=AlphabetPosition("alphabet", quantity=1),
        ),
        cash=USDCash("cash", amount=100),
    )


def example_named_market_data() -> NamedMarketData:
    """Synthetic price vectors for all three declared members."""
    data = example_market_data()
    return NamedMarketData(
        data.aapl.close.dates,
        NamedMap(
            {
                "aapl": data.aapl,
                "msft": data.msft,
                "googl": EquityPrices(Alphabet, Series(data.aapl.close.dates, (1000, 1005, 990, 1010, 1020))),
            }
        ),
    )


def main() -> None:
    """Show the same named domain carried through valuation and risk."""
    path = NamedPortfolioPricer(example_named_portfolio())(example_named_market_data())
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

"""Portfolio computation over named members, using synthetic observations.

Run ``python -m examples.named_portfolio`` from a checkout.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from math import sqrt
from statistics import covariance, mean, variance
from typing import Generic, TypeVar, cast

from sympy import Expr, Symbol, lambdify

from examples.named_members import NamedMap, component_members
from examples.portfolio import (
    ApplePosition,
    Equity,
    EquityPosition,
    EquityPrices,
    MicrosoftPosition,
    PositionPath,
    Series,
    USDCash,
    example_market_data,
)
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


@dataclass(frozen=True)
class NamedRisk:
    """Fixed-reference-weight volatility and its named equity contributions."""

    volatility: float
    contributions: NamedMap[float]


@dataclass(frozen=True)
class NamedAnalytics:
    """Path statistics and covariance indexed by two member-name axes."""

    total_return: float
    mean_period_return: float
    annualized_volatility: float
    maximum_drawdown: float
    annualized_covariance: NamedMap[NamedMap[float]]
    latest_weights_risk: NamedRisk


@dataclass(frozen=True)
class NamedPortfolioPath:
    """Named equity paths with a common observation axis and USD cash."""

    positions: NamedMap[PositionPath]
    cash_value: Series
    cash_weight: Series
    nav: Series

    @property
    def values(self) -> NamedMap[Series]:
        """Expose value vectors using the same position names."""
        return self.positions.map(lambda position: position.value)

    @property
    def weights(self) -> NamedMap[Series]:
        """Expose weight vectors using the same position names."""
        return self.positions.map(lambda position: position.weight)

    def analytics(self, *, periods_per_year: int) -> NamedAnalytics:
        """Reduce time observations while retaining the member domain."""
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer")
        returns = self.nav.simple_returns().values
        if len(returns) < 2:
            raise ValueError("Sample risk estimates need at least three price observations")
        asset_returns = self.positions.map(lambda position: position.returns.values)
        cov = asset_returns.map(
            lambda left: asset_returns.map(lambda right: covariance(left, right) * periods_per_year)
        )
        weights = self.weights.map(lambda series: series.values[-1])
        weighted_returns = weights.combine(
            asset_returns, lambda weight, series: tuple(weight * value for value in series)
        )
        fixed_returns = tuple(
            sum(series[index] for series in weighted_returns.values()) for index in range(len(returns))
        )
        volatility = sqrt(variance(fixed_returns) * periods_per_year)
        contributions = weights.combine(
            cov,
            lambda weight, row: (
                weight * sum(row.combine(weights, lambda c, w: c * w).values()) / volatility if volatility else 0.0
            ),
        )
        return NamedAnalytics(
            self.nav.values[-1] / self.nav.values[0] - 1,
            mean(returns),
            sqrt(variance(returns) * periods_per_year),
            -min(self.nav.drawdowns().values),
            cov,
            NamedRisk(volatility, contributions),
        )


class NamedPortfolioPricer:
    """Compile all named equity values and reuse the plan on aligned price data."""

    def __init__(self, portfolio: NamedPortfolio[_Members]) -> None:
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
        self._calculate = cast(
            Callable[..., tuple[float, ...]],
            lambdify(tuple(prices.values()), outputs, modules="math", cse=True, docstring_limit=0),
        )

    def __call__(self, data: NamedMarketData) -> NamedPortfolioPath:
        """Align data by name before constructing the numerical input layout."""

        def checked_prices(instrument: type[Equity], prices: EquityPrices[Equity]) -> Series:
            if instrument is not prices.equity:
                raise ValueError(f"Price instrument must be {instrument.ticker}")
            return prices.close

        prices = self._instruments.combine(data.prices, checked_prices)
        rows = tuple(
            self._calculate(*(series.values[index] for series in prices.values())) for index in range(len(data.dates))
        )

        def column(index: int) -> Series:
            return Series(data.dates, tuple(row[index] for row in rows))

        nav = column(len(prices) + 1)
        if any(value <= 0 for value in nav.values):
            raise ValueError("Portfolio value must stay strictly positive")

        def position(name: str, index: int) -> PositionPath:
            value = column(index)
            weight = Series(data.dates, tuple(v / n for v, n in zip(value.values, nav.values, strict=True)))
            returns = prices[name].simple_returns()
            contribution = Series(
                returns.dates,
                tuple(w * r for w, r in zip(weight.values[:-1], returns.values, strict=True)),
            )
            return PositionPath(value, weight, returns, contribution)

        cash = column(len(prices))
        return NamedPortfolioPath(
            NamedMap((name, position(name, index)) for index, name in enumerate(prices)),
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
    print("Total price return:", f"{result.total_return:.4%}")
    print("Latest-weights volatility:", f"{result.latest_weights_risk.volatility:.4%}")
    print("Volatility contributions:", dict(result.latest_weights_risk.contributions))


if __name__ == "__main__":
    main()

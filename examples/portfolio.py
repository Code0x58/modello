"""An explicit equity portfolio evaluated over synthetic price vectors.

Run ``python -m examples.portfolio`` from a checkout for the worked example.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from math import isfinite, sqrt
from statistics import covariance, mean, variance
from typing import ClassVar, Generic, TypeVar, cast

from sympy import ImmutableMatrix, Symbol, lambdify

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


class Portfolio(Modello):
    """Two named equity positions and cash, held without trading."""

    aapl = component(ApplePosition)
    msft = component(MicrosoftPosition)
    cash = component(USDCash)
    nav = InstanceDummy("nav", positive=True)
    nav = aapl.market_value + msft.market_value + cash.amount
    aapl_weight = aapl.market_value / nav
    msft_weight = msft.market_value / nav
    cash_weight = cash.amount / nav


@dataclass(frozen=True)
class Series:
    """A finite vector on a strictly increasing observation-date axis."""

    dates: tuple[date, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Own immutable data and validate its axis before doing arithmetic."""
        object.__setattr__(self, "dates", tuple(self.dates))
        object.__setattr__(self, "values", tuple(float(value) for value in self.values))
        if not self.dates or len(self.dates) != len(self.values):
            raise ValueError("A series needs one value per observation date")
        if any(left >= right for left, right in zip(self.dates, self.dates[1:])):
            raise ValueError("Observation dates must be unique and strictly increasing")
        if not all(isfinite(value) for value in self.values):
            raise ValueError("Series values must be finite")

    def as_vector(self) -> ImmutableMatrix:
        """Expose the observations as an immutable column vector."""
        return ImmutableMatrix(self.values)

    def simple_returns(self) -> Series:
        """Label each price ratio minus one with its interval's ending date."""
        if len(self.values) < 2 or any(value <= 0 for value in self.values):
            raise ValueError("Returns need at least two strictly positive observations")
        return Series(
            self.dates[1:],
            tuple(end / start - 1 for start, end in zip(self.values, self.values[1:])),
        )

    def drawdowns(self) -> Series:
        """Measure the fraction below the running peak, including the initial value."""
        if any(value <= 0 for value in self.values):
            raise ValueError("Drawdowns need strictly positive observations")
        peak = self.values[0]
        values = []
        for value in self.values:
            peak = max(peak, value)
            values.append(value / peak - 1)
        return Series(self.dates, tuple(values))


_EquityT = TypeVar("_EquityT", bound=Equity, covariant=True)


@dataclass(frozen=True)
class EquityPrices(Generic[_EquityT]):
    """An instrument's positive closing prices on a dated observation axis."""

    equity: type[_EquityT]
    close: Series

    def __post_init__(self) -> None:
        """Keep invalid quotes out of valuation and return calculations."""
        if any(value <= 0 for value in self.close.values):
            raise ValueError("Equity prices must be strictly positive")


@dataclass(frozen=True)
class MarketData:
    """The explicit portfolio's two aligned price vectors."""

    aapl: EquityPrices[AAPL]
    msft: EquityPrices[MSFT]

    def __post_init__(self) -> None:
        """Require instrument identity and exact date alignment."""
        if self.aapl.equity is not AAPL or self.msft.equity is not MSFT:
            raise ValueError("AAPL and MSFT price histories must match their named fields")
        if self.aapl.close.dates != self.msft.close.dates:
            raise ValueError("Price histories must use the same observation dates")


@dataclass(frozen=True)
class PositionPath:
    """Value, drifting weight, and price-return attribution for one position."""

    value: Series
    weight: Series
    returns: Series
    return_contribution: Series


@dataclass(frozen=True)
class EquityCovariance:
    """Annualized sample covariance, with explicit AAPL and MSFT identities."""

    aapl_variance: float
    cross_covariance: float
    msft_variance: float

    def as_matrix(self) -> ImmutableMatrix:
        """Return a matrix whose rows and columns are ordered AAPL, MSFT."""
        return ImmutableMatrix(
            [[self.aapl_variance, self.cross_covariance], [self.cross_covariance, self.msft_variance]]
        )


@dataclass(frozen=True)
class WeightRisk:
    """Annualized volatility and Euler contributions at fixed reference weights."""

    volatility: float
    aapl_contribution: float
    msft_contribution: float
    cash_contribution: float = 0.0


@dataclass(frozen=True)
class Analytics:
    """Observed path statistics and a separate latest-weights risk estimate."""

    total_return: float
    mean_period_return: float
    annualized_volatility: float
    maximum_drawdown: float
    annualized_covariance: EquityCovariance
    latest_weights_risk: WeightRisk


@dataclass(frozen=True)
class PortfolioPath:
    """The evaluated portfolio, retaining named positions and their common axis."""

    aapl: PositionPath
    msft: PositionPath
    cash_value: Series
    cash_weight: Series
    nav: Series

    def analytics(self, *, periods_per_year: int) -> Analytics:
        """Use sample statistics; annualize variance by the supplied frequency."""
        if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer")
        returns = self.nav.simple_returns().values
        if len(returns) < 2:
            raise ValueError("Sample risk estimates need at least three price observations")
        apple = self.aapl.returns.values
        microsoft = self.msft.returns.values
        cov = EquityCovariance(
            variance(apple) * periods_per_year,
            covariance(apple, microsoft) * periods_per_year,
            variance(microsoft) * periods_per_year,
        )
        wa = self.aapl.weight.values[-1]
        wm = self.msft.weight.values[-1]
        # This scenario holds the latest weights fixed over the sample. Its risk
        # differs from the actual buy-and-hold path, whose weights drift daily.
        fixed_returns = tuple(wa * a + wm * m for a, m in zip(apple, microsoft, strict=True))
        volatility = sqrt(variance(fixed_returns) * periods_per_year)
        risk = WeightRisk(
            volatility,
            wa * (cov.aapl_variance * wa + cov.cross_covariance * wm) / volatility if volatility else 0.0,
            wm * (cov.cross_covariance * wa + cov.msft_variance * wm) / volatility if volatility else 0.0,
        )
        return Analytics(
            self.nav.values[-1] / self.nav.values[0] - 1,
            mean(returns),
            sqrt(variance(returns) * periods_per_year),
            -min(self.nav.drawdowns().values),
            cov,
            risk,
        )


class PortfolioPricer:
    """Compile a scalar model once, then evaluate its formulas over price vectors."""

    def __init__(self, portfolio: Portfolio) -> None:
        """Freeze known holdings and cash into a reusable numerical calculation."""
        prices = (portfolio.aapl.instrument.price, portfolio.msft.instrument.price)
        if not all(isinstance(price, Symbol) for price in prices) or prices[0] == prices[1]:
            raise ValueError("Leave each instrument price as an independent symbolic input")
        outputs = (
            portfolio.aapl.market_value,
            portfolio.msft.market_value,
            portfolio.cash.amount,
            portfolio.nav,
            portfolio.aapl_weight,
            portfolio.msft_weight,
            portfolio.cash_weight,
        )
        if set().union(*(expression.free_symbols for expression in outputs)) - set(prices):
            raise ValueError("Specify all position quantities and the cash balance before compiling")
        self._calculate = cast(
            Callable[[float, float], tuple[float, ...]],
            lambdify(prices, outputs, modules="math", cse=True, docstring_limit=0),
        )

    def __call__(self, data: MarketData) -> PortfolioPath:
        """Evaluate aligned observations without solving the model again."""
        dates = data.aapl.close.dates
        apple_returns = data.aapl.close.simple_returns()
        microsoft_returns = data.msft.close.simple_returns()
        rows = tuple(
            self._calculate(apple, microsoft)
            for apple, microsoft in zip(data.aapl.close.values, data.msft.close.values, strict=True)
        )

        def column(index: int) -> Series:
            return Series(dates, tuple(row[index] for row in rows))

        def position(value: Series, weight: Series, returns: Series) -> PositionPath:
            contribution = Series(
                returns.dates,
                tuple(w * r for w, r in zip(weight.values[:-1], returns.values, strict=True)),
            )
            return PositionPath(value, weight, returns, contribution)

        nav = column(3)
        if any(value <= 0 for value in nav.values):
            raise ValueError("Portfolio value must stay strictly positive")
        return PortfolioPath(
            position(column(0), column(4), apple_returns),
            position(column(1), column(5), microsoft_returns),
            column(2),
            column(6),
            nav,
        )


def example_portfolio() -> Portfolio:
    """Ten AAPL shares, twenty MSFT shares, and USD 100 cash."""
    return Portfolio(
        "holdings",
        aapl=ApplePosition("apple", quantity=10),
        msft=MicrosoftPosition("microsoft", quantity=20),
        cash=USDCash("cash", amount=100),
    )


def example_market_data() -> MarketData:
    """Five synthetic, synchronized business-day observations, not market quotes."""
    dates = tuple(date(2026, 1, day) for day in range(5, 10))
    return MarketData(
        aapl=EquityPrices(AAPL, Series(dates, (100, 102, 98, 104, 103))),
        msft=EquityPrices(MSFT, Series(dates, (50, 49, 48, 52, 54))),
    )


def main() -> None:
    """Show valuation, attribution, and risk for the fixed example portfolio."""
    path = PortfolioPricer(example_portfolio())(example_market_data())
    analytics = path.analytics(periods_per_year=252)
    print("Synthetic AAPL / MSFT portfolio, fixed shares and USD cash")
    print("Date          AAPL value  MSFT value       Cash        NAV")
    for index, day in enumerate(path.nav.dates):
        print(
            f"{day}  {path.aapl.value.values[index]:10.2f}  {path.msft.value.values[index]:10.2f}"
            f"  {path.cash_value.values[index]:9.2f}  {path.nav.values[index]:9.2f}"
        )
    print(f"Total price return: {analytics.total_return:.4%}")
    print(f"Maximum drawdown: {analytics.maximum_drawdown:.4%}")
    print(f"Observed-path annualized volatility: {analytics.annualized_volatility:.4%}")
    print(f"Latest-weights annualized volatility: {analytics.latest_weights_risk.volatility:.4%}")
    print(
        "Latest-weights volatility contributions: "
        f"AAPL {analytics.latest_weights_risk.aapl_contribution:.4%}, "
        f"MSFT {analytics.latest_weights_risk.msft_contribution:.4%}, cash 0.0000%"
    )


if __name__ == "__main__":
    main()

"""An explicit equity portfolio evaluated over synthetic price vectors.

Run ``python -m examples.portfolio`` from a checkout for the worked example.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from sympy import Expr

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.models import (
    AAPL,
    MSFT,
    ApplePosition,
    EquityPosition,
    EquityPrices,
    MicrosoftPosition,
    USDCash,
)
from examples.portfolio.named import NamedMarketData, NamedPortfolioPricer
from modello import InstanceDummy, Modello, component


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

    @property
    def positions(self) -> NamedMap[EquityPosition]:
        """Expose the explicitly declared positions to the shared evaluator."""
        return NamedMap({"aapl": self.aapl, "msft": self.msft})

    @property
    def values(self) -> NamedMap[Expr]:
        """Project positions to market values."""
        return self.positions.map(lambda position: position.market_value)


@dataclass(frozen=True)
class MarketData:
    """The explicit portfolio's two aligned price vectors."""

    aapl: EquityPrices[AAPL]
    msft: EquityPrices[MSFT]

    def __post_init__(self) -> None:
        """Require instrument identity and exact date alignment."""
        if self.aapl.equity is not AAPL or self.msft.equity is not MSFT:
            raise ValueError("AAPL and MSFT price histories must match their named fields")
        if self.aapl.close.times != self.msft.close.times:
            raise ValueError("Price histories must use the same observation dates")


class PortfolioPricer:
    """Compile a scalar model once, then evaluate its formulas over price vectors."""

    def __init__(self, portfolio: Portfolio) -> None:
        """Freeze known holdings and cash into a reusable numerical calculation."""
        self._pricer = NamedPortfolioPricer(portfolio)

    def __call__(self, data: MarketData) -> NamedPortfolioPath[date]:
        """Adapt the explicit tutorial fields to the shared named calculation."""
        return self._pricer(NamedMarketData(data.aapl.close.times, NamedMap({"aapl": data.aapl, "msft": data.msft})))


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
    for index, day in enumerate(path.nav.times):
        print(
            f"{day}  {path.positions['aapl'].value.values[index]:10.2f}"
            f"  {path.positions['msft'].value.values[index]:10.2f}"
            f"  {path.cash_value.values[index]:9.2f}  {path.nav.values[index]:9.2f}"
        )
    print(f"Total price return: {analytics.performance.total_return:.4%}")
    print(f"Maximum drawdown: {analytics.performance.maximum_drawdown:.4%}")
    print(f"Observed-path annualized volatility: {analytics.performance.annualized_volatility:.4%}")
    print(f"Latest-weights annualized volatility: {analytics.latest_weights_risk.volatility:.4%}")
    print(
        "Latest-weights volatility contributions: "
        f"AAPL {analytics.latest_weights_risk.contributions['aapl']:.4%}, "
        f"MSFT {analytics.latest_weights_risk.contributions['msft']:.4%}, cash 0.0000%"
    )


if __name__ == "__main__":
    main()

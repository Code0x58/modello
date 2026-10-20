"""An explicit equity portfolio evaluated over synthetic price vectors.

Run ``python -m examples.portfolio`` for the worked example.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.basket import EquityBasket, EquityHoldings, equity
from examples.portfolio.models import AAPL, MSFT, EquityPrices, USDCash
from examples.portfolio.providers import DummyEquityProvider
from examples.portfolio.valuation import NamedMarketData, NamedPortfolio
from modello import component


class EquityMembers(EquityHoldings):
    """A declared domain of two named equity positions."""

    aapl = equity.component(DummyEquityProvider(symbol="AAPL"))
    msft = equity.component(DummyEquityProvider(symbol="MSFT"))


class Basket(EquityBasket[EquityMembers]):
    """A declared equity basket with provider bindings and an explicit reduction."""

    members = component(EquityMembers)
    market_value = EquityBasket.total(members)


class Portfolio(NamedPortfolio[EquityMembers]):
    """A basket and cash valued in one joint symbolic model."""

    basket = component(Basket)
    cash = component(USDCash)
    nav = basket.capital
    nav = basket.market_value + cash.amount

    @classmethod
    def example(cls) -> Portfolio:
        """Ten AAPL shares, twenty MSFT shares, and USD 100 cash."""
        return cls(
            "holdings",
            basket=Basket(
                "basket",
                members=EquityMembers(
                    "equities",
                    aapl=equity("apple", quantity=10),
                    msft=equity("microsoft", quantity=20),
                ),
            ),
            cash=USDCash("cash", amount=100),
        )


@dataclass(frozen=True)
class MarketData:
    """The explicit portfolio's two aligned price vectors."""

    aapl: EquityPrices
    msft: EquityPrices

    def __post_init__(self) -> None:
        """Require instrument identity and exact date alignment."""
        if self.aapl.equity != AAPL or self.msft.equity != MSFT:
            raise ValueError("AAPL and MSFT price histories must match their named fields")
        if self.aapl.close.times != self.msft.close.times:
            raise ValueError("Price histories must use the same observation dates")

    def named(self) -> NamedMarketData:
        """Expose the same observations on the portfolio's declared named domain."""
        return NamedMarketData(self.aapl.close.times, NamedMap({"aapl": self.aapl, "msft": self.msft}))

    @classmethod
    def example(cls) -> MarketData:
        """Five synthetic, synchronized business-day observations, not market quotes."""
        dates = tuple(date(2026, 1, day) for day in range(5, 10))
        return cls(
            aapl=EquityPrices(AAPL, Series(dates, (100, 102, 98, 104, 103))),
            msft=EquityPrices(MSFT, Series(dates, (50, 49, 48, 52, 54))),
        )


def main() -> None:
    """Show valuation, attribution, and risk for the fixed example portfolio."""
    path = Portfolio.example().pricer().over(MarketData.example().named())
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

"""Extend the explicit two-equity example with a third declared holding.

Run ``python -m examples.portfolio.named`` for the worked example.
"""

from __future__ import annotations

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.basket import EquityBasket, equity
from examples.portfolio.explicit import EquityMembers, MarketData
from examples.portfolio.models import GOOGL, EquityPrices, USDCash
from examples.portfolio.providers import DummyEquityProvider
from examples.portfolio.valuation import NamedMarketData, NamedPortfolio
from modello import component


class ThreeEquities(EquityMembers):
    """Add one generic position; aggregates and projected maps follow automatically."""

    googl = equity.component(DummyEquityProvider(symbol="GOOGL"))


class ThreeEquityBasket(EquityBasket[ThreeEquities]):
    """The same basket equations over an extended named holding domain."""

    members = component(ThreeEquities)
    market_value = EquityBasket.total(members)


class ThreeEquityPortfolio(NamedPortfolio[ThreeEquities]):
    """A stable model definition for the extended worked example."""

    basket = component(ThreeEquityBasket)
    cash = component(USDCash)
    nav = basket.capital
    nav = basket.market_value + cash.amount

    @classmethod
    def example(cls) -> ThreeEquityPortfolio:
        """Three declared equities with fixed quantities and USD cash."""
        return cls(
            "holdings",
            basket=ThreeEquityBasket(
                "basket",
                members=ThreeEquities(
                    "equities",
                    aapl=equity("apple", quantity=10),
                    msft=equity("microsoft", quantity=20),
                    googl=equity("alphabet", quantity=1),
                ),
            ),
            cash=USDCash("cash", amount=100),
        )

    @staticmethod
    def example_data() -> NamedMarketData:
        """Synthetic price vectors for the three declared members."""
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
    path = ThreeEquityPortfolio.example().pricer().over(ThreeEquityPortfolio.example_data())
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

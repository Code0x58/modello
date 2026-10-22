"""Packaged synthetic data and observation times shared by portfolio tests."""

from datetime import date, timedelta
from pathlib import Path

import examples.portfolio
from examples.data import NamedMap, Series
from examples.portfolio.demo import Basket, EquityMembers, Portfolio
from examples.portfolio.models import AAPL, GOOGL, MSFT, EquityPrices, Quote, USDCash
from examples.portfolio.sources import DummyEquityProvider, timestamp
from examples.portfolio.valuation import EquityBasket, NamedMarketData, NamedPortfolio, equity
from modello import component

DATA = Path(examples.portfolio.__file__).with_name("data")
AGE = timedelta(days=4)
TIMES = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
DECISION = timestamp("2026-01-07T09:00:00Z")
ARRIVAL = timestamp("2026-01-07T09:05:00Z")
ASSESSMENT = timestamp("2026-01-07T09:15:00Z")


def fixed_portfolio(
    *, quantities: tuple[int, int] = (10, 20), cash: int = 100, prices: tuple[int, int] | None = None
) -> Portfolio:
    return Portfolio(
        "fixed",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "equities",
                aapl=equity(
                    "apple",
                    quantity=quantities[0],
                    **({} if prices is None else {"instrument": Quote("a", price=prices[0])}),
                ),
                msft=equity(
                    "microsoft",
                    quantity=quantities[1],
                    **({} if prices is None else {"instrument": Quote("m", price=prices[1])}),
                ),
            ),
        ),
        cash=USDCash("cash", amount=cash),
    )


def market_data() -> NamedMarketData[date]:
    times = tuple(date(2026, 1, day) for day in range(5, 10))
    return NamedMarketData(
        times,
        NamedMap(
            {
                "aapl": EquityPrices(AAPL, Series(times, (100, 102, 98, 104, 103))),
                "msft": EquityPrices(MSFT, Series(times, (50, 49, 48, 52, 54))),
            }
        ),
    )


class ThreeEquities(EquityMembers):
    googl = equity.component(DummyEquityProvider(symbol="GOOGL"))


ThreeBasket = EquityBasket.define(ThreeEquities)


class ThreeEquityPortfolio(NamedPortfolio[ThreeEquities]):
    basket = component(ThreeBasket)
    cash = component(USDCash)
    nav = basket.capital
    nav = basket.market_value + cash.amount

    @classmethod
    def example(cls) -> "ThreeEquityPortfolio":
        base = fixed_portfolio()
        return cls(
            "extended",
            basket=ThreeBasket(
                "basket",
                members=ThreeEquities(
                    "members",
                    aapl=base.basket.members.aapl,
                    msft=base.basket.members.msft,
                    googl=equity("alphabet", quantity=1),
                ),
            ),
            cash=base.cash,
        )

    @staticmethod
    def example_data() -> NamedMarketData[date]:
        data = market_data()
        return NamedMarketData(
            data.times,
            NamedMap(
                {
                    **data.prices,
                    "googl": EquityPrices(GOOGL, Series(data.times, (1000, 1005, 990, 1010, 1020))),
                }
            ),
        )

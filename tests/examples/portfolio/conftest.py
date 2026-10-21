"""Shared synthetic portfolio fixtures."""

import pytest

from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.demo import Basket, EquityMembers, ResearchScenario
from examples.portfolio.models import EquityPosition, USDCash
from examples.portfolio.sources import (
    CSVTransactionSource,
)
from examples.portfolio.valuation import NamedPortfolio

from ._data import AGE, DATA, TIMES


@pytest.fixture(scope="module")
def market():
    return NamedPortfolio.define(Basket)("market").load_market()


@pytest.fixture(scope="module")
def ledger():
    return CSVTransactionSource(DATA / "transactions.csv").load()


@pytest.fixture(scope="module")
def engine(market):
    opening = NamedPortfolio.define(Basket)(
        "opening",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "equities", aapl=EquityPosition("apple", quantity=10), msft=EquityPosition("microsoft", quantity=10)
            ),
        ),
        cash=USDCash("cash", amount=1000),
    )
    return AccountEngine(Account("research", TIMES[0], opening), market, max_price_age=AGE)


@pytest.fixture(scope="module")
def scenario():
    return ResearchScenario.example().run()


@pytest.fixture(scope="module")
def reporting(scenario):
    return scenario.report()

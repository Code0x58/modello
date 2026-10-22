"""Shared synthetic portfolio fixtures."""

import pytest

from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.demo import Basket, ResearchScenario
from examples.portfolio.sources import (
    CSVTransactionSource,
)
from examples.portfolio.valuation import NamedPortfolio

from ._data import AGE, DATA, TIMES, fixed_portfolio


@pytest.fixture(scope="module")
def market():
    return NamedPortfolio.define(Basket)("market").load_market()


@pytest.fixture(scope="module")
def ledger():
    return CSVTransactionSource(DATA / "transactions.csv").load()


@pytest.fixture(scope="module")
def engine(market):
    opening = fixed_portfolio(quantities=(10, 10), cash=1000)
    return AccountEngine(Account("research", TIMES[0], opening), market, max_price_age=AGE)


@pytest.fixture(scope="module")
def scenario():
    return ResearchScenario.example().run()


@pytest.fixture(scope="module")
def reporting(scenario):
    return scenario.report()

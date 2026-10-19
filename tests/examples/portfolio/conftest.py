"""Shared synthetic portfolio fixtures."""

import pytest

from examples.compute.named_map import NamedMap
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.demo import example_run, research_report
from examples.portfolio.models import AAPL, MSFT, ApplePosition, MicrosoftPosition, USDCash
from examples.portfolio.named import EquityMembers, portfolio_model
from examples.portfolio.sources import (
    CSVMarketSource,
    CSVTransactionSource,
)

from ._data import AGE, DATA, TIMES


@pytest.fixture(scope="module")
def market():
    return CSVMarketSource(DATA / "market.csv", NamedMap({"aapl": AAPL, "msft": MSFT})).load()


@pytest.fixture(scope="module")
def ledger():
    return CSVTransactionSource(DATA / "transactions.csv").load()


@pytest.fixture(scope="module")
def engine(market):
    opening = portfolio_model(EquityMembers)(
        "opening",
        equities=EquityMembers(
            "equities", aapl=ApplePosition("apple", quantity=10), msft=MicrosoftPosition("microsoft", quantity=10)
        ),
        cash=USDCash("cash", amount=1000),
    )
    return AccountEngine(Account("research", TIMES[0], opening), market, max_price_age=AGE)


@pytest.fixture(scope="module")
def scenario():
    return example_run()


@pytest.fixture(scope="module")
def reporting(scenario):
    return research_report(scenario)

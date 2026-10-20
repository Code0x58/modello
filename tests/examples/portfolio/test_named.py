"""The extended three-equity example and its stable model definition."""

import pytest
from sympy import Rational

from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.models import EquityPosition, Quote, USDCash
from examples.portfolio.named import ThreeEquities, ThreeEquityBasket, ThreeEquityPortfolio
from examples.portfolio.valuation import NamedPortfolio
from modello import BoundInstanceDummy


def test_new_member_flows_through_aggregates_and_all_projections():
    portfolio = ThreeEquityPortfolio.example()
    path = NamedPortfolioPath.from_model(portfolio.pricer().over(ThreeEquityPortfolio.example_data()))
    assert path.nav.values == (3100, 3105, 3030, 3190, 3230)
    assert (
        tuple(portfolio.positions) == tuple(portfolio.values) == tuple(portfolio.weights) == ("aapl", "msft", "googl")
    )
    assert tuple(path.positions) == tuple(path.values) == tuple(path.weights) == ("aapl", "msft", "googl")
    assert path.values["googl"].values == (1000, 1005, 990, 1010, 1020)
    for index, nav in enumerate(path.nav.values):
        assert nav == sum(value.values[index] for value in path.values.values()) + path.cash_value.values[index]
        total_weight = sum(weight.values[index] for weight in path.weights.values()) + path.cash_weight.values[index]
        assert total_weight == pytest.approx(1)
    returns = path.nav.simple_returns()
    for index, actual in enumerate(returns.values):
        assert sum(position.return_contribution.values[index] for position in path.positions.values()) == pytest.approx(
            actual, abs=1e-15
        )


def test_worked_portfolio_reuses_its_model_definition_and_compiled_account_view():
    first, second = ThreeEquityPortfolio.example(), ThreeEquityPortfolio.example()
    assert type(first) is type(second) is ThreeEquityPortfolio
    assert type(first).plan() is type(second).plan()
    assert first.basket.members.aapl.instrument.price != second.basket.members.aapl.instrument.price


def test_map_reduction_keeps_inverse_valuation():
    source = ThreeEquities(
        "equities",
        aapl=EquityPosition("apple", instrument=Quote("quote", price=100)),
        msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=50)),
        googl=EquityPosition("alphabet", quantity=1, instrument=Quote("quote", price=1000)),
    )
    portfolio = NamedPortfolio.define(ThreeEquityBasket)(
        "target", basket=ThreeEquityBasket("basket", members=source), cash=USDCash("cash", amount=100), nav=3100
    )
    assert portfolio.positions["aapl"].quantity == 10
    assert portfolio.weights == {"aapl": Rational(10, 31), "msft": Rational(10, 31), "googl": Rational(10, 31)}
    assert isinstance(source.aapl.quantity, BoundInstanceDummy)

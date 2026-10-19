"""Explicit holdings, valuation, and inverse portfolio relationships."""

from datetime import date

import pytest
from sympy import Rational, Symbol

from examples.compute.series import Series
from examples.portfolio.explicit import MarketData, Portfolio
from examples.portfolio.models import (
    AAPL,
    MSFT,
    EquityPosition,
    EquityPrices,
    Quote,
    USDCash,
)
from examples.portfolio.named import Basket, EquityMembers
from modello import BoundInstanceDummy


def test_explicit_instrument_types_and_inverse_valuation():
    """A NAV constraint solves the quantity of an explicitly named equity."""
    source = EquityPosition("apple", instrument=Quote("quote", price=100))
    portfolio = Portfolio(
        "target",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "members",
                aapl=source,
                msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=50)),
            ),
        ),
        cash=USDCash("cash", amount=100),
        nav=2100,
    )
    assert portfolio.instruments["aapl"].symbol == "AAPL"
    assert portfolio.instruments["msft"].symbol == "MSFT"
    assert portfolio.basket.members.aapl.quantity == 10
    assert portfolio.basket.weights["aapl"] == Rational(10, 21)
    assert isinstance(source.quantity, BoundInstanceDummy)


def test_valuation_vectors_and_drifting_weights():
    """A fixed-share portfolio values each aligned observation independently."""
    path = Portfolio.example().pricer().over(MarketData.example().named())
    assert path.nav.values == (2100, 2100, 2040, 2180, 2210)
    assert path.positions["aapl"].value.values == (1000, 1020, 980, 1040, 1030)
    assert path.positions["msft"].value.values == (1000, 980, 960, 1040, 1080)
    assert path.positions["aapl"].weight.values[0] == pytest.approx(1000 / 2100)
    assert path.positions["aapl"].weight.values[-1] == pytest.approx(1030 / 2210)
    for apple, microsoft, cash in zip(
        path.positions["aapl"].weight.values, path.positions["msft"].weight.values, path.cash_weight.values
    ):
        assert apple + microsoft + cash == pytest.approx(1)


def test_cash_only_portfolio_and_zero_risk():
    """Zero equity positions have zero returns and risk without dividing by zero."""
    portfolio = Portfolio(
        "cash only",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "members", aapl=EquityPosition("apple", quantity=0), msft=EquityPosition("microsoft", quantity=0)
            ),
        ),
        cash=USDCash("cash", amount=100),
    )
    path = portfolio.pricer().over(MarketData.example().named())
    result = path.analytics(periods_per_year=252)
    assert path.nav.values == (100,) * 5
    assert path.cash_weight.values == (1,) * 5
    assert (
        result.performance.total_return
        == result.performance.annualized_volatility
        == result.performance.maximum_drawdown
        == 0
    )
    assert result.latest_weights_risk.volatility == 0
    assert result.latest_weights_risk.contributions["aapl"] == result.latest_weights_risk.contributions["msft"] == 0


def test_pricer_reuses_the_model_for_another_price_vector():
    """Holdings stay fixed while another observation grid can be evaluated."""
    pricer = Portfolio.example().pricer()
    dates = (date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4))
    data = MarketData(
        EquityPrices(AAPL, Series(dates, (10, 20, 30))),
        EquityPrices(MSFT, Series(dates, (5, 10, 15))),
    )
    assert pricer.over(data.named()).nav.values == (300, 500, 700)
    assert pricer.over(MarketData.example().named()).nav.values[-1] == 2210


def test_prices_are_not_silently_aligned_or_relabelled():
    data = MarketData.example()
    shifted = Series(tuple(date(2026, 2, day) for day in range(2, 7)), data.msft.close.values)
    with pytest.raises(ValueError, match="same observation dates"):
        MarketData(data.aapl, EquityPrices(MSFT, shifted))
    with pytest.raises(ValueError, match="match their named fields"):
        MarketData(data.msft, data.aapl)


def test_compile_requires_fixed_holdings_and_independent_symbolic_prices():
    with pytest.raises(ValueError, match="position quantities"):
        Portfolio("unspecified").pricer()
    for price in (100, Symbol("shared", positive=True)):
        portfolio = Portfolio(
            "known or linked quotes",
            basket=Basket(
                "basket",
                members=EquityMembers(
                    "members",
                    aapl=EquityPosition("apple", quantity=10, instrument=Quote("quote", price=price)),
                    msft=EquityPosition("microsoft", quantity=20, instrument=Quote("quote", price=price)),
                ),
            ),
            cash=USDCash("cash", amount=100),
        )
        with pytest.raises(ValueError, match="independent symbolic input"):
            portfolio.pricer()

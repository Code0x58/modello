"""Explicit holdings, valuation, and inverse portfolio relationships."""

from datetime import date

import pytest
from sympy import Rational, Symbol

from examples.compute.series import Series
from examples.portfolio.explicit import MarketData, Portfolio, PortfolioPricer, example_market_data, example_portfolio
from examples.portfolio.models import (
    AAPL,
    MSFT,
    ApplePosition,
    Equity,
    EquityPosition,
    EquityPrices,
    MicrosoftPosition,
    USDCash,
)
from modello import BoundInstanceDummy, component


def test_explicit_instrument_types_and_inverse_valuation():
    """A NAV constraint solves the quantity of an explicitly named equity."""
    source = ApplePosition("apple", instrument=AAPL("quote", price=100))
    portfolio = Portfolio(
        "target",
        aapl=source,
        msft=MicrosoftPosition("microsoft", quantity=20, instrument=MSFT("quote", price=50)),
        cash=USDCash("cash", amount=100),
        nav=2100,
    )
    assert portfolio.aapl.instrument.ticker == "AAPL"
    assert portfolio.msft.instrument.ticker == "MSFT"
    assert portfolio.aapl.quantity == 10
    assert portfolio.aapl_weight == Rational(10, 21)
    assert isinstance(source.quantity, BoundInstanceDummy)


def test_valuation_vectors_and_drifting_weights():
    """A fixed-share portfolio values each aligned observation independently."""
    path = PortfolioPricer(example_portfolio())(example_market_data())
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
        aapl=ApplePosition("apple", quantity=0),
        msft=MicrosoftPosition("microsoft", quantity=0),
        cash=USDCash("cash", amount=100),
    )
    path = PortfolioPricer(portfolio)(example_market_data())
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
    pricer = PortfolioPricer(example_portfolio())
    dates = (date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 4))
    data = MarketData(
        EquityPrices(AAPL, Series(dates, (10, 20, 30))),
        EquityPrices(MSFT, Series(dates, (5, 10, 15))),
    )
    assert pricer(data).nav.values == (300, 500, 700)
    assert pricer(example_market_data()).nav.values[-1] == 2210


def test_prices_are_not_silently_aligned_or_relabelled():
    data = example_market_data()
    shifted = Series(tuple(date(2026, 2, day) for day in range(2, 7)), data.msft.close.values)
    with pytest.raises(ValueError, match="same observation dates"):
        MarketData(data.aapl, EquityPrices(MSFT, shifted))
    with pytest.raises(ValueError, match="match their named fields"):
        MarketData(data.msft, data.aapl)


def test_compile_requires_fixed_holdings_and_independent_symbolic_prices():
    with pytest.raises(ValueError, match="position quantities"):
        PortfolioPricer(Portfolio("unspecified"))
    for price in (100, Symbol("shared", positive=True)):
        portfolio = Portfolio(
            "known or linked quotes",
            aapl=ApplePosition("apple", quantity=10, instrument=AAPL("quote", price=price)),
            msft=MicrosoftPosition("microsoft", quantity=20, instrument=MSFT("quote", price=price)),
            cash=USDCash("cash", amount=100),
        )
        with pytest.raises(ValueError, match="independent symbolic input"):
            PortfolioPricer(portfolio)


def test_another_named_instrument_can_extend_the_symbolic_model():
    """A third explicit position works, including inherited weights, without containers."""

    class Alphabet(Equity):
        ticker = "GOOGL"

    class AlphabetPosition(EquityPosition):
        instrument = component(Alphabet)

    class ExtendedPortfolio(Portfolio):
        googl = component(AlphabetPosition)
        nav = Portfolio.aapl.market_value + Portfolio.msft.market_value + Portfolio.cash.amount + googl.market_value
        googl_weight = googl.market_value / nav

    portfolio = ExtendedPortfolio(
        "three equities",
        aapl=ApplePosition("apple", quantity=10, instrument=AAPL("quote", price=100)),
        msft=MicrosoftPosition("microsoft", quantity=20, instrument=MSFT("quote", price=50)),
        cash=USDCash("cash", amount=100),
        googl=AlphabetPosition("alphabet", quantity=1, instrument=Alphabet("quote", price=1000)),
    )
    assert portfolio.nav == 3100
    assert portfolio.aapl_weight == portfolio.msft_weight == portfolio.googl_weight == Rational(10, 31)
    assert portfolio.cash_weight == Rational(1, 31)

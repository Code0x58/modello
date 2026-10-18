"""Portfolio accounting and risk checks against independent calculations."""

from datetime import date
from math import sqrt

import pytest
from sympy import ImmutableMatrix, Rational, Symbol, ones

from examples.portfolio import (
    AAPL,
    MSFT,
    ApplePosition,
    Equity,
    EquityPosition,
    EquityPrices,
    MarketData,
    MicrosoftPosition,
    Portfolio,
    PortfolioPricer,
    Series,
    USDCash,
    example_market_data,
    example_portfolio,
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
    assert path.aapl.value.values == (1000, 1020, 980, 1040, 1030)
    assert path.msft.value.values == (1000, 980, 960, 1040, 1080)
    assert path.nav.as_vector().shape == (5, 1)
    assert path.aapl.weight.values[0] == pytest.approx(1000 / 2100)
    assert path.aapl.weight.values[-1] == pytest.approx(1030 / 2210)
    for apple, microsoft, cash in zip(path.aapl.weight.values, path.msft.weight.values, path.cash_weight.values):
        assert apple + microsoft + cash == pytest.approx(1)


def test_return_attribution_uses_start_of_period_weights():
    """Contributions reconcile to actual NAV returns as the weights drift."""
    data = example_market_data()
    path = PortfolioPricer(example_portfolio())(data)
    returns = path.nav.simple_returns()
    assert returns.dates == data.aapl.close.dates[1:]
    assert returns.values == pytest.approx((0, -1 / 35, 7 / 102, 3 / 218))
    for actual, apple, microsoft in zip(
        returns.values, path.aapl.return_contribution.values, path.msft.return_contribution.values
    ):
        assert actual == pytest.approx(apple + microsoft, abs=1e-15)


def test_sample_statistics_and_covariance_matrix():
    """Reference covariance uses a centered return matrix and n-1 degrees of freedom."""
    path = PortfolioPricer(example_portfolio())(example_market_data())
    analytics = path.analytics(periods_per_year=252)
    returns = path.nav.simple_returns().values
    sample_mean = sum(returns) / len(returns)
    sample_variance = sum((value - sample_mean) ** 2 for value in returns) / (len(returns) - 1)
    assert analytics.total_return == pytest.approx(11 / 210)
    assert analytics.mean_period_return == pytest.approx(sample_mean)
    assert analytics.annualized_volatility == pytest.approx(sqrt(252 * sample_variance))
    assert analytics.maximum_drawdown == pytest.approx(1 / 35)
    observations = ImmutableMatrix.hstack(path.aapl.returns.as_vector(), path.msft.returns.as_vector())
    count = observations.rows
    centered = observations - ones(count, 1) * (ones(1, count) * observations / count)
    reference = centered.T * centered * 252 / (count - 1)
    actual = analytics.annualized_covariance.as_matrix()
    for index in range(4):
        assert float(actual[index]) == pytest.approx(float(reference[index]))


def test_euler_contributions_and_reference_weights_risk():
    """Volatility allocation agrees with finite differences of the covariance formula."""
    path = PortfolioPricer(example_portfolio())(example_market_data())
    analytics = path.analytics(periods_per_year=252)
    risk = analytics.latest_weights_risk
    cov = analytics.annualized_covariance.as_matrix()
    wa = path.aapl.weight.values[-1]
    wm = path.msft.weight.values[-1]

    def volatility(apple, microsoft):
        weights = ImmutableMatrix([apple, microsoft])
        return sqrt(float((weights.T * cov * weights)[0]))

    step = 1e-6
    assert risk.volatility == pytest.approx(volatility(wa, wm))
    assert risk.aapl_contribution == pytest.approx(
        wa * (volatility(wa + step, wm) - volatility(wa - step, wm)) / (2 * step)
    )
    assert risk.msft_contribution == pytest.approx(
        wm * (volatility(wa, wm + step) - volatility(wa, wm - step)) / (2 * step)
    )
    assert risk.cash_contribution == 0
    assert risk.aapl_contribution + risk.msft_contribution == pytest.approx(risk.volatility)
    assert risk.volatility != pytest.approx(analytics.annualized_volatility)
    monthly = path.analytics(periods_per_year=12)
    assert monthly.annualized_volatility == pytest.approx(analytics.annualized_volatility * sqrt(12 / 252))


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
    assert result.total_return == result.annualized_volatility == result.maximum_drawdown == 0
    assert result.latest_weights_risk.volatility == 0
    assert result.latest_weights_risk.aapl_contribution == result.latest_weights_risk.msft_contribution == 0


def test_opposite_asset_returns_diversify_reference_weights_risk():
    """Negative covariance can offset fixed-weight risk while buy-and-hold weights drift."""
    portfolio = Portfolio(
        "opposite returns",
        aapl=ApplePosition("apple", quantity=1),
        msft=MicrosoftPosition("microsoft", quantity=1),
        cash=USDCash("cash", amount=0),
    )
    dates = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    data = MarketData(
        EquityPrices(AAPL, Series(dates, (100, 110, 99))),
        EquityPrices(MSFT, Series(dates, (100, 90, 99))),
    )
    path = PortfolioPricer(portfolio)(data)
    result = path.analytics(periods_per_year=252)
    assert result.annualized_covariance.cross_covariance < 0
    assert result.latest_weights_risk.volatility == pytest.approx(0, abs=1e-12)
    assert result.annualized_volatility > 0
    assert path.nav.values == (200, 200, 198)


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


@pytest.mark.parametrize(
    "dates, values",
    [
        ((), ()),
        ((date(2026, 1, 1),), (1, 2)),
        ((date(2026, 1, 1), date(2026, 1, 1)), (1, 2)),
        ((date(2026, 1, 2), date(2026, 1, 1)), (1, 2)),
        ((date(2026, 1, 1),), (float("nan"),)),
        ((date(2026, 1, 1),), (float("inf"),)),
    ],
)
def test_invalid_vectors(dates, values):
    """Malformed observation axes and nonfinite data are rejected before calculation."""
    with pytest.raises(ValueError):
        Series(dates, values)


@pytest.mark.parametrize("price", [0, -1])
def test_nonpositive_prices(price):
    with pytest.raises(ValueError, match="strictly positive"):
        EquityPrices(AAPL, Series((date(2026, 1, 1),), (price,)))


def test_prices_are_not_silently_aligned_or_relabelled():
    data = example_market_data()
    shifted = Series(tuple(date(2026, 2, day) for day in range(2, 7)), data.msft.close.values)
    with pytest.raises(ValueError, match="same observation dates"):
        MarketData(data.aapl, EquityPrices(MSFT, shifted))
    with pytest.raises(ValueError, match="match their named fields"):
        MarketData(data.msft, data.aapl)


@pytest.mark.parametrize("frequency", [0, -1, True, 1.5])
def test_invalid_risk_frequency(frequency):
    path = PortfolioPricer(example_portfolio())(example_market_data())
    with pytest.raises(ValueError, match="positive integer"):
        path.analytics(periods_per_year=frequency)


def test_insufficient_observations():
    first = date(2026, 1, 1)
    with pytest.raises(ValueError, match="at least two"):
        Series((first,), (1,)).simple_returns()
    dates = (first, date(2026, 1, 2))
    data = MarketData(EquityPrices(AAPL, Series(dates, (1, 2))), EquityPrices(MSFT, Series(dates, (1, 2))))
    with pytest.raises(ValueError, match="at least three"):
        PortfolioPricer(example_portfolio())(data).analytics(periods_per_year=252)


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

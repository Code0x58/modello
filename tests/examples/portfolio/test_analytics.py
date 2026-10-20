"""Return attribution, covariance, and fixed-weight portfolio risk."""

from datetime import date
from math import sqrt

import pytest
from sympy import ImmutableMatrix, ones

from examples.compute.series import Series
from examples.portfolio.explicit import Basket, EquityMembers, MarketData, Portfolio
from examples.portfolio.models import (
    AAPL,
    MSFT,
    EquityPosition,
    EquityPrices,
    USDCash,
)
from examples.portfolio.named import ThreeEquityPortfolio
from examples.portfolio.valuation import NamedPortfolioPricer


def test_return_attribution_uses_start_of_period_weights():
    """Contributions reconcile to actual NAV returns as the weights drift."""
    data = MarketData.example()
    path = Portfolio.example().pricer().over(data.named())
    returns = path.nav.simple_returns()
    assert returns.times == data.aapl.close.times[1:]
    assert returns.values == pytest.approx((0, -1 / 35, 7 / 102, 3 / 218))
    for actual, apple, microsoft in zip(
        returns.values,
        path.positions["aapl"].return_contribution.values,
        path.positions["msft"].return_contribution.values,
    ):
        assert actual == pytest.approx(apple + microsoft, abs=1e-15)


def test_sample_statistics_and_covariance_matrix():
    """Reference covariance uses a centered return matrix and n-1 degrees of freedom."""
    path = Portfolio.example().pricer().over(MarketData.example().named())
    analytics = path.analytics(periods_per_year=252)
    returns = path.nav.simple_returns().values
    sample_mean = sum(returns) / len(returns)
    sample_variance = sum((value - sample_mean) ** 2 for value in returns) / (len(returns) - 1)
    assert analytics.performance.total_return == pytest.approx(11 / 210)
    assert analytics.performance.mean_period_return == pytest.approx(sample_mean)
    assert analytics.performance.annualized_volatility == pytest.approx(sqrt(252 * sample_variance))
    assert analytics.performance.maximum_drawdown == pytest.approx(1 / 35)
    observations = ImmutableMatrix.hstack(
        ImmutableMatrix(path.positions["aapl"].returns.values), ImmutableMatrix(path.positions["msft"].returns.values)
    )
    count = observations.rows
    centered = observations - ones(count, 1) * (ones(1, count) * observations / count)
    reference = centered.T * centered * 252 / (count - 1)
    actual = ImmutableMatrix(
        [[analytics.annualized_covariance[left][right] for right in ("aapl", "msft")] for left in ("aapl", "msft")]
    )
    for index in range(4):
        assert float(actual[index]) == pytest.approx(float(reference[index]))


def test_euler_contributions_and_reference_weights_risk():
    """Volatility allocation agrees with finite differences of the covariance formula."""
    path = Portfolio.example().pricer().over(MarketData.example().named())
    analytics = path.analytics(periods_per_year=252)
    risk = analytics.latest_weights_risk
    cov = ImmutableMatrix(
        [[analytics.annualized_covariance[left][right] for right in ("aapl", "msft")] for left in ("aapl", "msft")]
    )
    wa = path.positions["aapl"].weight.values[-1]
    wm = path.positions["msft"].weight.values[-1]

    def volatility(apple, microsoft):
        weights = ImmutableMatrix([apple, microsoft])
        return sqrt(float((weights.T * cov * weights)[0]))

    step = 1e-6
    assert risk.volatility == pytest.approx(volatility(wa, wm))
    assert risk.contributions["aapl"] == pytest.approx(
        wa * (volatility(wa + step, wm) - volatility(wa - step, wm)) / (2 * step)
    )
    assert risk.contributions["msft"] == pytest.approx(
        wm * (volatility(wa, wm + step) - volatility(wa, wm - step)) / (2 * step)
    )
    assert risk.cash_contribution == 0
    assert risk.contributions["aapl"] + risk.contributions["msft"] == pytest.approx(risk.volatility)
    assert risk.volatility != pytest.approx(analytics.performance.annualized_volatility)
    monthly = path.analytics(periods_per_year=12)
    assert monthly.performance.annualized_volatility == pytest.approx(
        analytics.performance.annualized_volatility * sqrt(12 / 252)
    )


def test_opposite_asset_returns_diversify_reference_weights_risk():
    """Negative covariance can offset fixed-weight risk while buy-and-hold weights drift."""
    portfolio = Portfolio(
        "opposite returns",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "members", aapl=EquityPosition("apple", quantity=1), msft=EquityPosition("microsoft", quantity=1)
            ),
        ),
        cash=USDCash("cash", amount=0),
    )
    dates = (date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7))
    data = MarketData(
        EquityPrices(AAPL, Series(dates, (100, 110, 99))),
        EquityPrices(MSFT, Series(dates, (100, 90, 99))),
    )
    path = portfolio.pricer().over(data.named())
    result = path.analytics(periods_per_year=252)
    assert result.annualized_covariance["aapl"]["msft"] < 0
    assert result.latest_weights_risk.volatility == pytest.approx(0, abs=1e-12)
    assert result.performance.annualized_volatility > 0
    assert path.nav.values == (200, 200, 198)


@pytest.mark.parametrize("frequency", [0, -1, True, 1.5])
def test_invalid_risk_frequency(frequency):
    path = Portfolio.example().pricer().over(MarketData.example().named())
    with pytest.raises(ValueError, match="positive integer"):
        path.analytics(periods_per_year=frequency)


def test_insufficient_observations():
    first = date(2026, 1, 1)
    with pytest.raises(ValueError, match="at least two"):
        Series((first,), (1,)).simple_returns()
    dates = (first, date(2026, 1, 2))
    data = MarketData(EquityPrices(AAPL, Series(dates, (1, 2))), EquityPrices(MSFT, Series(dates, (1, 2))))
    with pytest.raises(ValueError, match="at least three"):
        Portfolio.example().pricer().over(data.named()).analytics(periods_per_year=252)


def test_named_covariance_and_risk_agree_with_independent_matrix_calculation():
    path = NamedPortfolioPricer(ThreeEquityPortfolio.example()).over(ThreeEquityPortfolio.example_data())
    result = path.analytics(periods_per_year=252)
    names = tuple(path.positions)
    observations = ImmutableMatrix.hstack(*(ImmutableMatrix(path.positions[name].returns.values) for name in names))
    count = observations.rows
    centered = observations - ones(count, 1) * (ones(1, count) * observations / count)
    cov = centered.T * centered * 252 / (count - 1)
    weights = ImmutableMatrix([path.weights[name].values[-1] for name in names])
    volatility = sqrt(float((weights.T * cov * weights)[0]))
    assert result.latest_weights_risk.volatility == pytest.approx(volatility)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            assert result.annualized_covariance[left][right] == pytest.approx(float(cov[i, j]))
        contribution = float(weights[i] * (cov * weights)[i]) / volatility
        assert result.latest_weights_risk.contributions[left] == pytest.approx(contribution)
    assert sum(result.latest_weights_risk.contributions.values()) == pytest.approx(volatility)

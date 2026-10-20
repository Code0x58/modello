"""Return attribution, covariance, and fixed-weight portfolio risk."""

from datetime import date
from math import sqrt

import pytest
from sympy import ImmutableMatrix, ones

from examples.compute.series import Series
from examples.model_plan import Known
from examples.portfolio.analytics import NamedPortfolioPath
from examples.portfolio.explicit import Basket, EquityMembers, MarketData, Portfolio
from examples.portfolio.models import (
    AAPL,
    MSFT,
    EquityPosition,
    EquityPrices,
    USDCash,
)
from examples.portfolio.named import ThreeEquityPortfolio


@pytest.mark.parametrize("varying", ["quantity", "cash"])
def test_fixed_holding_attribution_rejects_state_changes_on_a_general_model_path(varying):
    view = Portfolio.plan()
    model = view.source
    data = MarketData.example().named()
    times = data.times
    quantities = Series(times, (10, 11, 12, 13, 14) if varying == "quantity" else (10,) * len(times))
    cash = Series(times, (100, 200, 300, 400, 500) if varying == "cash" else (100,) * len(times))
    path = view.over(
        times,
        data,
        Known(model.basket.members.aapl.quantity, quantities),
        Known(model.basket.members.msft.quantity, Series(times, (20,) * len(times))),
        Known(model.cash.amount, cash),
    )
    with pytest.raises(ValueError, match="constant quantities and cash"):
        NamedPortfolioPath.from_model(path)


def test_return_attribution_uses_start_of_period_weights():
    """Contributions reconcile to actual NAV returns as the weights drift."""
    data = MarketData.example()
    path = NamedPortfolioPath.from_model(Portfolio.example().pricer().over(data.named()))
    returns = path.nav.simple_returns()
    assert returns.times == data.aapl.close.times[1:]
    assert returns.values == pytest.approx((0, -1 / 35, 7 / 102, 3 / 218))
    for actual, apple, microsoft in zip(
        returns.values,
        path.positions["aapl"].return_contribution.values,
        path.positions["msft"].return_contribution.values,
    ):
        assert actual == pytest.approx(apple + microsoft, abs=1e-15)


def test_euler_contributions_and_reference_weights_risk():
    """Volatility allocation agrees with finite differences of the covariance formula."""
    path = NamedPortfolioPath.from_model(Portfolio.example().pricer().over(MarketData.example().named()))
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
    path = NamedPortfolioPath.from_model(portfolio.pricer().over(data.named()))
    result = path.analytics(periods_per_year=252)
    assert result.annualized_covariance["aapl"]["msft"] < 0
    assert result.latest_weights_risk.volatility == pytest.approx(0, abs=1e-12)
    assert result.performance.annualized_volatility > 0
    assert path.nav.values == (200, 200, 198)


def test_named_covariance_and_risk_agree_with_independent_matrix_calculation():
    path = NamedPortfolioPath.from_model(
        ThreeEquityPortfolio.example().pricer().over(ThreeEquityPortfolio.example_data())
    )
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
    path = NamedPortfolioPath.from_model(portfolio.pricer().over(MarketData.example().named()))
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

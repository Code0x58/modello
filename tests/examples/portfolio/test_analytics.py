"""Independent return, covariance, risk, and index-comparison checks."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from math import prod, sqrt
from statistics import mean, stdev

import pytest
from sympy import ImmutableMatrix, ones

from examples.data import NamedMap, Series
from examples.portfolio.analytics import NamedPortfolioPath, PerformanceIndex, measure
from examples.portfolio.demo import Basket, EquityMembers, Portfolio
from examples.portfolio.models import AAPL, MSFT, EquityPosition, EquityPrices, USDCash
from examples.portfolio.valuation import NamedMarketData
from examples.symbolic import Known

from ._data import ThreeEquityPortfolio, fixed_portfolio, market_data


@pytest.mark.parametrize("varying", ["quantity", "cash"])
def test_fixed_holding_attribution_rejects_state_changes_on_a_general_model_path(varying):
    view = Portfolio.plan()
    model = view.source
    data = market_data()
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
    data = market_data()
    path = NamedPortfolioPath.from_model(fixed_portfolio().pricer().over(data))
    returns = path.nav.simple_returns()
    assert returns.times == data.prices["aapl"].close.times[1:]
    assert returns.values == pytest.approx((0, -1 / 35, 7 / 102, 3 / 218))
    for actual, apple, microsoft in zip(
        returns.values,
        path.positions["aapl"].return_contribution.values,
        path.positions["msft"].return_contribution.values,
    ):
        assert actual == pytest.approx(apple + microsoft, abs=1e-15)


def test_euler_contributions_and_reference_weights_risk():
    """Volatility allocation agrees with finite differences of the covariance formula."""
    path = NamedPortfolioPath.from_model(fixed_portfolio().pricer().over(market_data()))
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
    data = NamedMarketData(
        dates,
        NamedMap(
            {
                "aapl": EquityPrices(AAPL, Series(dates, (100, 110, 99))),
                "msft": EquityPrices(MSFT, Series(dates, (100, 90, 99))),
            }
        ),
    )
    path = NamedPortfolioPath.from_model(portfolio.pricer().over(data))
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
    path = NamedPortfolioPath.from_model(portfolio.pricer().over(market_data()))
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


def quarterly(values):
    return Series(tuple(date(2024, month, 1) for month in (1, 4, 7, 10)) + (date(2025, 1, 1),), tuple(values))


def levels(returns):
    return (100.0, *(100 * prod(1 + r for r in returns[:i]) for i in range(1, len(returns) + 1)))


def test_calendar_cagr_and_excess_return_sharpe():
    returns = (0.1, -0.05, 0.08, 0.02)
    cash_returns = (0.01, 0.011, 0.012, 0.013)
    index = quarterly(levels(returns))
    cash = quarterly(levels(cash_returns))
    result = measure(index, periods_per_year=4, risk_free=cash)
    excess = tuple(a - b for a, b in zip(returns, cash_returns, strict=True))
    assert result.return_count == 4
    assert result.elapsed_days == 366
    assert result.annualization_eligible
    assert result.total_return == pytest.approx(prod(1 + r for r in returns) - 1)
    assert result.cagr == pytest.approx(prod(1 + r for r in returns) ** (365.25 / 366) - 1)
    assert result.sharpe_ratio == pytest.approx(mean(excess) / stdev(excess) * 2)
    assert result.annualized_volatility == pytest.approx(stdev(returns) * 2)
    assert result.maximum_drawdown == pytest.approx(0.05)
    # Calendar growth does not change when the caller changes the sampling convention.
    daily = measure(index, periods_per_year=252, risk_free=cash)
    assert daily.cagr == result.cagr
    assert daily.sharpe_ratio == pytest.approx(result.sharpe_ratio * sqrt(252 / 4))
    assert result.sharpe_ratio != pytest.approx(mean(returns) / stdev(returns) * 2)


@pytest.mark.parametrize(
    "start,end",
    [(date(2025, 1, 1), date(2026, 1, 1)), (date(2024, 2, 29), date(2025, 2, 28))],
)
def test_full_calendar_year_and_leap_anniversary(start, end):
    path = Series((start, start + timedelta(days=90), end), (100, 110, 121))
    assert measure(path, periods_per_year=2).cagr == pytest.approx(1.21 ** (365.25 / 365) - 1)
    shorter = Series((start, start + timedelta(days=90), end - timedelta(days=1)), (100, 110, 121))
    result = measure(shorter, periods_per_year=2, risk_free=shorter.map(lambda _: 100))
    assert not result.annualization_eligible
    assert result.cagr is result.sharpe_ratio is None


def test_timestamp_cagr_uses_elapsed_time_and_can_describe_losses():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    path = Series((start, start + timedelta(days=365), start + timedelta(days=730, hours=12)), (100, 90, 81))
    result = measure(path, periods_per_year=1)
    assert result.cagr == pytest.approx(-0.1)
    assert result.sharpe_ratio is None  # No implicit risk-free assumption.


def test_zero_excess_risk_has_no_sharpe():
    cash = quarterly((100, 101, 102, 103, 104))
    result = measure(cash, periods_per_year=4, risk_free=cash)
    assert result.sharpe_ratio is None
    assert result.cagr is not None


def test_index_comparison_normalizes_levels_and_distinguishes_return_spread_from_relative_wealth():
    portfolio = PerformanceIndex("Account", quarterly((200, 220, 209, 230, 242)))
    benchmark = PerformanceIndex("Total-return index", quarterly((1000, 1050, 1100, 1150, 1250)))
    cash = PerformanceIndex("Cash", quarterly((100, 100, 100, 100, 100)))
    result = portfolio.compare(benchmark, risk_free=cash, periods_per_year=4)
    assert result.normalized["portfolio"].values == pytest.approx((100, 110, 104.5, 115, 121))
    assert result.normalized["benchmark"].values == (100, 105, 110, 115, 125)
    assert result.return_difference == pytest.approx(-0.04)
    assert result.relative_return == pytest.approx(1.21 / 1.25 - 1)
    assert result.active_returns.values[0] == pytest.approx(0.05)
    assert result.portfolio.cagr is not None
    assert result.benchmark.sharpe_ratio is not None
    with pytest.raises(ValueError, match="currency"):
        portfolio.compare(replace(benchmark, currency="EUR"), risk_free=cash, periods_per_year=4)
    shifted = replace(
        benchmark, levels=Series(tuple(t + timedelta(days=1) for t in benchmark.levels.times), benchmark.levels.values)
    )
    with pytest.raises(ValueError, match="observation times"):
        portfolio.compare(shifted, risk_free=cash, periods_per_year=4)


@pytest.mark.parametrize("frequency", [0, -1, True, 2.5])
def test_invalid_frequency(frequency):
    with pytest.raises(ValueError, match="positive integer"):
        measure(quarterly((100, 102, 101, 103, 104)), periods_per_year=frequency)


def test_performance_requires_positive_levels_sufficient_returns_and_exact_cash_alignment():
    with pytest.raises(ValueError, match="three price observations"):
        measure(Series((date(2024, 1, 1), date(2025, 1, 1)), (100, 110)), periods_per_year=1)
    path = quarterly((100, 101, 102, 103, 104))
    for values in [(), (100, 0, 102, 103, 104), (100, -1, 102, 103, 104)]:
        series = Series(path.times if values else (), values)
        with pytest.raises(ValueError, match="positive"):
            PerformanceIndex("Index", series)
    # Equal return end-times do not excuse a different initial valuation time.
    cash = Series((date(2023, 12, 31), *path.times[1:]), path.values)
    with pytest.raises(ValueError, match="observation times"):
        measure(path, periods_per_year=4, risk_free=cash)

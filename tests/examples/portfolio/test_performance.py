"""Performance indices, calendar statistics, and benchmark comparisons."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from math import prod, sqrt
from statistics import mean, stdev

import pytest

from examples.compute.series import Series
from examples.portfolio.performance import PerformanceIndex, measure


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

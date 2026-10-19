"""Measure aligned total-return indices independently of portfolio holdings."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import sqrt
from statistics import mean, variance
from typing import Generic

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, Time


@dataclass(frozen=True)
class Performance(Generic[Time]):
    """Observed path statistics, with annualized performance withheld below one year."""

    start: Time
    end: Time
    return_count: int
    elapsed_days: float
    total_return: float
    mean_period_return: float
    annualized_volatility: float
    maximum_drawdown: float
    annualization_eligible: bool
    cagr: float | None
    sharpe_ratio: float | None


def measure(index: Series[Time], *, periods_per_year: int, risk_free: Series[Time] | None = None) -> Performance[Time]:
    """Use simple interval returns, sample variance, and an explicit cash index for Sharpe.

    The caller supplies a regular observation schedule and its annual frequency.
    CAGR uses elapsed calendar time (ACT/365.25). CAGR and annualized Sharpe are
    reported after a full calendar year; zero excess-return variance has no Sharpe.
    Annualized volatility remains available as a descriptive sample risk estimate.
    """
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    returns = index.simple_returns()
    if len(returns.values) < 2:
        raise ValueError("Sample risk estimates need at least three price observations")
    if risk_free is not None and risk_free.times != index.times:
        raise ValueError("Risk-free and performance indices must have the same observation times")
    excess = returns.combine(risk_free.simple_returns(), lambda a, b: a - b) if risk_free is not None else None
    start, end = index.times[0], index.times[-1]
    try:
        anniversary = start.replace(year=start.year + 1)
    except ValueError:  # February 29 in a non-leap following year.
        anniversary = start.replace(year=start.year + 1, day=28)
    eligible = end >= anniversary
    days = (end - start) / timedelta(days=1)
    growth = index.values[-1] / index.values[0]
    excess_variance = variance(excess.values) if excess is not None else 0.0
    sharpe = (
        mean(excess.values) / sqrt(excess_variance) * sqrt(periods_per_year)
        if eligible and excess is not None and excess_variance > 0
        else None
    )
    return Performance(
        start,
        end,
        len(returns.values),
        days,
        growth - 1,
        mean(returns.values),
        sqrt(variance(returns.values) * periods_per_year),
        max(0.0, -min(index.drawdowns().values)),
        eligible,
        growth ** (365.25 / days) - 1 if eligible else None,
        sharpe,
    )


@dataclass(frozen=True)
class PerformanceIndex(Generic[Time]):
    """A named, positive total-return index in one reporting currency."""

    name: str
    levels: Series[Time]
    currency: str = "USD"

    def __post_init__(self) -> None:
        if not self.name or not self.currency:
            raise ValueError("A performance index needs a name and reporting currency")
        if not self.levels.values or any(value <= 0 for value in self.levels.values):
            raise ValueError("A performance index needs strictly positive levels")

    def compare(
        self,
        benchmark: PerformanceIndex[Time],
        *,
        risk_free: PerformanceIndex[Time],
        periods_per_year: int,
    ) -> Comparison[Time]:
        """Compare matching total-return observations without implicit filling or currency conversion."""
        if self.currency != benchmark.currency or self.currency != risk_free.currency:
            raise ValueError("Performance indices must use the same reporting currency")
        if self.levels.times != benchmark.levels.times or self.levels.times != risk_free.levels.times:
            raise ValueError("Performance indices must have the same observation times")
        account = measure(self.levels, periods_per_year=periods_per_year, risk_free=risk_free.levels)
        reference = measure(benchmark.levels, periods_per_year=periods_per_year, risk_free=risk_free.levels)
        return Comparison(
            self.name,
            benchmark.name,
            risk_free.name,
            self.currency,
            periods_per_year,
            account,
            reference,
            NamedMap({"portfolio": self.levels, "benchmark": benchmark.levels}).map(
                lambda series: series.map(lambda value: 100 * value / series.values[0])
            ),
            self.levels.simple_returns().combine(benchmark.levels.simple_returns(), lambda a, b: a - b),
            account.total_return - reference.total_return,
            (1 + account.total_return) / (1 + reference.total_return) - 1,
        )


@dataclass(frozen=True)
class Comparison(Generic[Time]):
    """Two total-return paths on the same schedule, measured against the same cash index."""

    portfolio_name: str
    benchmark_name: str
    risk_free_name: str
    currency: str
    periods_per_year: int
    portfolio: Performance[Time]
    benchmark: Performance[Time]
    normalized: NamedMap[Series[Time]]
    active_returns: Series[Time]
    return_difference: float
    relative_return: float

"""Portfolio statistics over a performance index and named asset-return vectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import covariance, mean, variance
from typing import Generic

from examples.named_members import NamedMap
from examples.series import Series, Time


@dataclass(frozen=True)
class PositionPath(Generic[Time]):
    """A fixed holding's value, weight, and interval return attribution."""

    value: Series[Time]
    weight: Series[Time]
    returns: Series[Time]
    return_contribution: Series[Time]


@dataclass(frozen=True)
class NamedRisk:
    """Fixed-reference-weight volatility and its named equity contributions."""

    volatility: float
    contributions: NamedMap[float]
    cash_contribution: float = 0.0


@dataclass(frozen=True)
class NamedAnalytics:
    """Path statistics and covariance indexed by two member-name axes."""

    total_return: float
    mean_period_return: float
    annualized_volatility: float
    maximum_drawdown: float
    annualized_covariance: NamedMap[NamedMap[float]]
    latest_weights_risk: NamedRisk


@dataclass(frozen=True)
class NamedPortfolioPath(Generic[Time]):
    """Named equity paths with a common observation axis and USD cash."""

    positions: NamedMap[PositionPath[Time]]
    cash_value: Series[Time]
    cash_weight: Series[Time]
    nav: Series[Time]

    @property
    def values(self) -> NamedMap[Series[Time]]:
        """Expose value vectors using the same position names."""
        return self.positions.map(lambda position: position.value)

    @property
    def weights(self) -> NamedMap[Series[Time]]:
        """Expose weight vectors using the same position names."""
        return self.positions.map(lambda position: position.weight)

    def analytics(self, *, periods_per_year: int) -> NamedAnalytics:
        """Use the shared path statistics and fixed-reference-weight risk estimate."""
        return analyse(
            self.nav,
            self.positions.map(lambda position: position.returns),
            self.weights.map(lambda series: series.values[-1]),
            periods_per_year=periods_per_year,
        )


def analyse(
    index: Series[Time],
    asset_returns: NamedMap[Series[Time]],
    latest_weights: NamedMap[float],
    *,
    periods_per_year: int,
) -> NamedAnalytics:
    """Measure index returns and separate reference-weight asset risk."""
    if isinstance(periods_per_year, bool) or not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer")
    returns = index.simple_returns()
    if len(returns.values) < 2:
        raise ValueError("Sample risk estimates need at least three price observations")
    if any(series.times != returns.times for series in asset_returns.values()):
        raise ValueError("Asset returns must match the performance return axis")
    weighted = latest_weights.combine(asset_returns, lambda weight, series: series.map(lambda value: weight * value))
    fixed_returns = tuple(sum(series.values[i] for series in weighted.values()) for i in range(len(returns.values)))
    cov = asset_returns.map(
        lambda left: asset_returns.map(lambda right: covariance(left.values, right.values) * periods_per_year)
    )
    volatility = sqrt(variance(fixed_returns) * periods_per_year)
    # Cov(r_i, weighted returns) avoids cancellation from multiplying a rounded matrix.
    contributions = latest_weights.combine(
        asset_returns,
        lambda weight, series: (
            weight * covariance(series.values, fixed_returns) * periods_per_year / volatility if volatility else 0.0
        ),
    )
    return NamedAnalytics(
        index.values[-1] / index.values[0] - 1,
        mean(returns.values),
        sqrt(variance(returns.values) * periods_per_year),
        -min(index.drawdowns().values),
        cov,
        NamedRisk(volatility, contributions),
    )

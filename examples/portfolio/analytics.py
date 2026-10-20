"""Portfolio statistics over a performance index and named asset-return vectors."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import covariance, variance
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, Time
from examples.portfolio.performance import Performance, measure

if TYPE_CHECKING:
    from examples.model_plan import ModelPath
    from examples.portfolio.valuation import NamedPortfolio

_Portfolio = TypeVar("_Portfolio", bound="NamedPortfolio[Any]")


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
class NamedAnalytics(Generic[Time]):
    """Path statistics and covariance indexed by two member-name axes."""

    performance: Performance[Time]
    annualized_covariance: NamedMap[NamedMap[float]]
    latest_weights_risk: NamedRisk


@dataclass(frozen=True)
class NamedPortfolioPath(Generic[Time]):
    """Named equity paths with a common observation axis and USD cash."""

    positions: NamedMap[PositionPath[Time]]
    cash_value: Series[Time]
    cash_weight: Series[Time]
    nav: Series[Time]

    @classmethod
    def from_model(cls, path: ModelPath[_Portfolio, Time]) -> NamedPortfolioPath[Time]:
        """Project fixed holdings' valuation columns into interval return analytics."""
        model = path.view.source
        if any(instrument.currency != "USD" for instrument in model.instruments.values()):
            raise ValueError("The example requires USD equities and cash")
        if (
            any(len(set(path.column(holding.quantity).values)) > 1 for holding in model.positions.values())
            or len(set(path.column(model.cash.amount).values)) > 1
        ):
            raise ValueError("Fixed-holding return attribution requires constant quantities and cash")

        def position(name: str) -> PositionPath[Time]:
            holding = model.positions[name]
            value = path.column(holding.market_value)
            weight = path.column(holding.weight)
            returns = path.column(holding.price).simple_returns()
            contribution = Series(
                returns.times, tuple(w * r for w, r in zip(weight.values[:-1], returns.values, strict=True))
            )
            return PositionPath(value, weight, returns, contribution)

        return cls(
            NamedMap((name, position(name)) for name in model.positions),
            path.column(model.cash.amount),
            path.column(model.cash_weight),
            path.column(model.nav),
        )

    @property
    def values(self) -> NamedMap[Series[Time]]:
        """Expose value vectors using the same position names."""
        return self.positions.map(lambda position: position.value)

    @property
    def weights(self) -> NamedMap[Series[Time]]:
        """Expose weight vectors using the same position names."""
        return self.positions.map(lambda position: position.weight)

    def analytics(self, *, periods_per_year: int, risk_free: Series[Time] | None = None) -> NamedAnalytics[Time]:
        """Use the shared path statistics and fixed-reference-weight risk estimate."""
        return analyse(
            self.nav,
            self.positions.map(lambda position: position.returns),
            self.weights.map(lambda series: series.values[-1]),
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )


def analyse(
    index: Series[Time],
    asset_returns: NamedMap[Series[Time]],
    latest_weights: NamedMap[float],
    *,
    periods_per_year: int,
    risk_free: Series[Time] | None = None,
) -> NamedAnalytics[Time]:
    """Measure index returns and separate reference-weight asset risk."""
    performance = measure(index, periods_per_year=periods_per_year, risk_free=risk_free)
    returns = index.simple_returns()
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
        performance,
        cov,
        NamedRisk(volatility, contributions),
    )

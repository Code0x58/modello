"""Causal signals and an allocation-to-order boundary, separate from accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, utc_time
from examples.portfolio.account import AccountSnapshot
from examples.portfolio.models import Equity
from examples.portfolio.sources import finite


class SignalSource(Protocol):
    """Signals indexed by when they become usable by a decision."""

    def load(self) -> NamedMap[Series[datetime]]: ...


@dataclass(frozen=True)
class MomentumSignals:
    """An observation-count momentum feature, computed only from trailing prices."""

    prices: NamedMap[Series[datetime]]
    lookback: int

    def load(self) -> NamedMap[Series[datetime]]:
        def momentum(prices: Series[datetime]) -> Series[datetime]:
            earlier = prices.lag(self.lookback)
            current = Series(earlier.times, prices.values[self.lookback :])
            return current.combine(earlier, lambda end, start: end / start - 1)

        return self.prices.map(momentum)


@dataclass(frozen=True)
class TargetWeights:
    """An allocation policy's result; cash is the residual weight."""

    at: datetime
    weights: NamedMap[float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", utc_time(self.at))
        finite(*self.weights.values())
        if any(weight < 0 for weight in self.weights.values()) or sum(self.weights.values()) > 1 + 1e-12:
            raise ValueError("Target weights must be nonnegative and sum to at most one")


def positive_signal_targets(
    signals: NamedMap[Series[datetime]], at: datetime, *, exposure: float = 0.8
) -> TargetWeights:
    """An illustrative equal-weight policy, leaving a cash reserve."""
    finite(exposure)
    if not 0 <= exposure <= 1:
        raise ValueError("Exposure must be between zero and one")
    active = signals.map(lambda signal: signal.asof(utc_time(at)) > 0)
    count = sum(active.values())
    return TargetWeights(at, active.map(lambda selected: exposure / count if selected else 0.0))


@dataclass(frozen=True)
class OrderIntent:
    """A model decision, distinct from subsequent externally reported fills."""

    id: str
    account: str
    member: str
    equity: Equity
    decision_at: datetime
    arrival_at: datetime
    quantity: float
    model_price: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision_at", utc_time(self.decision_at))
        object.__setattr__(self, "arrival_at", utc_time(self.arrival_at))
        finite(self.quantity, self.model_price)
        if not self.id or not self.account or not self.member or not self.quantity or self.model_price <= 0:
            raise ValueError("Orders need identities, nonzero shares, and a positive model price")
        if self.arrival_at < self.decision_at:
            raise ValueError("Order arrival must follow its decision")


def rebalance_orders(
    snapshot: AccountSnapshot,
    targets: TargetWeights,
    *,
    batch: str,
    arrival_at: datetime,
) -> NamedMap[OrderIntent]:
    """Translate targets to fractional-share intentions; submit no actual orders."""
    if snapshot.at != targets.at or snapshot.known_at > targets.at:
        raise ValueError("Use an account snapshot evaluated with knowledge at the decision time")
    quantities = targets.weights.combine(snapshot.quantities, lambda weight, held: (weight, held))
    orders = []
    for name, (weight, held) in quantities.items():
        price = snapshot.prices[name]
        quantity = weight * snapshot.nav / price - held
        if abs(quantity) > 1e-12:
            order_id = f"{batch}:{name}"
            orders.append(
                (
                    order_id,
                    OrderIntent(
                        order_id,
                        snapshot.account,
                        name,
                        snapshot.instruments[name],
                        targets.at,
                        arrival_at,
                        quantity,
                        price,
                    ),
                )
            )
    return NamedMap(orders)

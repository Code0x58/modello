"""Causal signals and an allocation-to-order boundary, separate from accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, utc_time
from examples.model_plan import Known
from examples.portfolio.account import AccountSnapshot
from examples.portfolio.models import AccountValue, Allocation, Equity, EquityPosition, PriceChange, Trade
from examples.portfolio.sources import finite
from examples.symbolic import SymbolicModel
from modello import component


class Momentum(SymbolicModel):
    prices = component(PriceChange)
    opening_price = prices.before.price
    closing_price = prices.after.price
    signal = prices.simple_return


class Rebalance(SymbolicModel):
    """A target allocation and its signed order are views of the same equation."""

    before = component(EquityPosition)
    after = component(EquityPosition)
    account = component(AccountValue)
    allocation = component(Allocation)
    order = component(Trade)

    target_quantity = after.quantity
    nav = account.nav
    nav = allocation.capital
    price = before.instrument.price
    price = after.instrument.price
    price = order.price
    opening_quantity = before.quantity
    target_weight = allocation.weight
    target_value = after.market_value
    target_value = allocation.value
    order_quantity = order.quantity
    order_quantity = after.quantity - before.quantity


class SignalSource(Protocol):
    """Signals indexed by when they become usable by a decision."""

    def load(self) -> NamedMap[Series[datetime]]: ...


@dataclass(frozen=True)
class MomentumSignals:
    """An observation-count momentum feature, computed only from trailing prices."""

    prices: NamedMap[Series[datetime]]
    lookback: int

    def load(self) -> NamedMap[Series[datetime]]:
        view = Momentum.plan()
        model = view.source

        def momentum(prices: Series[datetime]) -> Series[datetime]:
            earlier = prices.lag(self.lookback)
            current = Series(earlier.times, prices.values[self.lookback :])
            path = view.over(earlier.times, Known(model.opening_price, earlier), Known(model.closing_price, current))
            return path.column(model.signal)

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

    @classmethod
    def from_signals(cls, signals: NamedMap[Series[datetime]], at: datetime, *, exposure: float = 0.8) -> TargetWeights:
        """An illustrative equal-weight policy, leaving a cash reserve."""
        finite(exposure)
        if not 0 <= exposure <= 1:
            raise ValueError("Exposure must be between zero and one")
        active = signals.map(lambda signal: signal.asof(utc_time(at)) > 0)
        count = sum(active.values())
        return cls(at, active.map(lambda selected: exposure / count if selected else 0.0))

    def orders(
        self,
        snapshot: AccountSnapshot,
        *,
        batch: str,
        arrival_at: datetime,
    ) -> NamedMap[OrderIntent]:
        """Translate targets to fractional-share intentions; submit no actual orders."""
        if snapshot.at != self.at or snapshot.known_at > self.at:
            raise ValueError("Use an account snapshot evaluated with knowledge at the decision time")
        quantities = self.weights.combine(snapshot.quantities, lambda weight, held: (weight, held))
        view = Rebalance.plan()
        model = view.source
        orders = []
        for name, (weight, held) in quantities.items():
            price = snapshot.prices[name]
            result = view.evaluate(
                Known(model.nav, snapshot.nav),
                Known(model.price, price),
                Known(model.opening_quantity, held),
                Known(model.target_weight, weight),
            )
            quantity = float(result.order_quantity)
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
                            self.at,
                            arrival_at,
                            quantity,
                            price,
                        ),
                    )
                )
        return NamedMap(orders)


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

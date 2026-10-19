"""Benchmark externally reported fills against recorded decisions and market prints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from examples.compute.named_map import NamedMap
from examples.compute.series import utc_time
from examples.portfolio.signals import OrderIntent
from examples.portfolio.sources import Fill, PriceHistory, TransactionLog


@dataclass(frozen=True)
class ExecutionBenchmark:
    """Positive costs are adverse for both buys and sells; monetary values are USD."""

    order: OrderIntent
    assessed_at: datetime
    filled_quantity: float
    remaining_quantity: float
    execution_vwap: float | None
    market_vwap: float | None
    decision_price: float
    arrival_price: float
    model_slippage_cost: float
    model_slippage_bps: float | None
    arrival_slippage_bps: float | None
    market_vwap_slippage_bps: float | None
    delay_cost: float
    trading_cost: float
    fees: float
    opportunity_cost: float

    @property
    def implementation_shortfall(self) -> float:
        return self.delay_cost + self.trading_cost + self.fees + self.opportunity_cost


def benchmark_executions(
    orders: NamedMap[OrderIntent],
    ledger: TransactionLog,
    market: NamedMap[PriceHistory],
    *,
    at: datetime,
    max_price_age: timedelta,
) -> NamedMap[ExecutionBenchmark]:
    """Assess filled and unfilled shares over the explicit [arrival, at] window."""
    at = utc_time(at)
    if len({order.id for order in orders.values()}) != len(orders):
        raise ValueError("Execution analysis needs unique order IDs")

    def benchmark(order: OrderIntent) -> ExecutionBenchmark:
        if at < order.arrival_at:
            raise ValueError("Execution assessment must be at or after order arrival")
        history = market[order.member]
        if history.equity != order.equity:
            raise ValueError("Execution benchmark instrument must match the model order")
        decision = history.mark(order.decision_at, max_age=max_price_age)
        arrival = history.mark(order.arrival_at, max_age=max_price_age)
        terminal = history.mark(at, max_age=max_price_age)
        fills = tuple(
            event
            for event in ledger.known(order.account, at).events
            if isinstance(event, Fill) and event.order_id == order.id
        )
        side = 1 if order.quantity > 0 else -1
        if any(
            fill.member != order.member or fill.quantity * side <= 0 or fill.at < order.arrival_at for fill in fills
        ):
            raise ValueError("Fills must match their order's member, side, and arrival time")
        quantity = sum(abs(fill.quantity) for fill in fills)
        if quantity > abs(order.quantity) + 1e-10:
            raise ValueError("Executed quantity exceeds the model order")
        remaining = max(0.0, abs(order.quantity) - quantity)
        notional = sum(abs(fill.quantity) * fill.price for fill in fills)
        fees = sum(fill.fee for fill in fills)
        execution_vwap = notional / quantity if quantity else None
        market_vwap = history.vwap(order.arrival_at, at)
        model_cost = side * (notional - quantity * order.model_price) + fees
        return ExecutionBenchmark(
            order,
            at,
            quantity,
            remaining,
            execution_vwap,
            market_vwap,
            decision,
            arrival,
            model_cost,
            model_cost / (quantity * order.model_price) * 10000 if quantity else None,
            side * (execution_vwap / arrival - 1) * 10000 if execution_vwap is not None else None,
            side * (execution_vwap / market_vwap - 1) * 10000
            if execution_vwap is not None and market_vwap is not None
            else None,
            quantity * side * (arrival - decision),
            side * (notional - quantity * arrival),
            fees,
            remaining * side * (terminal - decision),
        )

    return orders.map(benchmark)

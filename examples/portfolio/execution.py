"""Benchmark externally reported fills against recorded decisions and market prints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import ClassVar

from examples.compute.named_map import NamedMap
from examples.compute.series import utc_time
from examples.portfolio.signals import OrderIntent
from examples.portfolio.sources import Fill, PriceHistory, TransactionLog
from examples.symbolic import SymbolicModel
from modello import InstanceDummy


class Shortfall(SymbolicModel):
    delay_cost = InstanceDummy("delay_cost", real=True)
    trading_cost = InstanceDummy("trading_cost", real=True)
    fees = InstanceDummy("fees", nonnegative=True)
    opportunity_cost = InstanceDummy("opportunity_cost", real=True)
    implementation_shortfall = delay_cost + trading_cost + fees + opportunity_cost

    inputs: ClassVar[tuple[str, ...]] = ("delay_cost", "trading_cost", "fees", "opportunity_cost")
    outputs: ClassVar[tuple[str, ...]] = ("implementation_shortfall",)


class ExecutionCosts(Shortfall):
    """Execution and opportunity costs share one symbolic accounting definition."""

    quantity = InstanceDummy("quantity", nonnegative=True)
    remaining = InstanceDummy("remaining", nonnegative=True)
    side = InstanceDummy("side", real=True)
    notional = InstanceDummy("notional", nonnegative=True)
    model_price = InstanceDummy("model_price", positive=True)
    decision_price = InstanceDummy("decision_price", positive=True)
    arrival_price = InstanceDummy("arrival_price", positive=True)
    terminal_price = InstanceDummy("terminal_price", positive=True)
    model_slippage_cost = side * (notional - quantity * model_price) + Shortfall.fees
    delay_cost = quantity * side * (arrival_price - decision_price)
    trading_cost = side * (notional - quantity * arrival_price)
    opportunity_cost = remaining * side * (terminal_price - decision_price)

    inputs: ClassVar[tuple[str, ...]] = (
        "quantity",
        "remaining",
        "side",
        "notional",
        "model_price",
        "decision_price",
        "arrival_price",
        "terminal_price",
        "fees",
    )
    outputs: ClassVar[tuple[str, ...]] = (
        "model_slippage_cost",
        "delay_cost",
        "trading_cost",
        "fees",
        "opportunity_cost",
        "implementation_shortfall",
    )


class Slippage(SymbolicModel):
    side = InstanceDummy("side", real=True)
    execution_price = InstanceDummy("execution_price", positive=True)
    reference_price = InstanceDummy("reference_price", positive=True)
    bps = side * (execution_price / reference_price - 1) * 10000

    inputs: ClassVar[tuple[str, ...]] = ("side", "execution_price", "reference_price")
    outputs: ClassVar[tuple[str, ...]] = ("bps",)


class CostRate(SymbolicModel):
    cost = InstanceDummy("cost", real=True)
    notional = InstanceDummy("notional", positive=True)
    bps = cost / notional * 10000

    inputs: ClassVar[tuple[str, ...]] = ("cost", "notional")
    outputs: ClassVar[tuple[str, ...]] = ("bps",)


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
        return Shortfall.plan().evaluate(
            NamedMap(
                {
                    "delay_cost": self.delay_cost,
                    "trading_cost": self.trading_cost,
                    "fees": self.fees,
                    "opportunity_cost": self.opportunity_cost,
                }
            )
        )["implementation_shortfall"]

    @classmethod
    def assess(
        cls,
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
            costs = ExecutionCosts.plan().evaluate(
                NamedMap(
                    {
                        "quantity": quantity,
                        "remaining": remaining,
                        "side": side,
                        "notional": notional,
                        "model_price": order.model_price,
                        "decision_price": decision,
                        "arrival_price": arrival,
                        "terminal_price": terminal,
                        "fees": fees,
                    }
                )
            )
            model_cost = costs["model_slippage_cost"]
            return cls(
                order,
                at,
                quantity,
                remaining,
                execution_vwap,
                market_vwap,
                decision,
                arrival,
                model_cost,
                CostRate.plan().evaluate(NamedMap({"cost": model_cost, "notional": quantity * order.model_price}))[
                    "bps"
                ]
                if quantity
                else None,
                Slippage.plan().evaluate(
                    NamedMap({"side": side, "execution_price": execution_vwap, "reference_price": arrival})
                )["bps"]
                if execution_vwap is not None
                else None,
                Slippage.plan().evaluate(
                    NamedMap({"side": side, "execution_price": execution_vwap, "reference_price": market_vwap})
                )["bps"]
                if execution_vwap is not None and market_vwap is not None
                else None,
                costs["delay_cost"],
                costs["trading_cost"],
                fees,
                costs["opportunity_cost"],
            )

        return orders.map(benchmark)

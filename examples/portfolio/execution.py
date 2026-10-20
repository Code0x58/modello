"""Benchmark externally reported fills against recorded decisions and market prints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from examples.compute.named_map import NamedMap
from examples.compute.series import utc_time
from examples.model_plan import Known
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


class Slippage(SymbolicModel):
    side = InstanceDummy("side", real=True)
    execution_price = InstanceDummy("execution_price", positive=True)
    reference_price = InstanceDummy("reference_price", positive=True)
    bps = side * (execution_price / reference_price - 1) * 10000


class CostRate(SymbolicModel):
    cost = InstanceDummy("cost", real=True)
    notional = InstanceDummy("notional", positive=True)
    bps = cost / notional * 10000


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
        view = Shortfall.plan()
        model = view.source
        result = view.evaluate(
            Known(model.delay_cost, self.delay_cost),
            Known(model.trading_cost, self.trading_cost),
            Known(model.fees, self.fees),
            Known(model.opportunity_cost, self.opportunity_cost),
        )
        return float(result.implementation_shortfall)

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
        costs_view = ExecutionCosts.plan()
        model = costs_view.source
        rate_view = CostRate.plan()
        rate = rate_view.source
        slippage_view = Slippage.plan()
        slippage = slippage_view.source

        def slippage_bps(side: float, price: float, reference: float) -> float:
            result = slippage_view.evaluate(
                Known(slippage.side, side),
                Known(slippage.execution_price, price),
                Known(slippage.reference_price, reference),
            )
            return float(result.bps)

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
            costs = costs_view.evaluate(
                Known(model.quantity, quantity),
                Known(model.remaining, remaining),
                Known(model.side, side),
                Known(model.notional, notional),
                Known(model.model_price, order.model_price),
                Known(model.decision_price, decision),
                Known(model.arrival_price, arrival),
                Known(model.terminal_price, terminal),
                Known(model.fees, fees),
            )
            model_cost = float(costs.model_slippage_cost)
            model_bps = (
                float(
                    rate_view.evaluate(
                        Known(rate.cost, model_cost), Known(rate.notional, quantity * order.model_price)
                    ).bps
                )
                if quantity
                else None
            )
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
                model_bps,
                slippage_bps(side, execution_vwap, arrival) if execution_vwap is not None else None,
                slippage_bps(side, execution_vwap, market_vwap)
                if execution_vwap is not None and market_vwap is not None
                else None,
                float(costs.delay_cost),
                float(costs.trading_cost),
                fees,
                float(costs.opportunity_cost),
            )

        return orders.map(benchmark)

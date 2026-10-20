"""Benchmark externally reported fills against recorded decisions and market prints."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sympy import S

from examples.compute.named_map import NamedMap
from examples.compute.series import utc_time
from examples.model_plan import Known
from examples.portfolio.models import PriceChange, Quote
from examples.portfolio.signals import OrderIntent
from examples.portfolio.sources import Fill, PriceHistory, TransactionLog
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, component


class Shortfall(SymbolicModel):
    delay_cost = InstanceDummy("delay_cost", real=True)
    trading_cost = InstanceDummy("trading_cost", real=True)
    fees = InstanceDummy("fees", nonnegative=True)
    opportunity_cost = InstanceDummy("opportunity_cost", real=True)
    implementation_shortfall = delay_cost + trading_cost + fees + opportunity_cost


class PriceCost(Modello):
    """The directional cost of a price movement on a number of shares."""

    prices = component(PriceChange)
    quantity = InstanceDummy("quantity", nonnegative=True)
    side = InstanceDummy("side", real=True)
    cost = quantity * side * prices.change


class FillCost(Modello):
    """A fill total against a reference quote, including unfilled intervals."""

    reference = component(Quote)
    quantity = InstanceDummy("quantity", nonnegative=True)
    notional = InstanceDummy("notional", nonnegative=True)
    side = InstanceDummy("side", real=True)
    fees = InstanceDummy("fees", nonnegative=True)
    cost = side * (notional - quantity * reference.price) + fees


class ExecutionCosts(Shortfall):
    """Bind delay, fills, and unexecuted shares to one shortfall account."""

    shortfall = component(Shortfall)
    delay = component(PriceCost)
    trading = component(FillCost)
    opportunity = component(PriceCost)
    model_slippage = component(FillCost)

    delay_cost = shortfall.delay_cost
    delay_cost = delay.cost
    trading_cost = shortfall.trading_cost
    trading_cost = trading.cost
    opportunity_cost = shortfall.opportunity_cost
    opportunity_cost = opportunity.cost
    implementation_shortfall = shortfall.implementation_shortfall
    fees = shortfall.fees
    fees = model_slippage.fees

    quantity = delay.quantity
    quantity = trading.quantity
    quantity = model_slippage.quantity
    remaining = opportunity.quantity
    side = delay.side
    side = trading.side
    side = opportunity.side
    side = model_slippage.side
    notional = trading.notional
    notional = model_slippage.notional
    model_price = model_slippage.reference.price
    decision_price = delay.prices.before.price
    decision_price = opportunity.prices.before.price
    arrival_price = delay.prices.after.price
    arrival_price = trading.reference.price
    terminal_price = opportunity.prices.after.price
    trading_fees = trading.fees
    trading_fees = S.Zero
    model_slippage_cost = model_slippage.cost


class Slippage(SymbolicModel):
    prices = component(PriceChange)
    side = InstanceDummy("side", real=True)
    execution_price = prices.after.price
    reference_price = prices.before.price
    bps = side * prices.simple_return * 10000


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

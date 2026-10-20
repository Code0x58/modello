# mypy: warn-unused-ignores
"""Joint inference and compilation through composed economic models."""

from typing import TYPE_CHECKING

import pytest
from sympy import Rational, Symbol, simplify

from examples.model_plan import Known
from examples.portfolio.models import AccountValue, Allocation, EquityPosition, Position, Quote, Trade
from examples.portfolio.pnl import AccountPnL, InstrumentPnL, Profit, TradingActivity
from examples.portfolio.postings import CashMovement, Execution, FillPosting, HoldingState
from examples.portfolio.signals import Rebalance
from examples.symbolic import SymbolicModel
from modello import component


class ExecutedProfit(SymbolicModel):
    posting = component(FillPosting)
    pnl = component(InstrumentPnL)

    opening_quantity = pnl.before.quantity
    opening_quantity = posting.before.quantity
    filled = pnl.activity.quantity
    filled = posting.fill.quantity
    notional = pnl.activity.notional
    notional = posting.fill.notional
    fees = pnl.activity.fees
    fees = posting.fill.fee


def test_profit_observation_can_infer_execution_and_settlement_through_shared_components():
    source = ExecutedProfit(
        "inferred_fill",
        posting=FillPosting(
            "fill", before=HoldingState("opening", quantity=0, cash=1000), fill=Execution("buy", quantity=7, fee=1)
        ),
        pnl=InstrumentPnL(
            "observed",
            before=Position("opening", instrument=Quote("quote", price=100)),
            after=Position("closing", instrument=Quote("quote", price=107)),
            activity=TradingActivity("totals", income=Rational("3.5")),
            profit=Profit("observed", net=Symbol("observed_profit", real=True)),
        ),
    )
    result = source.compile().evaluate(Known(source.pnl.profit.net, 15.9))
    assert float(result.posting.fill.price) == pytest.approx(735.6 / 7)
    assert float(result.posting.after.cash) == pytest.approx(263.4)
    assert float(result.posting.after.quantity) == float(result.pnl.after.quantity) == 7
    assert float(result.pnl.activity.notional) == pytest.approx(735.6)
    assert type(result.pnl.profit) is Profit
    assert type(result.pnl.before) is type(result.pnl.after) is Position
    assert source.posting.fill.price.free_symbols


def test_account_reconciliation_solves_the_missing_state_from_profit_and_capital():
    result = AccountPnL(
        "closing",
        before=AccountValue("opening", nav=1000),
        capital=CashMovement("deposit", amount=500),
        profit=Profit("profit", holding=20, trading=-2, income=3, fees=1),
        residual=0,
    )
    assert result.after.nav == 1520
    assert result.profit.net == 20
    assert result.nav_change == 520


def test_allocation_and_marked_position_jointly_determine_the_order_or_weight():
    opening = EquityPosition("opening", quantity=3, instrument=Quote("quote", price=100))
    result = Rebalance(
        "target",
        before=opening,
        account=AccountValue("account", nav=1000),
        allocation=Allocation("allocation", weight=Rational(1, 2)),
    )
    assert result.after.quantity == 5
    assert result.after.market_value == result.allocation.value == 500
    assert result.order.quantity == 2
    assert result.order.price == 100
    inverse = Rebalance(
        "observed", before=opening, account=AccountValue("account", nav=1000), order=Trade("order", quantity=2)
    )
    assert inverse.allocation.weight == Rational(1, 2)
    symbolic = Rebalance("identity")
    assert simplify(symbolic.after.market_value - symbolic.allocation.value) == 0
    assert simplify(symbolic.order.quantity - symbolic.after.quantity + symbolic.before.quantity) == 0


if TYPE_CHECKING:

    def typed_components(model: ExecutedProfit, allocation: Rebalance) -> None:
        profit: Profit = model.pnl.profit
        position: Position = model.pnl.after
        order: Trade = allocation.order
        target: EquityPosition = allocation.after
        assert profit and position and order and target
        _wrong: Execution = model.pnl.profit  # type: ignore[assignment]

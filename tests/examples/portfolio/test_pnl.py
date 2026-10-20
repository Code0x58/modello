"""Shared symbolic profit models, sensitivities, and inverse reconciliation."""

import pytest
from sympy import Rational, Symbol, diff, simplify

from examples.model_plan import Known
from examples.portfolio.models import Position, Quote
from examples.portfolio.pnl import AccountPnL, InstrumentPnL, Profit, TradingActivity
from examples.portfolio.postings import Execution, FillPosting, HoldingState
from examples.symbolic import SymbolicModel
from modello import component


def test_reporting_models_support_symbolic_sensitivities_and_inverse_reconciliation():
    instrument = InstrumentPnL("instrument")
    assert (
        simplify(
            diff(instrument.net, instrument.opening_quantity) - (instrument.closing_price - instrument.opening_price)
        )
        == 0
    )
    assert (
        simplify(
            diff(instrument.net, instrument.closing_price) - instrument.opening_quantity - instrument.fill_quantity
        )
        == 0
    )
    assert diff(instrument.net, instrument.fees) == -1
    account = AccountPnL(
        "account",
        opening_nav=2500,
        external_flows=500,
        holding=Rational(57),
        trading=Rational(17, 5),
        income=Rational(17, 2),
        fees=2,
        residual=0,
    )
    assert account.net == Rational(669, 10)
    assert account.closing_nav == Rational(30669, 10)


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

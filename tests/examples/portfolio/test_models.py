"""Joint inference through financial transitions and profit components."""

from __future__ import annotations

from datetime import date

import pytest
from sympy import Rational, Symbol, diff, simplify

from examples.data import Series
from examples.portfolio.models import (
    AccountFill,
    AccountPnL,
    AccountState,
    CapitalFlow,
    CashMovement,
    Execution,
    FillPosting,
    HoldingState,
    IncomePosting,
    InstrumentPnL,
    Position,
    Profit,
    Quote,
    TradingActivity,
)
from examples.symbolic import Known, SymbolicModel
from modello import component


class FollowingFill(FillPosting):
    before = component(FillPosting)


class QuotedExecution(Execution):
    quote = component(Quote)
    price = quote.price


class Reinvestment(FillPosting):
    before = component(FollowingFill)
    fill = component(QuotedExecution)


class FillAfterCapital(AccountFill):
    before = component(CapitalFlow)


class IncomeAfterFill(IncomePosting):
    before = component(FillAfterCapital)


def test_mixed_account_events_nest_and_infer_an_upstream_execution_price():
    observed_cash = Symbol("observed_cash", real=True)
    source = IncomeAfterFill(
        "deposit_buy_income",
        before=FillAfterCapital(
            "buy",
            before=CapitalFlow(
                "deposit",
                before=AccountState("start", cash=1000, nav=2500, units=25),
                flow=CashMovement("capital", amount=500),
            ),
            settlement=FillPosting(
                "settlement",
                before=HoldingState("holding", quantity=3),
                fill=Execution("execution", quantity=2, fee=1),
            ),
            mark=Quote("market", price=110),
        ),
        income=CashMovement("dividend", amount=10),
        cash=observed_cash,
    )
    result = source.compile().evaluate(Known(source.cash, 1319))
    assert type(result) is IncomeAfterFill
    assert type(result.before) is FillAfterCapital
    assert type(result.before.before) is CapitalFlow
    assert float(result.before.settlement.fill.price) == 95
    assert float(result.before.settlement.quantity) == 5
    assert float(result.before.before.cash) == 1500
    assert float(result.before.cash) == 1309
    assert float(result.nav) == 3039
    assert float(result.units) == 30
    income = IncomePosting("sensitivity")
    assert simplify(diff(income.nav, income.income.amount) - 1) == 0
    assert source.cash == observed_cash


def test_a_deeper_operation_tree_can_infer_a_quote_inside_its_execution_branch():
    observed_cash = Symbol("closing_cash", real=True)
    source = Reinvestment(
        "reinvest",
        before=FollowingFill(
            "sale",
            before=FillPosting(
                "buy",
                before=HoldingState("opening", quantity=3, cash=1000),
                fill=Execution("buy", quantity=2, price=100, fee=1),
            ),
            fill=Execution("sell", quantity=-1, price=110, fee=1),
        ),
        fill=QuotedExecution("reinvest", quantity=1, fee=1),
        cash=observed_cash,
    )
    view = source.compile()
    result = view.evaluate(Known(source.cash, 817))
    assert type(result.before) is FollowingFill
    assert type(result.before.before) is FillPosting
    assert type(result.fill) is QuotedExecution
    assert type(result.fill.quote) is Quote
    assert float(result.quantity) == 5
    assert float(result.fill.quote.price) == 90
    assert float(result.before.cash) == 908
    assert float(result.before.before.cash) == 799
    fields = {field.path: field for field in view.walk(result)}
    assert fields[("fill", "quote", "price")].value == 90
    assert source.fill.quote.price.free_symbols


@pytest.mark.parametrize("amount", [-1000.0, -1500.0])
def test_invalid_closing_states_are_rejected_at_points_and_on_paths(amount):
    view = CapitalFlow.plan()
    source = view.source
    knowns = (
        Known(source.before.cash, 1000.0),
        Known(source.before.nav, 1000.0),
        Known(source.before.units, 10.0),
        Known(source.flow.amount, amount),
    )
    with pytest.raises(ValueError, match="model assumptions"):
        view.evaluate(*knowns)
    times = (date(2026, 1, 1),)
    with pytest.raises(ValueError, match="model assumptions"):
        view.over(times, *(Known(known.field, Series(times, (known.value,))) for known in knowns))


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

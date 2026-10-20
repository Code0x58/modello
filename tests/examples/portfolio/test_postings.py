"""Typed transitions reuse state equations in forward, inverse, and joint views."""

from datetime import date

import pytest
from sympy import Symbol, diff, simplify

from examples.compute.series import Series
from examples.model_plan import Known
from examples.portfolio.models import Quote
from examples.portfolio.postings import (
    AccountFill,
    AccountState,
    CapitalFlow,
    CashMovement,
    Execution,
    FillPosting,
    HoldingState,
    IncomePosting,
)
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

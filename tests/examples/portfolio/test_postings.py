# mypy: warn-unused-ignores
"""Typed transitions reuse state equations in forward, inverse, and joint views."""

from datetime import date
from typing import TYPE_CHECKING

import pytest
from sympy import Symbol, diff, simplify

from examples.compute.series import Series
from examples.model_plan import Known, ModelPath, ModelPlan
from examples.portfolio.models import Quote
from examples.portfolio.postings import AccountState, CapitalFlow, CashMovement, Execution, FillPosting, HoldingState
from modello import component


def test_a_fill_can_be_valued_or_its_execution_price_inferred_from_cash():
    before = HoldingState("opening", quantity=3, cash=1000)
    posted = FillPosting("fill", before=before, fill=Execution("execution", quantity=2, price=100, fee=1))
    inferred = FillPosting(
        "fit", before=before, fill=Execution("execution", quantity=2, fee=1), after=HoldingState("observed", cash=799)
    )
    assert posted.after.quantity == 5
    assert posted.after.cash == 799
    assert inferred.fill.price == 100
    assert inferred.after.quantity == posted.after.quantity
    assert before.quantity == 3 and before.cash == 1000


def test_external_capital_preserves_the_pre_flow_unit_price_in_both_directions():
    before = AccountState("opening", cash=1000, nav=2500, units=25)
    flow = CapitalFlow("deposit", before=before, flow=CashMovement("external", amount=500))
    inferred = CapitalFlow("fit", before=before, after=AccountState("observed", units=30))
    assert flow.before.unit_price == flow.after.unit_price == 100
    assert flow.after.cash == 1500
    assert flow.after.nav == 3000
    assert flow.after.units == 30
    assert inferred.flow.amount == 500
    assert inferred.after.cash == flow.after.cash
    source = CapitalFlow("identity")
    assert simplify(source.after.unit_price - source.before.unit_price) == 0


@pytest.mark.parametrize("amount", [500.0, -500.0, 0.0])
def test_compiled_capital_movements_preserve_unit_nav_and_state_types(amount):
    view = CapitalFlow.plan()
    source = view.source
    result = view.evaluate(
        Known(source.before.cash, 1000),
        Known(source.before.nav, 2500),
        Known(source.before.units, 25),
        Known(source.flow.amount, amount),
    )
    assert type(result) is CapitalFlow
    assert type(result.before) is type(result.after) is AccountState
    assert type(result.flow) is CashMovement
    assert float(result.after.cash) == 1000 + amount
    assert float(result.after.nav) == 2500 + amount
    assert float(result.after.units) == 25 + amount / 100
    assert float(result.after.unit_price) == 100


def test_compiled_transitions_keep_real_component_paths_and_do_not_solve_again(monkeypatch):
    view = FillPosting.plan()
    source = view.source
    monkeypatch.setattr("modello.solve", lambda *args, **kwargs: pytest.fail("Repeated symbolic solve"))
    result = view.evaluate(
        Known(source.before.quantity, 3),
        Known(source.before.cash, 1000),
        Known(source.fill.quantity, -1),
        Known(source.fill.price, 110),
        Known(source.fill.fee, 1),
    )
    assert type(result) is FillPosting
    assert type(result.before) is type(result.after) is HoldingState
    assert type(result.fill) is Execution
    assert (float(result.after.quantity), float(result.after.cash)) == (2, 1109)
    fields = {field.path: field for field in view.walk(result)}
    assert fields[("before", "quantity")].role == "input"
    assert fields[("after", "cash")].role == "derived"
    assert fields[("after", "cash")].value == 1109
    assert source.after.cash.free_symbols
    assert diff(source.after.cash, source.fill.quantity) == -source.fill.price


class FollowingFill(FillPosting):
    before = component(FillPosting)


class QuotedExecution(Execution):
    quote = component(Quote)
    price = quote.price


class Reinvestment(FillPosting):
    before = component(FollowingFill)
    fill = component(QuotedExecution)


class FollowingCapitalFlow(CapitalFlow):
    before = component(CapitalFlow)


def test_connected_states_support_joint_inference_and_compiled_observation_paths():
    observed_cash = Symbol("observed_cash", real=True)
    source = FollowingFill(
        "observed_chain",
        before=FillPosting(
            "buy", before=HoldingState("start", quantity=3, cash=1000), fill=Execution("buy", quantity=2, fee=1)
        ),
        fill=Execution("sell", quantity=-1, price=110, fee=1),
        after=HoldingState("end", cash=observed_cash),
    )
    assert simplify(source.before.fill.price - (1108 - observed_cash) / 2) == 0
    assert source.before.cash == source.before.after.cash
    times = (date(2026, 1, 1), date(2026, 1, 2))
    path = source.compile().over(times, Known(source.cash, Series(times, (908, 906))))
    assert path.column(source.before.fill.price).values == (100, 101)
    assert path.column(source.quantity).values == (4, 4)
    assert type(path.at(0)) is FollowingFill
    assert type(path.at(0).before) is FillPosting
    assert isinstance(path.at(0), HoldingState)
    assert float(path.at(0).before.cash) == 799
    # The observation axis compares alternative cases; the two child models
    # describe chronological steps within each case.
    assert float(path.at(1).before.cash) == 797


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


def test_capital_movements_also_nest_as_account_states():
    flow = FollowingCapitalFlow(
        "second_deposit",
        before=CapitalFlow(
            "first_deposit",
            before=AccountState("opening", cash=1000, nav=2500, units=25),
            flow=CashMovement("first", amount=500),
        ),
        cash=1600,
    )
    assert isinstance(flow, AccountState)
    assert flow.before.cash == 1500
    assert flow.flow.amount == 100
    assert flow.cash == flow.after.cash == 1600
    assert flow.nav == 3100
    assert flow.units == 31
    assert flow.unit_price == flow.before.unit_price == 100


def test_conflicting_states_are_rejected():
    with pytest.raises(ValueError, match="0 solutions"):
        FillPosting(
            "conflict",
            before=HoldingState("start", quantity=3),
            fill=Execution("buy", quantity=2),
            after=HoldingState("end", quantity=4),
        )


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


if TYPE_CHECKING:

    def typed_transitions(source: FillPosting, times: tuple[date, ...], values: Series[date]) -> None:
        view: ModelPlan[FillPosting] = source.compile()
        cached: ModelPlan[FillPosting] = FillPosting.plan()
        result: FillPosting = view.evaluate(Known(source.fill.price, 100.0))
        state: HoldingState = result
        execution: Execution = result.fill
        path: ModelPath[FillPosting, date] = view.over(times, Known(source.fill.price, values))
        assert state and execution and path and cached
        _wrong: AccountState = result.after  # type: ignore[assignment]
        _wrong_result: Execution = view.evaluate(Known(source.fill.price, 100.0))  # type: ignore[assignment]
        view.evaluate(Known("fill.price", 100.0))  # type: ignore[arg-type]

    def typed_nested_states(fill: Reinvestment, flow: FollowingCapitalFlow) -> None:
        state: HoldingState = fill
        previous: FollowingFill = fill.before
        earlier: FillPosting = previous.before
        quote: Quote = fill.fill.quote
        account: AccountState = flow
        previous_flow: CapitalFlow = flow.before
        assert state and earlier and quote and account and previous_flow

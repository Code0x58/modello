"""Account replay, cash-flow unitization, and P&L reconciliation."""

from datetime import timedelta

import pytest

from examples.compute.named_map import NamedMap
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.basket import EquityBasket, EquityHoldings
from examples.portfolio.demo import asdict_snapshot
from examples.portfolio.explicit import Basket, EquityMembers
from examples.portfolio.models import EquityPosition, Quote, USDCash
from examples.portfolio.postings import AccountFill, CapitalFlow, IncomePosting
from examples.portfolio.signals import TargetWeights
from examples.portfolio.sources import (
    CashFlow,
    Fill,
    Income,
    TransactionLog,
    timestamp,
)
from examples.portfolio.valuation import NamedPortfolio

from ._data import AGE, DECISION, TIMES


def test_account_balances_pnl_and_time_weighted_returns(scenario):
    path = scenario.account
    assert path.nav.values == pytest.approx((2500, 2510, 3041.4, 3066.9))
    assert path.cash.values == pytest.approx((1000, 1000, 1239.4, 1247.9))
    assert path.positions["aapl"].quantity.values == (10, 10, 17, 17)
    assert path.positions["msft"].quantity.values == (10, 10, 0, 0)
    pnl = path.pnl
    assert pnl.holding["aapl"].values == pytest.approx((20, 40, 17))
    assert pnl.holding["msft"].values == pytest.approx((-10, -20, 0))
    assert pnl.trading["aapl"].values == pytest.approx((0, 6.4, 0))
    assert pnl.trading["msft"].values == pytest.approx((0, 7, 0))
    assert pnl.income.values == (0, 0, 8.5)
    assert pnl.fees.values == (0, 2, 0)
    assert pnl.external_flows.values == (0, 500, 0)
    assert pnl.total.values == pytest.approx((10, 31.4, 25.5))
    assert path.reconciliation.values == pytest.approx((0, 0, 0), abs=1e-10)
    # Independent geometric linking before and after the external deposit.
    expected_return = (2532.9 / 2500) * (3066.9 / 3032.9) - 1
    assert path.analytics(periods_per_year=252).performance.total_return == pytest.approx(expected_return)
    assert path.unit_nav.values[-1] == pytest.approx((1 + expected_return) * 100)
    assert expected_return != pytest.approx(3066.9 / 2500 - 1)
    final = path.snapshots[-1]
    assert sum(final.weights.values()) + final.cash / final.nav == pytest.approx(1)


def test_cash_flow_valuation_does_not_depend_on_the_output_grid(engine, ledger):
    baseline = engine.evaluate(ledger, TIMES)
    more_times = tuple(sorted((*TIMES, DECISION, timestamp("2026-01-07T12:00:00Z"), timestamp("2026-01-07T09:06:00Z"))))
    refined = engine.evaluate(ledger, more_times)
    assert asdict_snapshot(refined.snapshots[-1]) == asdict_snapshot(baseline.snapshots[-1])
    assert sum(refined.pnl.total.values) == pytest.approx(sum(baseline.pnl.total.values))
    assert refined.reconciliation.values == pytest.approx((0,) * (len(more_times) - 1), abs=1e-10)
    assert tuple(step.after.cash for step in refined.transitions) == tuple(
        step.after.cash for step in baseline.transitions
    )
    assert tuple(step.after.units for step in refined.transitions) == tuple(
        step.after.units for step in baseline.transitions
    )


def test_replay_retains_typed_event_states_and_their_symbolic_models(scenario):
    path = scenario.account
    assert tuple(step.event for step in path.transitions) == path.ledger.events
    assert tuple(type(step.posting) for step in path.transitions) == (
        AccountFill,
        AccountFill,
        AccountFill,
        CapitalFlow,
        IncomePosting,
    )
    for step in path.transitions:
        assert type(step.before.portfolio) is type(step.after.portfolio) is type(path.snapshots[0].portfolio)
        assert step.before.at == step.after.at == step.event.at
        assert step.before.known_at == step.after.known_at == TIMES[-1]
        assert step.before.prices == step.after.prices
        assert float(step.posting.before.cash) == step.before.cash
        assert float(step.posting.before.nav) == step.before.nav
        assert float(step.posting.before.units) == step.before.units
        assert float(step.posting.after.cash) == step.after.cash
        assert float(step.posting.after.nav) == pytest.approx(step.after.nav)
        assert float(step.posting.after.units) == step.after.units
        if isinstance(step.posting, AccountFill):
            assert isinstance(step.event, Fill)
            name = step.event.member
            assert float(step.posting.settlement.before.quantity) == step.before.quantities[name]
            assert float(step.posting.settlement.quantity) == step.after.quantities[name]
            assert float(step.posting.mark.price) == step.before.prices[name]
    for previous, following in zip(path.transitions, path.transitions[1:]):
        assert previous.after.quantities == following.before.quantities
        assert previous.after.cash == following.before.cash
        assert previous.after.units == following.before.units
    # Subsequent events and output marks leave the retained states unchanged.
    first = path.transitions[0]
    assert float(first.before.portfolio.basket.members.aapl.quantity) == 10
    assert float(first.after.portfolio.basket.members.aapl.quantity) == 14
    assert first.before.cash == 1000
    assert first.after.cash == 579.5


def test_replaying_postings_reuses_compilation_without_symbolic_solves(engine, ledger, monkeypatch):
    baseline = engine.evaluate(ledger, TIMES)
    monkeypatch.setattr("modello.solve", lambda *args, **kwargs: pytest.fail("Repeated symbolic solve"))
    assert tuple(map(asdict_snapshot, engine.evaluate(ledger, TIMES).snapshots)) == tuple(
        map(asdict_snapshot, baseline.snapshots)
    )


def test_empty_equity_account_flows_are_not_investment_returns():
    class Empty(EquityHoldings):
        pass

    opening = NamedPortfolio.define(EquityBasket.define(Empty))("cash", cash=USDCash("cash", amount=100))
    engine = AccountEngine(Account("cash", TIMES[0], opening), NamedMap({}), max_price_age=AGE)
    ledger = TransactionLog(
        (
            CashFlow("deposit", "cash", TIMES[1], TIMES[1], 200),
            CashFlow("withdraw", "cash", TIMES[2], TIMES[2], -150),
        )
    )
    path = engine.evaluate(ledger, TIMES)
    assert path.nav.values == (100, 300, 150, 150)
    assert path.unit_nav.values == (100,) * 4
    assert path.analytics(periods_per_year=252).performance.total_return == 0
    assert path.positions == {}


def test_restatements_are_explicit_and_cannot_be_used_for_past_decisions(engine, ledger):
    before = timestamp("2026-01-07T09:08:00Z")
    late = Fill(
        "late", "research", timestamp("2026-01-07T09:06:00Z"), timestamp("2026-01-07T10:00:00Z"), "aapl", 1, 105
    )
    log = TransactionLog((*ledger.events, late))
    live = engine.evaluate(log, (TIMES[0], before)).snapshots[-1]
    restated = engine.evaluate(log, (TIMES[0], before, TIMES[-1])).snapshots[1]
    assert live.quantities["aapl"] == 14
    assert restated.quantities["aapl"] == 15
    with pytest.raises(ValueError, match="knowledge at the decision"):
        TargetWeights(before, NamedMap({"aapl": 0.5, "msft": 0.0})).orders(restated, batch="past", arrival_at=before)


def test_other_accounts_and_opening_history_are_excluded(engine, ledger):
    log = TransactionLog(
        (
            *ledger.events,
            CashFlow("other", "someone-else", TIMES[1], TIMES[1], 1e6),
            CashFlow("opening", "research", TIMES[0], TIMES[0], 1e6),
        )
    )
    assert tuple(map(asdict_snapshot, engine.evaluate(log, TIMES).snapshots)) == tuple(
        map(asdict_snapshot, engine.evaluate(ledger, TIMES).snapshots)
    )


def test_simultaneous_postings_preserve_source_order(engine):
    at = timestamp("2026-01-07T09:06:00Z")
    flow = CashFlow("deposit", "research", at, at, 2000)
    fill = Fill("buy", "research", at, at, "aapl", 20, 105)
    path = engine.evaluate(TransactionLog((flow, fill)), TIMES)
    assert path.snapshots[-1].quantities["aapl"] == 30
    assert path.transitions[1].before is path.transitions[0].after
    with pytest.raises(ValueError, match="funded cash"):
        engine.evaluate(TransactionLog((fill, flow)), TIMES)


def test_income_before_capital_changes_the_price_at_which_units_are_issued(engine):
    at = TIMES[1]
    income = Income("income", "research", at, at, 100)
    capital = CashFlow("deposit", "research", at, at, 500)
    income_first = engine.evaluate(TransactionLog((income, capital)), TIMES[:2])
    capital_first = engine.evaluate(TransactionLog((capital, income)), TIMES[:2])
    assert income_first.snapshots[-1].nav == capital_first.snapshots[-1].nav == 3110
    assert income_first.snapshots[-1].cash == capital_first.snapshots[-1].cash == 1600
    assert income_first.snapshots[-1].units == pytest.approx(25 + 500 / (2610 / 25))
    assert capital_first.snapshots[-1].units == pytest.approx(25 + 500 / (2510 / 25))
    assert income_first.transitions[1].before is income_first.transitions[0].after


@pytest.mark.parametrize("kind", [Fill, Income])
def test_each_event_requires_a_causal_market_mark_under_the_price_age_policy(engine, market, kind):
    strict = AccountEngine(engine.account, market, max_price_age=timedelta(hours=1))
    # Both reporting marks have fresh data, but the event's previous quote is stale.
    at = timestamp("2026-01-06T12:00:00Z")
    event = (
        Fill("buy", "research", at, at, "aapl", 1, 100) if kind is Fill else Income("income", "research", at, at, 10)
    )
    with pytest.raises(ValueError, match="stale"):
        strict.evaluate(TransactionLog((event,)), TIMES[:2])


@pytest.mark.parametrize("quantity", [100, -11])
def test_unfunded_or_short_positions_are_rejected(engine, quantity):
    at = timestamp("2026-01-07T09:06:00Z")
    with pytest.raises(ValueError, match="funded cash"):
        engine.evaluate(TransactionLog((Fill("bad", "research", at, at, "aapl", quantity, 105),)), TIMES)


def test_unknown_members_and_unfunded_withdrawals(engine):
    at = TIMES[1]
    for event in (Fill("bad", "research", at, at, "unknown", 1, 1), Income("bad", "research", at, at, 1, "unknown")):
        with pytest.raises(ValueError, match="Unknown"):
            engine.evaluate(TransactionLog((event,)), TIMES)
    for event in (CashFlow("bad", "research", at, at, -2000), Income("bad", "research", at, at, -2000)):
        with pytest.raises(ValueError, match="funded cash"):
            engine.evaluate(TransactionLog((event,)), TIMES)


def test_price_identity_staleness_and_mark_grid_are_checked(engine, market):
    with pytest.raises(ValueError, match="Price instrument must"):
        AccountEngine(engine.account, NamedMap({"aapl": market["msft"], "msft": market["aapl"]}), max_price_age=AGE)
    with pytest.raises(ValueError, match="stale"):
        engine.evaluate(TransactionLog(()), (TIMES[0], timestamp("2026-02-01T16:01:00Z")))
    with pytest.raises(ValueError, match="increase from the account opening"):
        engine.evaluate(TransactionLog(()), (TIMES[1], TIMES[2]))


def test_account_does_not_drop_opening_valuation_constraints(market):
    opening = NamedPortfolio.define(Basket)(
        "fixed quotes",
        basket=Basket(
            "basket",
            members=EquityMembers(
                "equities",
                aapl=EquityPosition("apple", quantity=10, instrument=Quote("quote", price=100)),
                msft=EquityPosition("microsoft", quantity=10, instrument=Quote("quote", price=50)),
            ),
        ),
        cash=USDCash("cash", amount=1000),
    )
    with pytest.raises(ValueError, match="opening prices as independent"):
        AccountEngine(Account("research", TIMES[0], opening), market, max_price_age=AGE)

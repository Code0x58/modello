"""Account replay, cash-flow unitization, and P&L reconciliation."""

import pytest

from examples.compute.named_map import NamedMap
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.basket import EquityBasket, EquityHoldings
from examples.portfolio.explicit import Basket, EquityMembers
from examples.portfolio.models import EquityPosition, Quote, USDCash
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
    assert refined.snapshots[-1] == baseline.snapshots[-1]
    assert sum(refined.pnl.total.values) == pytest.approx(sum(baseline.pnl.total.values))
    assert refined.reconciliation.values == pytest.approx((0,) * (len(more_times) - 1), abs=1e-10)


def test_replaying_postings_reuses_compilation_without_symbolic_solves(engine, ledger, monkeypatch):
    baseline = engine.evaluate(ledger, TIMES)
    monkeypatch.setattr("modello.solve", lambda *args, **kwargs: pytest.fail("Repeated symbolic solve"))
    assert engine.evaluate(ledger, TIMES).snapshots == baseline.snapshots


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
    assert engine.evaluate(log, TIMES).snapshots == engine.evaluate(ledger, TIMES).snapshots


def test_simultaneous_postings_preserve_source_order(engine):
    at = timestamp("2026-01-07T09:06:00Z")
    flow = CashFlow("deposit", "research", at, at, 2000)
    fill = Fill("buy", "research", at, at, "aapl", 20, 105)
    path = engine.evaluate(TransactionLog((flow, fill)), TIMES)
    assert path.snapshots[-1].quantities["aapl"] == 30
    with pytest.raises(ValueError, match="funded cash"):
        engine.evaluate(TransactionLog((fill, flow)), TIMES)


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
    with pytest.raises(ValueError, match="funded cash"):
        engine.evaluate(TransactionLog((CashFlow("bad", "research", at, at, -2000),)), TIMES)


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

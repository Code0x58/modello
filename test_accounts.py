"""Account economics, causal inputs, and independent execution-cost references."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from examples.account import Account, AccountEngine
from examples.account_demo import example_run
from examples.account_sources import (
    CashFlow,
    CSVMarketSource,
    CSVTransactionSource,
    Fill,
    Income,
    PriceHistory,
    TransactionLog,
    timestamp,
)
from examples.execution import benchmark_executions
from examples.named_members import NamedMap
from examples.named_portfolio import EquityMembers, portfolio_model
from examples.portfolio import AAPL, MSFT, ApplePosition, MicrosoftPosition, USDCash
from examples.series import Series
from examples.signals import MomentumSignals, TargetWeights, positive_signal_targets, rebalance_orders
from modello import Modello

DATA = Path(__file__).parent / "examples" / "data" / "portfolio"
AGE = timedelta(days=4)
TIMES = tuple(timestamp(f"2026-01-{day:02d}T16:01:00Z") for day in range(5, 9))
DECISION = timestamp("2026-01-07T09:00:00Z")
ARRIVAL = timestamp("2026-01-07T09:05:00Z")
ASSESSMENT = timestamp("2026-01-07T09:15:00Z")


@pytest.fixture(scope="module")
def market():
    return CSVMarketSource(DATA / "market.csv", NamedMap({"aapl": AAPL, "msft": MSFT})).load()


@pytest.fixture(scope="module")
def ledger():
    return CSVTransactionSource(DATA / "transactions.csv").load()


@pytest.fixture(scope="module")
def engine(market):
    opening = portfolio_model(EquityMembers)(
        "opening",
        equities=EquityMembers(
            "equities", aapl=ApplePosition("apple", quantity=10), msft=MicrosoftPosition("microsoft", quantity=10)
        ),
        cash=USDCash("cash", amount=1000),
    )
    return AccountEngine(Account("research", TIMES[0], opening), market, max_price_age=AGE)


@pytest.fixture(scope="module")
def scenario():
    return example_run()


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


def test_empty_equity_account_flows_are_not_investment_returns():
    class Empty(Modello):
        pass

    opening = portfolio_model(Empty)("cash", cash=USDCash("cash", amount=100))
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


def test_future_prices_and_postings_cannot_change_decisions(engine, market, ledger, scenario):
    changed = market.map(
        lambda history: replace(
            history,
            prices=Series(
                history.prices.times,
                tuple(
                    value if at <= DECISION else value * 10
                    for at, value in zip(history.prices.times, history.prices.values, strict=True)
                ),
            ),
        )
    )
    other_engine = AccountEngine(engine.account, changed, max_price_age=AGE)
    snapshot = other_engine.evaluate(ledger, (TIMES[0], DECISION)).snapshots[-1]
    feature_prices = changed.map(lambda history: history.sample((*TIMES[:2], DECISION), max_age=AGE))
    signals = MomentumSignals(feature_prices, lookback=2).load()
    targets = positive_signal_targets(signals, DECISION)
    assert targets == scenario.targets
    assert rebalance_orders(snapshot, targets, batch="rebalance-1", arrival_at=ARRIVAL) == scenario.orders
    assert snapshot.quantities == {"aapl": 10, "msft": 10}
    assert snapshot.cash == 1000


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
        rebalance_orders(
            restated, TargetWeights(before, NamedMap({"aapl": 0.5, "msft": 0.0})), batch="past", arrival_at=before
        )


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
    with pytest.raises(ValueError, match="match the declared"):
        AccountEngine(engine.account, NamedMap({"aapl": market["msft"], "msft": market["aapl"]}), max_price_age=AGE)
    with pytest.raises(ValueError, match="stale"):
        engine.evaluate(TransactionLog(()), (TIMES[0], timestamp("2026-02-01T16:01:00Z")))
    with pytest.raises(ValueError, match="increase from the account opening"):
        engine.evaluate(TransactionLog(()), (TIMES[1], TIMES[2]))


def test_account_does_not_drop_opening_valuation_constraints(market):
    opening = portfolio_model(EquityMembers)(
        "fixed quotes",
        equities=EquityMembers(
            "equities",
            aapl=ApplePosition("apple", quantity=10, instrument=AAPL("quote", price=100)),
            msft=MicrosoftPosition("microsoft", quantity=10, instrument=MSFT("quote", price=50)),
        ),
        cash=USDCash("cash", amount=1000),
    )
    with pytest.raises(ValueError, match="opening prices as independent"):
        AccountEngine(Account("research", TIMES[0], opening), market, max_price_age=AGE)


def test_publication_time_prevents_lookahead_and_age_uses_observation_time():
    observed = (timestamp("2026-01-07T09:00:00Z"), timestamp("2026-01-07T09:05:00Z"))
    available = (observed[0], timestamp("2026-01-07T09:10:00Z"))
    history = PriceHistory(AAPL, Series(observed, (100, 105)), available, (1, 1))
    assert history.mark(timestamp("2026-01-07T09:07:00Z"), max_age=AGE) == 100
    assert history.mark(available[1], max_age=AGE) == 105
    signal = MomentumSignals(NamedMap({"aapl": history.known_prices}), lookback=1).load()["aapl"]
    with pytest.raises(ValueError, match="No observation"):
        signal.asof(timestamp("2026-01-07T09:07:00Z"))
    assert signal.asof(available[1]) == pytest.approx(0.05)
    with pytest.raises(ValueError, match="stale"):
        history.mark(available[1], max_age=timedelta(minutes=2))


def test_transaction_series_aggregate_simultaneous_executions_without_losing_ids():
    at = TIMES[1]
    log = TransactionLog(
        (Fill("one", "research", at, at, "aapl", 2, 100, 0.1), Fill("two", "research", at, at, "aapl", -3, 102, 0.2))
    )
    series = log.executions("aapl")
    assert len(log.events) == 2
    assert series.quantity.values == (-1,)
    assert series.volume.values == (5,)
    assert series.vwap.values == (101.2,)
    assert series.fees.values == pytest.approx((0.3,))
    assert log.executions("msft").quantity.values == ()
    with pytest.raises(ValueError, match="Duplicate transaction ID"):
        TransactionLog((log.events[0], log.events[0]))


def test_feed_schema_identity_and_invalid_events_are_rejected(tmp_path, market):
    bad = tmp_path / "bad.csv"
    bad.write_text("member,price\naapl,100\n")
    with pytest.raises(ValueError, match="unique columns"):
        CSVMarketSource(bad, NamedMap({"aapl": AAPL})).load()
    bad.write_text((DATA / "market.csv").read_text().replace(",AAPL,", ",MSFT,"))
    with pytest.raises(ValueError, match="ticker for aapl"):
        CSVMarketSource(bad, NamedMap({"aapl": AAPL, "msft": MSFT})).load()
    with pytest.raises(ValueError, match="available|publication"):
        replace(market["aapl"], available_at=market["aapl"].available_at[::-1])
    with pytest.raises(ValueError, match="strictly positive"):
        replace(market["aapl"], volumes=(0,) * len(market["aapl"].volumes))
    with pytest.raises(ValueError, match="finite"):
        Fill("bad", "research", TIMES[1], TIMES[1], "aapl", 1, float("nan"))
    with pytest.raises(ValueError, match="reporting time"):
        CashFlow("bad", "research", TIMES[1], TIMES[0], 100)


def test_signal_targets_and_rebalance_quantities(scenario):
    assert scenario.targets.weights == {"aapl": 0.8, "msft": 0}
    apple = scenario.orders["rebalance-1:aapl"]
    microsoft = scenario.orders["rebalance-1:msft"]
    assert apple.quantity == pytest.approx(0.8 * 2520 / 104 - 10)
    assert microsoft.quantity == -10
    assert apple.model_price == 104
    assert apple.equity is AAPL and microsoft.equity is MSFT
    for invalid in ({"aapl": -0.1}, {"aapl": 1.1}, {"aapl": float("nan")}):
        with pytest.raises(ValueError):
            TargetWeights(DECISION, NamedMap(invalid))


def test_execution_cost_decomposition_and_partial_fill_opportunity(scenario):
    apple = scenario.execution["rebalance-1:aapl"]
    microsoft = scenario.execution["rebalance-1:msft"]
    assert apple.filled_quantity == 7
    assert apple.remaining_quantity == pytest.approx(0.8 * 2520 / 104 - 17)
    assert apple.execution_vwap == pytest.approx((4 * 105 + 3 * 105.2) / 7)
    assert apple.market_vwap == pytest.approx((500 * 104.5 + 100 * 104.7 + 200 * 104.8) / 800)
    assert apple.model_slippage_cost == pytest.approx(8.6)
    assert apple.delay_cost == pytest.approx(3.5)
    assert apple.trading_cost == pytest.approx(4.1)
    assert apple.opportunity_cost == pytest.approx(apple.remaining_quantity * (104.8 - 104))
    direct = 4 * 105 + 3 * 105.2 + apple.remaining_quantity * 104.8 - abs(apple.order.quantity) * 104 + 1
    assert apple.implementation_shortfall == pytest.approx(direct)
    assert microsoft.model_slippage_cost == pytest.approx(10 * (48 - 47.7) + 1)
    assert microsoft.implementation_shortfall == pytest.approx(4)
    assert microsoft.opportunity_cost == 0
    assert apple.arrival_slippage_bps > 0 and microsoft.arrival_slippage_bps > 0


def test_execution_improvement_has_a_negative_cost_for_both_sides(scenario, market):
    for name, fill_price in (("aapl", 103), ("msft", 49)):
        order = replace(scenario.orders[f"rebalance-1:{name}"], quantity=2 if name == "aapl" else -2)
        fill = Fill(
            "better",
            "research",
            timestamp("2026-01-07T09:06:00Z"),
            timestamp("2026-01-07T09:06:00Z"),
            name,
            order.quantity,
            fill_price,
            order_id=order.id,
        )
        result = benchmark_executions(
            NamedMap({order.id: order}), TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE
        )[order.id]
        assert result.model_slippage_cost == pytest.approx(-2)
        assert result.implementation_shortfall == pytest.approx(-2)


def test_unfilled_orders_and_late_publication_are_explicit(scenario, market):
    order = scenario.orders["rebalance-1:aapl"]
    fill = Fill(
        "late",
        "research",
        timestamp("2026-01-07T09:06:00Z"),
        timestamp("2026-01-07T09:20:00Z"),
        "aapl",
        2,
        105,
        order_id=order.id,
    )
    report = benchmark_executions(
        NamedMap({order.id: order}), TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE
    )[order.id]
    assert report.filled_quantity == 0
    assert report.execution_vwap is None and report.model_slippage_bps is None
    assert report.model_slippage_cost == 0
    assert report.implementation_shortfall == pytest.approx(abs(order.quantity) * (104.8 - 104))


@pytest.mark.parametrize(
    "change", [{"quantity": -1}, {"quantity": 100}, {"member": "msft"}, {"at": timestamp("2026-01-07T09:04:00Z")}]
)
def test_order_fill_linkage_is_validated(scenario, market, change):
    order = scenario.orders["rebalance-1:aapl"]
    fill = Fill(
        "bad",
        "research",
        timestamp("2026-01-07T09:06:00Z"),
        timestamp("2026-01-07T09:06:00Z"),
        "aapl",
        1,
        105,
        order_id=order.id,
    )
    with pytest.raises(ValueError):
        benchmark_executions(
            NamedMap({order.id: order}),
            TransactionLog((replace(fill, **change),)),
            market,
            at=ASSESSMENT,
            max_price_age=AGE,
        )


def test_market_benchmarks_are_independent_of_the_accounts_fill_prices(scenario, market):
    order = scenario.orders["rebalance-1:aapl"]
    fill = Fill("expensive", "research", ARRIVAL, ARRIVAL, "aapl", 1, 999, order_id=order.id)
    result = benchmark_executions(
        NamedMap({order.id: order}), TransactionLog((fill,)), market, at=ASSESSMENT, max_price_age=AGE
    )[order.id]
    assert result.market_vwap == scenario.execution[order.id].market_vwap
    assert result.execution_vwap == 999
    wrong = NamedMap({"aapl": replace(market["aapl"], equity=MSFT), "msft": market["msft"]})
    with pytest.raises(ValueError, match="instrument must match"):
        benchmark_executions(NamedMap({order.id: order}), TransactionLog(()), wrong, at=ASSESSMENT, max_price_age=AGE)

"""Signal attribution, accounting reconciliation, and wealth preferences."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import pytest
from sympy import Rational, Symbol, exp, log

from examples.data import NamedMap, Series
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.models import MSFT, USDCash
from examples.portfolio.reporting import AccountReport, CashUtility, OrderAttribution
from examples.portfolio.sources import CashFlow, Fill, Income, TransactionLog, timestamp
from examples.portfolio.valuation import EquityBasket, EquityHoldings, NamedPortfolio


def attribute(scenario, *, path=None, weights=None, signals=None, orders=None):
    return AccountReport.from_account(
        scenario.account if path is None else path,
        (scenario.orders if orders is None else orders).map(
            lambda order: OrderAttribution(order, NamedMap({"momentum": 1.0}) if weights is None else weights)
        ),
        signals=NamedMap({"momentum": scenario.signals}) if signals is None else signals,
        utilities=NamedMap({}),
    )


def test_attribution_reconciles_account_and_each_instrument(reporting):
    assert reporting.account["net"].values == pytest.approx((10, 31.4, 25.5))
    assert reporting.account["external_flows"].values == (0, 500, 0)
    assert reporting.account["residual"].values == pytest.approx((0, 0, 0), abs=1e-9)
    assert reporting.sources["opening"]["net"].values == pytest.approx((10, 20, 25))
    assert reporting.sources["signal:momentum"]["net"].values == pytest.approx((0, 11.4, 0.5))
    assert reporting.sources["unattributed"]["net"].values == (0, 0, 0)
    apple, microsoft = reporting.instruments["aapl"], reporting.instruments["msft"]
    assert sum(apple.profit["net"].values) == pytest.approx(90.9)
    assert sum(microsoft.profit["net"].values) == pytest.approx(-24)
    assert sum(apple.sources["signal:momentum"]["net"].values) == pytest.approx(15.9)
    assert sum(microsoft.sources["signal:momentum"]["net"].values) == pytest.approx(-4)
    # Dividends follow actual event-time sleeve shares: 10 opening and 7 signal shares.
    assert apple.sources["opening"]["income"].values == (0, 0, 5)
    assert apple.sources["signal:momentum"]["income"].values == (0, 0, 3.5)
    assert microsoft.quantities["signal:momentum"].values == (0, 0, -10, -10)
    for instrument in reporting.instruments.values():
        for i, quantity in enumerate(instrument.position.quantity.values):
            assert sum(series.values[i] for series in instrument.quantities.values()) == pytest.approx(quantity)


def test_multiple_signals_allocate_executed_positions_fees_and_income(scenario):
    result = attribute(
        scenario,
        weights=NamedMap({"fast": 0.25, "slow": 0.75}),
        signals=NamedMap({"fast": scenario.signals, "slow": scenario.signals}),
    )
    assert sum(result.sources["signal:fast"]["net"].values) == pytest.approx(11.9 * 0.25)
    assert sum(result.sources["signal:slow"]["net"].values) == pytest.approx(11.9 * 0.75)
    assert sum(result.sources["signal:fast"]["fees"].values) == pytest.approx(0.5)
    assert result.account["residual"].values == pytest.approx((0, 0, 0), abs=1e-9)


def test_attribution_is_independent_of_the_reporting_grid(scenario, engine, reporting):
    times = tuple(sorted((*scenario.account.nav.times, timestamp("2026-01-07T09:08:00Z"))))
    refined = attribute(scenario, path=engine.evaluate(scenario.account.ledger, times))
    for name in reporting.sources:
        assert sum(refined.sources[name]["net"].values) == pytest.approx(sum(reporting.sources[name]["net"].values))
    assert refined.account["residual"].values == pytest.approx((0,) * 4, abs=1e-9)


def test_unlinked_trades_remain_visible_instead_of_being_assigned_to_a_signal(scenario):
    ledger = TransactionLog(
        tuple(
            replace(event, order_id=None) if isinstance(event, Fill) and event.member == "aapl" else event
            for event in scenario.account.ledger.events
        )
    )
    result = attribute(scenario, path=replace(scenario.account, ledger=ledger))
    assert sum(result.sources["unattributed"]["net"].values) == pytest.approx(15.9)
    assert sum(result.sources["signal:momentum"]["net"].values) == pytest.approx(-4)


def test_income_after_a_position_is_closed_is_unattributed_to_sleeves(scenario, engine):
    at = timestamp("2026-01-08T12:00:00Z")
    ledger = TransactionLog((*scenario.account.ledger.events, Income("late-dividend", "research", at, at, 5, "msft")))
    result = attribute(scenario, path=engine.evaluate(ledger, scenario.account.nav.times))
    assert result.instruments["msft"].sources["unattributed"]["income"].values == (0, 0, 5)
    assert sum(result.account["net"].values) == pytest.approx(71.9)


def test_cash_only_reports_separate_income_and_external_capital():
    class Empty(EquityHoldings):
        pass

    start = timestamp("2026-01-05T16:01:00Z")
    times = tuple(start + timedelta(days=i) for i in range(3))
    account = Account(
        "cash",
        start,
        NamedPortfolio.define(EquityBasket.define(Empty))("opening", cash=USDCash("cash", amount=100)),
    )
    engine = AccountEngine(account, NamedMap({}), max_price_age=timedelta(days=1))
    ledger = TransactionLog(
        (
            CashFlow("deposit", "cash", times[1], times[1], 500),
            CashFlow("withdrawal", "cash", times[2], times[2], -100),
            Income("interest", "cash", times[2], times[2], 10),
            Income("expense", "cash", times[2], times[2], -2),
        )
    )
    result = AccountReport.from_account(
        engine.evaluate(ledger, times), (), signals=NamedMap({}), utilities=NamedMap({})
    )
    assert not result.instruments
    assert result.path.cash.values == (100, 600, 508)
    assert result.path.unit_nav.values == pytest.approx((100, 100, 101.6))
    assert result.account["net"].values == (0, 8)
    assert result.account["external_flows"].values == (500, -100)
    assert result.sources["unattributed"]["income"].values == (0, 8)


@pytest.mark.parametrize("weights", [{}, {"a": 0.5}, {"a": -1, "b": 2}, {"a": float("nan")}])
def test_invalid_attribution_weights(scenario, weights):
    with pytest.raises(ValueError, match="sum to one"):
        scenario.orders.for_member(scenario.decision.portfolio.basket.members.aapl).attribute(NamedMap(weights))


def test_attribution_requires_correct_order_identity_side_and_available_signals(scenario):
    for changes in ({"account": "other"}, {"equity": MSFT}, {"quantity": -1}, {"quantity": 1}):
        orders = scenario.orders.map(lambda order: replace(order, **changes) if order.member == "aapl" else order)
        with pytest.raises(ValueError):
            attribute(scenario, orders=orders)
    with pytest.raises(ValueError, match="named signal history"):
        attribute(scenario, weights=NamedMap({"unknown": 1.0}))
    future = scenario.signals.map(
        lambda series: Series(tuple(at + timedelta(seconds=1) for at in series.times), series.values)
    )
    with pytest.raises(ValueError, match="No observation"):
        attribute(scenario, signals=NamedMap({"momentum": future}))


def test_different_signals_can_drive_different_instruments(scenario):
    result = AccountReport.from_account(
        scenario.account,
        scenario.orders.map(
            lambda order: OrderAttribution(order, NamedMap({"fast" if order.member == "aapl" else "slow": 1.0}))
        ),
        signals=NamedMap({"fast": scenario.signals, "slow": scenario.signals}),
        utilities=NamedMap({}),
    )
    assert sum(result.sources["signal:fast"]["net"].values) == pytest.approx(15.9)
    assert sum(result.sources["signal:slow"]["net"].values) == pytest.approx(-4)


def test_reporting_requires_the_account_ledger_known_by_the_horizon(scenario):
    first, *rest = scenario.account.ledger.events
    for event in (
        replace(first, account="other"),
        replace(first, recorded_at=scenario.account.nav.times[-1] + timedelta(days=1)),
    ):
        with pytest.raises(ValueError, match="in-window ledger"):
            attribute(scenario, path=replace(scenario.account, ledger=TransactionLog((event, *rest))))


def test_reporting_derives_ledger_ids_and_supports_multiple_orders_per_member(scenario):
    first, second = scenario.attributions
    another = replace(first.order, id="another-decision").attribute(first.weights)
    report = AccountReport.from_account(
        scenario.account,
        (first, second, another),
        signals=NamedMap({"momentum": scenario.signals}),
        utilities=NamedMap({}),
    )
    assert tuple(record.order for record in report.attributions) == (first.order, second.order, another.order)
    assert sum(report.account["net"].values) == pytest.approx(66.9)
    with pytest.raises(ValueError, match="Duplicate member name"):
        AccountReport.from_account(scenario.account, (first, first), signals=NamedMap({}), utilities=NamedMap({}))


def test_utility_derivatives_and_affine_invariance():
    exponential = CashUtility.cara(0.001)
    wealth = exponential.wealth
    assert exponential.expressions["marginal_utility"] == exp(-wealth / 1000)
    assert exponential.expressions["absolute_risk_aversion"] == Rational(1, 1000)
    power = CashUtility.crra(2)
    assert power.expressions["absolute_risk_aversion"] == 2 / power.wealth
    assert power.expressions["relative_risk_aversion"] == 2
    logarithmic = CashUtility.crra(1)
    assert logarithmic.expression == log(logarithmic.wealth)
    transformed = CashUtility("Scaled log", logarithmic.wealth, 5 * logarithmic.expression + 7)
    for field in ("absolute_risk_aversion", "relative_risk_aversion"):
        assert transformed.expressions[field] == logarithmic.expressions[field]
    for utility in (CashUtility.cara(0), CashUtility.crra(0)):
        assert utility.expressions["absolute_risk_aversion"] == 0
        assert utility.expressions["relative_risk_aversion"] == 0


def test_utility_reports_wealth_preferences_including_the_deposit(reporting):
    assert reporting.utilities["cara"].values["absolute_risk_aversion"].values == (0.001,) * 4
    assert reporting.utilities["crra"].values["relative_risk_aversion"].values == (2,) * 4
    assert reporting.utilities["crra"].values["absolute_risk_aversion"].values == pytest.approx(
        tuple(2 / value for value in reporting.path.nav.values)
    )


@pytest.mark.parametrize("coefficient", [-1, float("inf"), float("nan")])
def test_invalid_risk_aversion_coefficients(coefficient):
    for constructor in (CashUtility.cara, CashUtility.crra):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            constructor(coefficient)


def test_utility_domains_and_undefined_ratios():
    times = (date(2025, 1, 1), date(2026, 1, 1))
    with pytest.raises(ValueError, match="positive wealth"):
        CashUtility.crra(2).over(Series(times, (0, 100)))
    w = Symbol("wealth", positive=True)
    with pytest.raises(ValueError, match="only free symbol"):
        CashUtility("Unbound", w, w * Symbol("alpha"))
    with pytest.raises(ValueError, match="vary with wealth"):
        CashUtility("Constant", w, Rational(1))
    with pytest.raises(ValueError, match="increasing"):
        CashUtility("Decreasing", w, -w).over(Series(times, (100, 200)))
    with pytest.raises(ValueError, match="finite"):
        CashUtility("Singular", w, log(w - 100)).over(Series(times, (100, 200)))

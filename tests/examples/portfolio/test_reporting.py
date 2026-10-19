"""Symbolic P&L relationships and signal-sleeve attribution."""

from dataclasses import replace
from datetime import timedelta

import pytest
from sympy import Rational, diff, simplify

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.account import Account, AccountEngine
from examples.portfolio.basket import basket_model
from examples.portfolio.models import MSFT, USDCash
from examples.portfolio.named import portfolio_model
from examples.portfolio.reporting import AccountPnL, InstrumentPnL, OrderAttribution, report
from examples.portfolio.sources import CashFlow, Fill, Income, TransactionLog, timestamp
from modello import Modello


def attribute(scenario, *, path=None, weights=None, signals=None, orders=None):
    return report(
        scenario.account if path is None else path,
        (scenario.orders if orders is None else orders).map(
            lambda order: OrderAttribution(order, NamedMap({"momentum": 1.0}) if weights is None else weights)
        ),
        signals=NamedMap({"momentum": scenario.signals}) if signals is None else signals,
        utilities=NamedMap({}),
    )


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
    class Empty(Modello):
        pass

    start = timestamp("2026-01-05T16:01:00Z")
    times = tuple(start + timedelta(days=i) for i in range(3))
    account = Account(
        "cash", start, portfolio_model(basket_model(Empty, NamedMap({})))("opening", cash=USDCash("cash", amount=100))
    )
    engine = AccountEngine(account, NamedMap({}), max_price_age=timedelta(days=1))
    ledger = TransactionLog(
        (CashFlow("deposit", "cash", times[1], times[1], 500), Income("interest", "cash", times[2], times[2], 10))
    )
    result = report(engine.evaluate(ledger, times), NamedMap({}), signals=NamedMap({}), utilities=NamedMap({}))
    assert not result.instruments
    assert result.account["net"].values == (0, 10)
    assert result.account["external_flows"].values == (500, 0)
    assert result.sources["unattributed"]["income"].values == (0, 10)


@pytest.mark.parametrize("weights", [{}, {"a": 0.5}, {"a": -1, "b": 2}, {"a": float("nan")}])
def test_invalid_attribution_weights(scenario, weights):
    with pytest.raises(ValueError, match="sum to one"):
        OrderAttribution(scenario.orders["rebalance-1:aapl"], NamedMap(weights))


def test_attribution_requires_correct_order_identity_side_and_available_signals(scenario):
    for changes in ({"account": "other"}, {"equity": MSFT}, {"quantity": -1}, {"quantity": 1}):
        orders = NamedMap(
            (name, replace(order, **changes) if order.member == "aapl" else order)
            for name, order in scenario.orders.items()
        )
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
    result = report(
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

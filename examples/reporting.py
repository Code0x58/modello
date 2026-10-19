"""Symbolic P&L reporting and explicit attribution to signal-linked trading sleeves."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isclose, isfinite

from examples.account import AccountPath, PositionHistory
from examples.account_sources import CashFlow, Fill, Income
from examples.compute.named_map import NamedMap
from examples.compute.scalar_plan import ScalarPlan
from examples.compute.series import Series
from examples.signals import OrderIntent
from examples.utility import CashUtility
from modello import InstanceDummy, Modello


class Profit(Modello):
    """Currency P&L shared by instrument and account reporting models."""

    holding = InstanceDummy("holding", real=True)
    trading = InstanceDummy("trading", real=True)
    income = InstanceDummy("income", real=True)
    fees = InstanceDummy("fees", nonnegative=True)
    gross = holding + trading + income
    net = gross - fees


class InstrumentPnL(Profit):
    """Exact interval mark-to-market P&L, also valid for signed virtual sleeves."""

    opening_quantity = InstanceDummy("opening_quantity", real=True)
    opening_price = InstanceDummy("opening_price", positive=True)
    closing_price = InstanceDummy("closing_price", positive=True)
    fill_quantity = InstanceDummy("fill_quantity", real=True)
    fill_notional = InstanceDummy("fill_notional", real=True)
    holding = opening_quantity * (closing_price - opening_price)
    trading = fill_quantity * closing_price - fill_notional


class AccountPnL(Profit):
    """Relate investment profit and external capital to account NAV changes."""

    opening_nav = InstanceDummy("opening_nav", positive=True)
    closing_nav = InstanceDummy("closing_nav", positive=True)
    external_flows = InstanceDummy("external_flows", real=True)
    nav_change = closing_nav - opening_nav
    residual = nav_change - Profit.net - external_flows


PROFIT_FIELDS = ("holding", "trading", "income", "fees", "gross", "net")


def model_plan(model: Modello, inputs: tuple[str, ...], outputs: tuple[str, ...]) -> ScalarPlan:
    """Compile resolved model expressions, retaining them for inspection and differentiation."""
    return ScalarPlan(
        NamedMap((name, getattr(model, name)) for name in inputs),
        NamedMap((name, getattr(model, name)) for name in outputs),
    )


@dataclass(frozen=True)
class OrderAttribution:
    """A decision-time allocation of one order's executed quantities to named signals."""

    order: OrderIntent
    weights: NamedMap[float]

    def __post_init__(self) -> None:
        if (
            not self.weights
            or any(not isfinite(weight) or weight < 0 for weight in self.weights.values())
            or not isclose(sum(self.weights.values()), 1.0, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("Signal attribution weights must be finite, nonnegative, and sum to one")


@dataclass(frozen=True)
class InstrumentReport:
    """Actual positions and a P&L decomposition over opening, signal, and unattributed sleeves."""

    position: PositionHistory
    profit: NamedMap[Series[datetime]]
    sources: NamedMap[NamedMap[Series[datetime]]]
    quantities: NamedMap[Series[datetime]]


@dataclass(frozen=True)
class UtilityReport:
    definition: CashUtility
    values: NamedMap[Series[datetime]]


@dataclass(frozen=True)
class AccountReport:
    """Typed, inspectable reporting layers; every monetary attribution adds back to the account."""

    path: AccountPath
    account: NamedMap[Series[datetime]]
    instruments: NamedMap[InstrumentReport]
    sources: NamedMap[NamedMap[Series[datetime]]]
    signals: NamedMap[NamedMap[Series[datetime]]]
    attributions: NamedMap[OrderAttribution]
    utilities: NamedMap[UtilityReport]
    instrument_plan: ScalarPlan
    account_plan: ScalarPlan


def sum_columns(
    times: tuple[datetime, ...], paths: tuple[NamedMap[Series[datetime]], ...]
) -> NamedMap[Series[datetime]]:
    """Reduce aligned, named numeric report outputs without reimplementing their formulas."""
    if any(set(path) != set(PROFIT_FIELDS) or any(series.times != times for series in path.values()) for path in paths):
        raise ValueError("Profit reports must share named columns and interval times")
    return NamedMap(
        (name, Series(times, tuple(sum(path[name].values[i] for path in paths) for i in range(len(times)))))
        for name in PROFIT_FIELDS
    )


def report(
    path: AccountPath,
    attributions: NamedMap[OrderAttribution],
    *,
    signals: NamedMap[NamedMap[Series[datetime]]],
    utilities: NamedMap[CashUtility],
) -> AccountReport:
    """Attribute executed decisions relative to the unchanged opening reference book.

    Signal sleeves start empty and retain signed incremental shares from tagged
    fills. Instrument income follows sleeve shares at the income event. Untagged
    fills, unassignable income, and account cash income remain visible as unattributed.
    """
    times = path.nav.times
    axis = times[1:]
    members = tuple(path.positions)
    sources = ("opening", *(f"signal:{name}" for name in signals), "unattributed")
    if any(
        event.account != path.snapshots[0].account
        or not times[0] < event.at <= times[-1]
        or event.recorded_at > times[-1]
        for event in path.ledger.events
    ):
        raise ValueError("Use the account's in-window ledger known by the reporting horizon")
    for order_id, attribution in attributions.items():
        order = attribution.order
        if order_id != order.id or order.account != path.snapshots[0].account or order.member not in members:
            raise ValueError("Attribution must identify an order for this account and instrument")
        if order.equity is not path.snapshots[0].instruments[order.member]:
            raise ValueError("Attribution must match the declared instrument type")
        for signal, weight in attribution.weights.items():
            if signal not in signals or order.member not in signals[signal]:
                raise ValueError("Attribution needs a named signal history for the order's instrument")
            if weight:
                signals[signal][order.member].asof(order.decision_at)

    instrument_plan = model_plan(
        InstrumentPnL("instrument_report"),
        ("opening_quantity", "opening_price", "closing_price", "fill_quantity", "fill_notional", "income", "fees"),
        PROFIT_FIELDS,
    )
    account_plan = model_plan(
        AccountPnL("account_report"),
        ("holding", "trading", "income", "fees", "opening_nav", "closing_nav", "external_flows"),
        (*PROFIT_FIELDS, "nav_change", "external_flows", "residual"),
    )
    # The mutable state is only the ledger-to-vector adapter. Formula evaluation
    # below runs through the symbolic models, compiled once per report.
    quantities = {
        (member, source): path.positions[member].quantity.values[0] if source == "opening" else 0.0
        for member in members
        for source in sources
    }
    quantity_paths = {key: [value] for key, value in quantities.items()}
    columns: dict[tuple[str, str], dict[str, list[float]]] = {
        key: {name: [] for name in instrument_plan.inputs} for key in quantities
    }
    cash_income = [0.0] * len(axis)
    filled = {key: 0.0 for key in attributions}
    cursor = 0
    for i, end in enumerate(axis):
        rows = {
            (member, source): {
                "opening_quantity": quantities[member, source],
                "opening_price": path.positions[member].price.values[i],
                "closing_price": path.positions[member].price.values[i + 1],
                "fill_quantity": 0.0,
                "fill_notional": 0.0,
                "income": 0.0,
                "fees": 0.0,
            }
            for member in members
            for source in sources
        }
        while cursor < len(path.ledger.events) and path.ledger.events[cursor].at <= end:
            event = path.ledger.events[cursor]
            cursor += 1
            if isinstance(event, CashFlow):
                continue
            if isinstance(event, Fill):
                linked = attributions.get(event.order_id or "")
                if linked is not None:
                    order = linked.order
                    if (
                        event.member != order.member
                        or event.at < order.arrival_at
                        or event.quantity * order.quantity <= 0
                    ):
                        raise ValueError("A linked fill must match the order's instrument, time, and side")
                    filled[order.id] += abs(event.quantity)
                    if filled[order.id] > abs(order.quantity) + 1e-10:
                        raise ValueError("Attributed fills exceed the order quantity")
                    weights = NamedMap((f"signal:{name}", weight) for name, weight in linked.weights.items())
                else:
                    weights = NamedMap({"unattributed": 1.0})
                for source, weight in weights.items():
                    row = rows[event.member, source]
                    quantity = event.quantity * weight
                    row["fill_quantity"] += quantity
                    row["fill_notional"] += quantity * event.price
                    row["fees"] += event.fee * weight
                    quantities[event.member, source] += quantity
            elif isinstance(event, Income):
                if event.member is None:
                    cash_income[i] += event.amount
                    continue
                held = sum(quantities[event.member, source] for source in sources)
                if abs(held) <= 1e-10:
                    rows[event.member, "unattributed"]["income"] += event.amount
                else:
                    for source in sources:
                        rows[event.member, source]["income"] += event.amount * quantities[event.member, source] / held
        for key, row in rows.items():
            for name, value in row.items():
                columns[key][name].append(value)
            quantity_paths[key].append(quantities[key])

    reports = {}
    for member in members:
        by_source = NamedMap(
            (
                source,
                instrument_plan.over(
                    axis,
                    NamedMap((name, Series(axis, tuple(values))) for name, values in columns[member, source].items()),
                ),
            )
            for source in sources
        )
        reports[member] = InstrumentReport(
            path.positions[member],
            sum_columns(axis, tuple(by_source.values())),
            by_source,
            NamedMap((source, Series(times, tuple(quantity_paths[member, source]))) for source in sources),
        )
    instruments = NamedMap(reports)
    cash = model_plan(Profit("cash_report"), ("holding", "trading", "income", "fees"), PROFIT_FIELDS).over(
        axis,
        NamedMap(
            {
                **{name: Series(axis, (0.0,) * len(axis)) for name in ("holding", "trading", "fees")},
                "income": Series(axis, tuple(cash_income)),
            }
        ),
    )
    source_reports = NamedMap(
        (
            source,
            sum_columns(
                axis,
                tuple(instrument.sources[source] for instrument in instruments.values())
                + ((cash,) if source == "unattributed" else ()),
            ),
        )
        for source in sources
    )
    totals = sum_columns(axis, tuple(source_reports.values()))
    account = account_plan.over(
        axis,
        NamedMap(
            {
                **{name: totals[name] for name in ("holding", "trading", "income", "fees")},
                "opening_nav": path.nav.lag(),
                "closing_nav": Series(axis, path.nav.values[1:]),
                "external_flows": path.pnl.external_flows,
            }
        ),
    )
    if any(abs(value) > 1e-8 for value in account["residual"].values):
        raise ValueError("Attributed investment P&L and external capital must reconcile to NAV")
    return AccountReport(
        path,
        account,
        instruments,
        source_reports,
        signals,
        attributions,
        utilities.map(lambda definition: UtilityReport(definition, definition.over(path.nav))),
        instrument_plan,
        account_plan,
    )

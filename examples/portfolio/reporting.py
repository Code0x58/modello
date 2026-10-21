"""Signal-level profit attribution, account reports, and cash utility."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from math import isclose, isfinite
from typing import Generic, TypeVar

from sympy import Expr, Rational, Symbol, diff, exp, log, simplify, sympify

from examples.data import NamedMap, Series, Time
from examples.portfolio.account import AccountPath, PositionHistory
from examples.portfolio.analytics import Comparison
from examples.portfolio.models import AccountPnL, InstrumentPnL, Profit
from examples.portfolio.signals import OrderIntent
from examples.portfolio.sources import CashFlow, Fill, Income
from examples.portfolio.valuation import EquityHoldings
from examples.symbolic import Known, ModelPlan, ScalarPlan


@dataclass(frozen=True)
class CashUtility:
    """An increasing utility expression with wealth as its sole free symbol.

    Wealth is measured in reporting-currency units. Utility levels depend on
    normalization; the derivative ratios survive positive affine transformations.
    """

    name: str
    wealth: Symbol
    expression: Expr

    def __post_init__(self) -> None:
        expression = sympify(self.expression)
        if not self.name or not isinstance(self.wealth, Symbol) or expression.free_symbols - {self.wealth}:
            raise ValueError("Name a utility expression with wealth as its only free symbol")
        if diff(expression, self.wealth) == 0:
            raise ValueError("Utility must vary with wealth")
        object.__setattr__(self, "expression", expression)

    @cached_property
    def expressions(self) -> NamedMap[Expr]:
        """Keep the utility, its derivatives, and their ratios inspectable."""
        first = diff(self.expression, self.wealth)
        second = diff(first, self.wealth)
        absolute = simplify(-second / first)
        return NamedMap(
            {
                "utility": self.expression,
                "marginal_utility": first,
                "curvature": second,
                "absolute_risk_aversion": absolute,
                "relative_risk_aversion": simplify(self.wealth * absolute),
            }
        )

    @cached_property
    def plan(self) -> ScalarPlan:
        return ScalarPlan(NamedMap({"wealth": self.wealth}), self.expressions)

    def over(self, wealth: Series[Time]) -> NamedMap[Series[Time]]:
        """Evaluate utility and local risk preferences on a positive wealth path."""
        if any(value <= 0 for value in wealth.values):
            raise ValueError("Utility reporting requires positive wealth")
        try:
            result = self.plan.over(wealth.times, NamedMap({"wealth": wealth}))
        except (ArithmeticError, ValueError) as error:
            raise ValueError("Utility and its derivative ratios must be finite on the wealth path") from error
        if any(value <= 0 for value in result["marginal_utility"].values):
            raise ValueError("Utility must be increasing on the wealth path")
        return result

    @staticmethod
    def _coefficient(value: float) -> Expr:
        if not isfinite(value) or value < 0:
            raise ValueError("Risk-aversion coefficients must be finite and nonnegative")
        return Rational(str(value))

    @classmethod
    def cara(cls, absolute_risk_aversion: float) -> CashUtility:
        """Constant absolute risk aversion, with a linear risk-neutral limit."""
        alpha = cls._coefficient(absolute_risk_aversion)
        wealth = Symbol("wealth", positive=True)
        expression = (1 - exp(-alpha * wealth)) / alpha if alpha else wealth
        return cls(f"CARA (alpha={absolute_risk_aversion:g} per USD)", wealth, expression)

    @classmethod
    def crra(cls, relative_risk_aversion: float) -> CashUtility:
        """Constant relative risk aversion, including logarithmic utility at gamma=1."""
        gamma = cls._coefficient(relative_risk_aversion)
        wealth = Symbol("wealth", positive=True)
        expression = log(wealth) if gamma == 1 else (wealth ** (1 - gamma) - 1) / (1 - gamma)
        return cls(f"CRRA (gamma={relative_risk_aversion:g})", wealth, expression)


_Members = TypeVar("_Members", bound=EquityHoldings)


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


@dataclass
class _InstrumentInterval:
    """Mutable ledger aggregates; symbolic P&L remains on InstrumentPnL."""

    opening_quantity: float
    opening_price: float
    closing_price: float
    fill_quantity: float = 0.0
    fill_notional: float = 0.0
    income: float = 0.0
    fees: float = 0.0


@dataclass(frozen=True)
class AccountReport(Generic[_Members]):
    """Typed, inspectable reporting layers; every monetary attribution adds back to the account."""

    path: AccountPath[_Members]
    account: NamedMap[Series[datetime]]
    instruments: NamedMap[InstrumentReport]
    sources: NamedMap[NamedMap[Series[datetime]]]
    signals: NamedMap[NamedMap[Series[datetime]]]
    attributions: tuple[OrderAttribution, ...]
    utilities: NamedMap[UtilityReport]
    instrument_plan: ModelPlan[InstrumentPnL]
    account_plan: ModelPlan[AccountPnL]

    def to_dict(self) -> dict[str, object]:
        from examples.portfolio.report_html import report_data

        return report_data(self)

    def to_html(self, comparison: Comparison[datetime], *, context: str = "") -> str:
        from examples.portfolio.report_html import render_html

        return render_html(self, comparison, context=context)

    @classmethod
    def from_account(
        cls,
        path: AccountPath[_Members],
        attributions: Iterable[OrderAttribution],
        *,
        signals: NamedMap[NamedMap[Series[datetime]]],
        utilities: NamedMap[CashUtility],
    ) -> AccountReport[_Members]:
        """Attribute executed decisions relative to the unchanged opening reference book.

        Signal sleeves start empty and retain signed incremental shares from tagged
        fills. Instrument income follows sleeve shares at the income event. Untagged
        fills, unassignable income, and account cash income remain visible as unattributed.
        """
        attributions = tuple(attributions)
        by_order = NamedMap((attribution.order.id, attribution) for attribution in attributions)
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
        for attribution in attributions:
            order = attribution.order
            if order.account != path.snapshots[0].account or order.member not in members:
                raise ValueError("Attribution must identify an order for this account and instrument")
            if order.equity != path.snapshots[0].instruments[order.member]:
                raise ValueError("Attribution must match the declared instrument identity")
            for signal, weight in attribution.weights.items():
                if signal not in signals or order.member not in signals[signal]:
                    raise ValueError("Attribution needs a named signal history for the order's instrument")
                if weight:
                    signals[signal][order.member].asof(order.decision_at)

        instrument_plan = InstrumentPnL.plan()
        account_plan = AccountPnL.plan()
        # The mutable state is only the ledger-to-vector adapter. Formula evaluation
        # below runs through the symbolic models, compiled once per report.
        quantities = {
            (member, source): path.positions[member].quantity.values[0] if source == "opening" else 0.0
            for member in members
            for source in sources
        }
        quantity_paths = {key: [value] for key, value in quantities.items()}
        columns: dict[tuple[str, str], list[_InstrumentInterval]] = {key: [] for key in quantities}
        cash_income = [0.0] * len(axis)
        filled = {key: 0.0 for key in by_order}
        cursor = 0
        for i, end in enumerate(axis):
            rows = {
                (member, source): _InstrumentInterval(
                    quantities[member, source],
                    path.positions[member].price.values[i],
                    path.positions[member].price.values[i + 1],
                )
                for member in members
                for source in sources
            }
            while cursor < len(path.ledger.events) and path.ledger.events[cursor].at <= end:
                event = path.ledger.events[cursor]
                cursor += 1
                if isinstance(event, CashFlow):
                    continue
                if isinstance(event, Fill):
                    linked = by_order.get(event.order_id or "")
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
                        row.fill_quantity += quantity
                        row.fill_notional += quantity * event.price
                        row.fees += event.fee * weight
                        quantities[event.member, source] += quantity
                elif isinstance(event, Income):
                    if event.member is None:
                        cash_income[i] += event.amount
                        continue
                    held = sum(quantities[event.member, source] for source in sources)
                    if abs(held) <= 1e-10:
                        rows[event.member, "unattributed"].income += event.amount
                    else:
                        for source in sources:
                            rows[event.member, source].income += event.amount * quantities[event.member, source] / held
            for key, row in rows.items():
                columns[key].append(row)
                quantity_paths[key].append(quantities[key])

        model = instrument_plan.source

        def profit(rows: list[_InstrumentInterval]) -> NamedMap[Series[datetime]]:
            result = instrument_plan.over(
                axis,
                Known(model.opening_quantity, Series(axis, tuple(row.opening_quantity for row in rows))),
                Known(model.opening_price, Series(axis, tuple(row.opening_price for row in rows))),
                Known(model.closing_price, Series(axis, tuple(row.closing_price for row in rows))),
                Known(model.fill_quantity, Series(axis, tuple(row.fill_quantity for row in rows))),
                Known(model.fill_notional, Series(axis, tuple(row.fill_notional for row in rows))),
                Known(model.income, Series(axis, tuple(row.income for row in rows))),
                Known(model.fees, Series(axis, tuple(row.fees for row in rows))),
            )
            return Profit.columns(result)

        reports = {}
        for member in members:
            by_source = NamedMap((source, profit(columns[member, source])) for source in sources)
            reports[member] = InstrumentReport(
                path.positions[member],
                Profit.sum(axis, tuple(by_source.values())),
                by_source,
                NamedMap((source, Series(times, tuple(quantity_paths[member, source]))) for source in sources),
            )
        instruments = NamedMap(reports)
        cash_plan = Profit.plan()
        cash_model = cash_plan.source
        zero = Series(axis, (0.0,) * len(axis))
        cash_result = cash_plan.over(
            axis,
            Known(cash_model.holding, zero),
            Known(cash_model.trading, zero),
            Known(cash_model.fees, zero),
            Known(cash_model.income, Series(axis, tuple(cash_income))),
        )
        cash = Profit.columns(cash_result)
        source_reports = NamedMap(
            (
                source,
                Profit.sum(
                    axis,
                    tuple(instrument.sources[source] for instrument in instruments.values())
                    + ((cash,) if source == "unattributed" else ()),
                ),
            )
            for source in sources
        )
        totals = Profit.sum(axis, tuple(source_reports.values()))
        account_model = account_plan.source
        result = account_plan.over(
            axis,
            Known(account_model.holding, totals["holding"]),
            Known(account_model.trading, totals["trading"]),
            Known(account_model.income, totals["income"]),
            Known(account_model.fees, totals["fees"]),
            Known(account_model.opening_nav, path.nav.lag()),
            Known(account_model.closing_nav, Series(axis, path.nav.values[1:])),
            Known(account_model.external_flows, path.pnl.external_flows),
        )
        account = NamedMap(
            {
                **Profit.columns(result),
                "nav_change": result.column(account_model.nav_change),
                "external_flows": result.column(account_model.external_flows),
                "residual": result.column(account_model.residual),
            }
        )
        if any(abs(value) > 1e-8 for value in account["residual"].values):
            raise ValueError("Attributed investment P&L and external capital must reconcile to NAV")
        return cls(
            path=path,
            account=account,
            instruments=instruments,
            sources=source_reports,
            signals=signals,
            attributions=attributions,
            utilities=utilities.map(lambda definition: UtilityReport(definition, definition.over(path.nav))),
            instrument_plan=instrument_plan,
            account_plan=account_plan,
        )

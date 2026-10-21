"""Replay an external ledger into valued positions and cash-flow-adjusted returns."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from math import isclose
from typing import TYPE_CHECKING, Generic, TypeVar

from sympy import Symbol

from examples.data import NamedMap, Series, utc_time
from examples.portfolio.analytics import NamedAnalytics, analyse
from examples.portfolio.models import (
    AccountFill,
    AccountPnL,
    AccountPosting,
    AccountState,
    CapitalFlow,
    Equity,
    IncomePosting,
    InstrumentPnL,
    Profit,
)
from examples.portfolio.sources import CashFlow, Fill, Income, LedgerEvent, PriceHistory, TransactionLog, finite
from examples.portfolio.valuation import EquityHoldings, MarketSnapshot, NamedPortfolio
from examples.symbolic import Known

if TYPE_CHECKING:
    from examples.named_members import MemberValues
    from examples.portfolio.models import EquityPosition
    from examples.portfolio.reporting import AccountReport, CashUtility, OrderAttribution
    from examples.portfolio.signals import OrderIntent, TargetWeights

_Members = TypeVar("_Members", bound=EquityHoldings)


@dataclass(frozen=True)
class Account(Generic[_Members]):
    """An opening portfolio and the time at which its balances apply."""

    name: str
    opened_at: datetime
    opening: NamedPortfolio[_Members]

    def __post_init__(self) -> None:
        object.__setattr__(self, "opened_at", utc_time(self.opened_at))
        if not self.name:
            raise ValueError("An account needs a name")

    def replay(
        self,
        ledger: TransactionLog,
        times: tuple[datetime, ...],
        *,
        market: NamedMap[PriceHistory],
        max_price_age: timedelta,
    ) -> AccountPath[_Members]:
        """Replay an explicitly supplied ledger and market snapshot through the models."""
        return AccountEngine(self, market, max_price_age=max_price_age).evaluate(ledger, times)


@dataclass(frozen=True)
class AccountSnapshot(Generic[_Members]):
    """A valued account after postings at this timestamp."""

    at: datetime
    known_at: datetime
    account: str
    portfolio: NamedPortfolio[_Members]
    units: float

    @cached_property
    def instruments(self) -> NamedMap[Equity]:
        return self.portfolio.instruments

    @cached_property
    def quantities(self) -> NamedMap[float]:
        return self.portfolio.positions.map(lambda position: float(position.quantity))

    @cached_property
    def prices(self) -> NamedMap[float]:
        return self.portfolio.positions.map(lambda position: float(position.price))

    @cached_property
    def values(self) -> NamedMap[float]:
        return self.portfolio.values.map(float)

    @property
    def cash(self) -> float:
        return float(self.portfolio.cash.amount)

    @property
    def nav(self) -> float:
        return float(self.portfolio.nav)

    @cached_property
    def weights(self) -> NamedMap[float]:
        return self.portfolio.weights.map(float)

    @property
    def unit_nav(self) -> float:
        return self.nav / self.units

    def knowns(self, model: AccountState) -> tuple[Known[float], ...]:
        """Supply the account balances to a transition's state component."""
        return (Known(model.cash, self.cash), Known(model.nav, self.nav), Known(model.units, self.units))

    def rebalance(
        self, targets: TargetWeights, *, batch: str, arrival_at: datetime
    ) -> MemberValues[EquityPosition, OrderIntent]:
        """Solve the target-allocation model using this decision-time account view."""
        return targets.orders(self, batch=batch, arrival_at=arrival_at)


@dataclass(frozen=True)
class AccountStep(Generic[_Members]):
    """One observed event and its evaluated before/event/after model."""

    before: AccountSnapshot[_Members]
    event: LedgerEvent
    posting: AccountPosting
    after: AccountSnapshot[_Members]


@dataclass(frozen=True)
class PositionHistory:
    """Time-varying shares, market marks, valuation, and allocation."""

    quantity: Series[datetime]
    price: Series[datetime]
    value: Series[datetime]
    weight: Series[datetime]


@dataclass(frozen=True)
class PnL:
    """Interval profit in currency units; external capital is shown separately."""

    holding: NamedMap[Series[datetime]]
    trading: NamedMap[Series[datetime]]
    income: Series[datetime]
    fees: Series[datetime]
    external_flows: Series[datetime]

    @cached_property
    def totals(self) -> NamedMap[Series[datetime]]:
        times = self.income.times
        view = Profit.plan()
        model = view.source
        total = view.over(
            times,
            Known(
                model.holding,
                Series(
                    times, tuple(sum(series.values[i] for series in self.holding.values()) for i in range(len(times)))
                ),
            ),
            Known(
                model.trading,
                Series(
                    times, tuple(sum(series.values[i] for series in self.trading.values()) for i in range(len(times)))
                ),
            ),
            Known(model.income, self.income),
            Known(model.fees, self.fees),
        )
        return Profit.columns(total)

    @property
    def total(self) -> Series[datetime]:
        return self.totals["net"]


@dataclass(frozen=True)
class AccountPath(Generic[_Members]):
    """Valuation snapshots and the ledger known by the reporting horizon."""

    snapshots: tuple[AccountSnapshot[_Members], ...]
    ledger: TransactionLog
    transitions: tuple[AccountStep[_Members], ...] = ()

    def series(self, project: Callable[[AccountSnapshot[_Members]], float]) -> Series[datetime]:
        return Series(
            tuple(snapshot.at for snapshot in self.snapshots), tuple(project(snapshot) for snapshot in self.snapshots)
        )

    @cached_property
    def nav(self) -> Series[datetime]:
        return self.series(lambda snapshot: snapshot.nav)

    @cached_property
    def unit_nav(self) -> Series[datetime]:
        """A performance index that issues or redeems units for external flows."""
        return self.series(lambda snapshot: snapshot.unit_nav)

    @cached_property
    def cash(self) -> Series[datetime]:
        return self.series(lambda snapshot: snapshot.cash)

    @cached_property
    def positions(self) -> NamedMap[PositionHistory]:
        return NamedMap(
            (
                name,
                PositionHistory(
                    self.series(lambda snapshot: snapshot.quantities[name]),
                    self.series(lambda snapshot: snapshot.prices[name]),
                    self.series(lambda snapshot: snapshot.values[name]),
                    self.series(lambda snapshot: snapshot.values[name] / snapshot.nav),
                ),
            )
            for name in self.snapshots[0].quantities
        )

    @cached_property
    def pnl(self) -> PnL:
        times = self.nav.times
        positions = self.positions
        axis = times[1:]
        fill_quantities = {name: [0.0] * len(axis) for name in positions}
        notionals = {name: [0.0] * len(axis) for name in positions}
        income, fees, flows = ([0.0] * len(axis) for _ in range(3))
        for event in self.ledger.events:
            index = bisect_left(times, event.at) - 1
            if isinstance(event, Fill):
                fill_quantities[event.member][index] += event.quantity
                notionals[event.member][index] += event.quantity * event.price
                fees[index] += event.fee
            elif isinstance(event, CashFlow):
                flows[index] += event.amount
            else:
                income[index] += event.amount
        zero = Series(axis, (0.0,) * len(axis))
        view = InstrumentPnL.plan()
        model = view.source
        reports = NamedMap(
            (
                name,
                view.over(
                    axis,
                    Known(model.opening_quantity, position.quantity.lag()),
                    Known(model.opening_price, position.price.lag()),
                    Known(model.closing_price, Series(axis, position.price.values[1:])),
                    Known(model.fill_quantity, Series(axis, tuple(fill_quantities[name]))),
                    Known(model.fill_notional, Series(axis, tuple(notionals[name]))),
                    Known(model.income, zero),
                    Known(model.fees, zero),
                ),
            )
            for name, position in positions.items()
        )
        return PnL(
            reports.map(lambda report: report.column(model.holding)),
            reports.map(lambda report: report.column(model.trading)),
            Series(axis, tuple(income)),
            Series(axis, tuple(fees)),
            Series(axis, tuple(flows)),
        )

    @cached_property
    def reconciliation(self) -> Series[datetime]:
        """NAV change minus investment P&L and external flows, ideally zero."""
        pnl = self.pnl
        times = self.nav.times[1:]
        view = AccountPnL.plan()
        model = view.source
        result = view.over(
            times,
            Known(model.opening_nav, self.nav.lag()),
            Known(model.closing_nav, Series(times, self.nav.values[1:])),
            Known(model.holding, pnl.totals["holding"]),
            Known(model.trading, pnl.totals["trading"]),
            Known(model.income, pnl.income),
            Known(model.fees, pnl.fees),
            Known(model.external_flows, pnl.external_flows),
        )
        return result.column(model.residual)

    def analytics(
        self, *, periods_per_year: int, risk_free: Series[datetime] | None = None
    ) -> NamedAnalytics[datetime]:
        return analyse(
            self.unit_nav,
            self.positions.map(lambda position: position.price.simple_returns()),
            self.snapshots[-1].weights,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )

    def report(
        self,
        attributions: Iterable[OrderAttribution],
        *,
        signals: NamedMap[NamedMap[Series[datetime]]],
        utilities: NamedMap[CashUtility],
    ) -> AccountReport[_Members]:
        """Add symbolic reporting and an explicit signal-attribution convention."""
        from examples.portfolio.reporting import AccountReport

        return AccountReport.from_account(self, attributions, signals=signals, utilities=utilities)


class AccountEngine(Generic[_Members]):
    """Advance typed account states through compiled event models and market marks."""

    def __init__(
        self,
        account: Account[_Members],
        market: NamedMap[PriceHistory],
        *,
        max_price_age: timedelta,
    ) -> None:
        self.account = account
        self.market = account.opening.match_market(market)
        if max_price_age < timedelta(0):
            raise ValueError("Price age must be nonnegative")
        self.max_price_age = max_price_age
        prices = account.opening.positions.map(lambda position: position.price)
        if not all(isinstance(price, Symbol) for price in prices.values()) or len(set(prices.values())) != len(prices):
            raise ValueError("Leave opening prices as independent symbolic inputs for the market source")
        try:
            self._opening_quantities = account.opening.positions.map(lambda position: float(position.quantity))
            self._opening_cash = float(account.opening.cash.amount)
        except (TypeError, ValueError) as error:
            raise ValueError("Specify all opening quantities and cash") from error
        finite(*self._opening_quantities.values(), self._opening_cash)
        self._check_balances(self._opening_quantities, self._opening_cash)
        self._plan = type(account.opening).plan()
        self._fill_plan = AccountFill.plan()
        self._capital_plan = CapitalFlow.plan()
        self._income_plan = IncomePosting.plan()

    @staticmethod
    def _check_balances(quantities: Mapping[str, float], cash: float) -> None:
        if cash < -1e-8 or any(quantity < -1e-10 for quantity in quantities.values()):
            raise ValueError("This account requires funded cash and nonnegative positions")

    def _value(self, at: datetime, quantities: NamedMap[float], cash: float) -> NamedPortfolio[_Members]:
        snapshot = MarketSnapshot.sample(self.market, at=at, max_price_age=self.max_price_age)
        model = self._plan.source
        quantity_inputs = model.positions.combine(quantities, lambda position, value: Known(position.quantity, value))
        return self._plan.evaluate(snapshot, *quantity_inputs.values(), Known(model.cash.amount, cash))

    def _mark(self, state: AccountSnapshot[_Members], at: datetime) -> AccountSnapshot[_Members]:
        """Change market observations while carrying the same account balances."""
        if state.at == at:
            return state
        return AccountSnapshot(
            at, state.known_at, state.account, self._value(at, state.quantities, state.cash), state.units
        )

    def _advance(self, state: AccountSnapshot[_Members], event: LedgerEvent) -> AccountStep[_Members]:
        """Bind a ledger variant to its model, retaining the resulting state."""
        before = self._mark(state, event.at)
        quantities = before.quantities
        changed: AccountPosting
        if isinstance(event, CashFlow):
            capital = self._capital_plan.source
            changed = self._capital_plan.evaluate(
                *before.knowns(capital.before), Known(capital.flow.amount, event.amount)
            )
        elif isinstance(event, Fill):
            if event.member not in quantities:
                raise ValueError(f"Unknown transaction member {event.member!r}")
            fill = self._fill_plan.source
            posted = self._fill_plan.evaluate(
                *before.knowns(fill.before),
                Known(fill.settlement.before.quantity, quantities[event.member]),
                Known(fill.settlement.fill.quantity, event.quantity),
                Known(fill.settlement.fill.price, event.price),
                Known(fill.settlement.fill.fee, event.fee),
                Known(fill.mark.price, before.prices[event.member]),
            )
            quantities = NamedMap(
                (name, float(posted.settlement.quantity) if name == event.member else value)
                for name, value in quantities.items()
            )
            changed = posted
        elif isinstance(event, Income):
            if event.member is not None and event.member not in quantities:
                raise ValueError(f"Unknown income member {event.member!r}")
            income = self._income_plan.source
            changed = self._income_plan.evaluate(
                *before.knowns(income.before), Known(income.income.amount, event.amount)
            )
        else:
            raise TypeError("Unsupported ledger event")
        cash = float(changed.cash)
        self._check_balances(quantities, cash)
        values = self._value(event.at, quantities.map(lambda quantity: max(0.0, quantity)), max(0.0, cash))
        if not isclose(float(values.nav), float(changed.nav), rel_tol=0, abs_tol=1e-8):
            raise ArithmeticError("Posting NAV does not reconcile to the resulting portfolio")
        after = AccountSnapshot(event.at, before.known_at, before.account, values, float(changed.units))
        return AccountStep(before, event, changed, after)

    def evaluate(self, ledger: TransactionLog, times: tuple[datetime, ...]) -> AccountPath[_Members]:
        """Report using knowledge at the final mark; balances include (opening, mark]."""
        times = tuple(utc_time(at) for at in times)
        if len(times) < 2 or times[0] != self.account.opened_at or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Marks must increase from the account opening time and include at least two points")
        events = tuple(
            event for event in ledger.known(self.account.name, times[-1]).events if times[0] < event.at <= times[-1]
        )
        opening = self._value(times[0], self._opening_quantities, self._opening_cash)
        state = AccountSnapshot(times[0], times[-1], self.account.name, opening, float(opening.nav) / 100)
        snapshots = []
        transitions = []
        cursor = 0
        for at in times:
            while cursor < len(events) and events[cursor].at <= at:
                step = self._advance(state, events[cursor])
                transitions.append(step)
                state = step.after
                cursor += 1
            state = self._mark(state, at)
            snapshots.append(state)
        path = AccountPath(tuple(snapshots), TransactionLog(events), tuple(transitions))
        if not all(isclose(value, 0, abs_tol=1e-8) for value in path.reconciliation.values):
            raise ArithmeticError("Account NAV does not reconcile to postings and position P&L")
        return path

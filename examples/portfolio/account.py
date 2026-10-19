"""Replay an external ledger into valued positions and cash-flow-adjusted returns."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import cached_property
from math import isclose
from typing import TYPE_CHECKING, Generic, TypeVar

from sympy import Symbol

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, utc_time
from examples.portfolio.analytics import NamedAnalytics, analyse
from examples.portfolio.basket import EquityHoldings
from examples.portfolio.models import Equity
from examples.portfolio.named import NamedPortfolio
from examples.portfolio.pnl import AccountPnL, InstrumentPnL, Profit
from examples.portfolio.postings import CapitalFlow, FillPosting
from examples.portfolio.sources import CashFlow, Fill, Income, PriceHistory, TransactionLog, finite

if TYPE_CHECKING:
    from examples.portfolio.reporting import AccountReport, OrderAttribution
    from examples.portfolio.signals import OrderIntent, TargetWeights
    from examples.portfolio.utility import CashUtility

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
    ) -> AccountPath:
        """Replay an explicitly supplied ledger and market snapshot through the models."""
        return AccountEngine(self, market, max_price_age=max_price_age).evaluate(ledger, times)


@dataclass(frozen=True)
class AccountSnapshot:
    """A valued account after postings at this timestamp."""

    at: datetime
    known_at: datetime
    account: str
    instruments: NamedMap[Equity]
    quantities: NamedMap[float]
    prices: NamedMap[float]
    values: NamedMap[float]
    cash: float
    nav: float
    units: float

    @property
    def weights(self) -> NamedMap[float]:
        return self.values.map(lambda value: value / self.nav)

    @property
    def unit_nav(self) -> float:
        return self.nav / self.units

    def rebalance(self, targets: TargetWeights, *, batch: str, arrival_at: datetime) -> NamedMap[OrderIntent]:
        """Solve the target-allocation model using this decision-time account view."""
        return targets.orders(self, batch=batch, arrival_at=arrival_at)


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
        total = Profit.over(
            times,
            NamedMap(
                {
                    "holding": Series(
                        times,
                        tuple(sum(series.values[i] for series in self.holding.values()) for i in range(len(times))),
                    ),
                    "trading": Series(
                        times,
                        tuple(sum(series.values[i] for series in self.trading.values()) for i in range(len(times))),
                    ),
                    "income": self.income,
                    "fees": self.fees,
                }
            ),
        )
        return total

    @property
    def total(self) -> Series[datetime]:
        return self.totals["net"]


@dataclass(frozen=True)
class AccountPath:
    """Valuation snapshots and the ledger known by the reporting horizon."""

    snapshots: tuple[AccountSnapshot, ...]
    ledger: TransactionLog

    def series(self, project: Callable[[AccountSnapshot], float]) -> Series[datetime]:
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
        reports = NamedMap(
            (
                name,
                InstrumentPnL.over(
                    axis,
                    NamedMap(
                        {
                            "opening_quantity": position.quantity.lag(),
                            "opening_price": position.price.lag(),
                            "closing_price": Series(axis, position.price.values[1:]),
                            "fill_quantity": Series(axis, tuple(fill_quantities[name])),
                            "fill_notional": Series(axis, tuple(notionals[name])),
                            "income": zero,
                            "fees": zero,
                        }
                    ),
                ),
            )
            for name, position in positions.items()
        )
        return PnL(
            reports.map(lambda report: report["holding"]),
            reports.map(lambda report: report["trading"]),
            Series(axis, tuple(income)),
            Series(axis, tuple(fees)),
            Series(axis, tuple(flows)),
        )

    @cached_property
    def reconciliation(self) -> Series[datetime]:
        """NAV change minus investment P&L and external flows, ideally zero."""
        pnl = self.pnl
        times = self.nav.times[1:]
        return AccountPnL.over(
            times,
            NamedMap(
                {
                    "opening_nav": self.nav.lag(),
                    "closing_nav": Series(times, self.nav.values[1:]),
                    **{name: pnl.totals[name] for name in ("holding", "trading", "income", "fees")},
                    "external_flows": pnl.external_flows,
                }
            ),
        )["residual"]

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
        attributions: NamedMap[OrderAttribution],
        *,
        signals: NamedMap[NamedMap[Series[datetime]]],
        utilities: NamedMap[CashUtility],
    ) -> AccountReport:
        """Add symbolic reporting and an explicit signal-attribution convention."""
        from examples.portfolio.reporting import AccountReport

        return AccountReport.from_account(self, attributions, signals=signals, utilities=utilities)


class AccountEngine:
    """Use one compiled valuation and an ordinary ledger state transition."""

    def __init__(
        self,
        account: Account[_Members],
        market: NamedMap[PriceHistory],
        *,
        max_price_age: timedelta,
    ) -> None:
        self.account = account
        instruments = account.opening.instruments
        self.instruments = instruments

        def checked(instrument: Equity, history: PriceHistory) -> PriceHistory:
            if history.equity != instrument or instrument.currency != "USD":
                raise ValueError("Account price histories must match the declared USD instruments")
            return history

        self.market = instruments.combine(market, checked)
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
        self._check_balances(dict(self._opening_quantities), self._opening_cash)
        self._plan = type(account.opening).plan()

    @staticmethod
    def _check_balances(quantities: dict[str, float], cash: float) -> None:
        if cash < -1e-8 or any(quantity < -1e-10 for quantity in quantities.values()):
            raise ValueError("This account requires funded cash and nonnegative positions")

    def _value(self, at: datetime, quantities: NamedMap[float], cash: float) -> tuple[NamedMap[float], NamedMap[float]]:
        prices = self.market.map(lambda history: history.mark(at, max_age=self.max_price_age))
        inputs = NamedMap(
            [
                *((f"price:{name}", value) for name, value in prices.items()),
                *((f"quantity:{name}", value) for name, value in quantities.items()),
                ("cash", cash),
            ]
        )
        values = self._plan.evaluate(inputs)
        if values["nav"] <= 0:
            raise ValueError("Account value must remain positive")
        return prices, values

    def evaluate(self, ledger: TransactionLog, times: tuple[datetime, ...]) -> AccountPath:
        """Report using knowledge at the final mark; balances include (opening, mark]."""
        times = tuple(utc_time(at) for at in times)
        if len(times) < 2 or times[0] != self.account.opened_at or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Marks must increase from the account opening time and include at least two points")
        events = tuple(
            event for event in ledger.known(self.account.name, times[-1]).events if times[0] < event.at <= times[-1]
        )
        quantities = dict(self._opening_quantities)
        cash = self._opening_cash
        _, opening = self._value(times[0], NamedMap(quantities), cash)
        units = opening["nav"] / 100
        snapshots = []
        cursor = 0
        for at in times:
            while cursor < len(events) and events[cursor].at <= at:
                event = events[cursor]
                if isinstance(event, CashFlow):
                    _, before = self._value(event.at, NamedMap(quantities), cash)
                    changed = CapitalFlow.plan().evaluate(
                        NamedMap(
                            {
                                "opening_cash": cash,
                                "opening_nav": before["nav"],
                                "opening_units": units,
                                "amount": event.amount,
                            }
                        )
                    )
                    cash, units = changed["closing_cash"], changed["closing_units"]
                elif isinstance(event, Fill):
                    if event.member not in quantities:
                        raise ValueError(f"Unknown transaction member {event.member!r}")
                    changed = FillPosting.plan().evaluate(
                        NamedMap(
                            {
                                "opening_quantity": quantities[event.member],
                                "opening_cash": cash,
                                "quantity": event.quantity,
                                "price": event.price,
                                "fee": event.fee,
                            }
                        )
                    )
                    quantities[event.member], cash = changed["closing_quantity"], changed["closing_cash"]
                elif isinstance(event, Income):
                    if event.member is not None and event.member not in quantities:
                        raise ValueError(f"Unknown income member {event.member!r}")
                    cash += event.amount
                self._check_balances(quantities, cash)
                quantities = {name: max(0.0, value) for name, value in quantities.items()}
                cash = max(0.0, cash)
                if units <= 0:
                    raise ValueError("External flows must leave positive account units")
                cursor += 1
            prices, values = self._value(at, NamedMap(quantities), cash)
            snapshots.append(
                AccountSnapshot(
                    at,
                    times[-1],
                    self.account.name,
                    self.instruments,
                    NamedMap(quantities),
                    prices,
                    NamedMap((name, values[f"value:{name}"]) for name in quantities),
                    cash,
                    values["nav"],
                    units,
                )
            )
        path = AccountPath(tuple(snapshots), TransactionLog(events))
        if not all(isclose(value, 0, abs_tol=1e-8) for value in path.reconciliation.values):
            raise ArithmeticError("Account NAV does not reconcile to postings and position P&L")
        return path

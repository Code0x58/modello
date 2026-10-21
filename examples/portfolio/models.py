"""Composable financial equations: positions, state transitions, and profit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import ClassVar, Generic, TypeVar

from examples.data import NamedMap, Series, Time
from examples.symbolic import ModelPath, SymbolicModel
from modello import InstanceDummy, Modello, component


@dataclass(frozen=True)
class Equity:
    """Serializable quote identity, separate from a configured data request."""

    symbol: str
    provider: str
    currency: str = "USD"

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip() for value in (self.symbol, self.provider, self.currency)
        ):
            raise ValueError("An equity needs a symbol, provider, and quote currency")


AAPL = Equity("AAPL", provider="synthetic")


MSFT = Equity("MSFT", provider="synthetic")


GOOGL = Equity("GOOGL", provider="synthetic")


class Quote(Modello):
    """A symbolic price; provider identity comes from the holding declaration."""

    price = InstanceDummy("price", positive=True)


class PriceChange(Modello):
    """Two marks of the same instrument, in currency and proportional terms."""

    before = component(Quote)
    after = component(Quote)
    change = after.price - before.price
    simple_return = after.price / before.price - 1


class Position(Modello):
    """A marked position, including signed virtual attribution sleeves."""

    instrument = component(Quote)
    quantity = InstanceDummy("quantity", real=True)
    market_value = quantity * instrument.price


class EquityPosition(Position):
    """A long position in an equity, separate from the instrument itself."""

    quantity = InstanceDummy("quantity", nonnegative=True)


class AccountValue(Modello):
    """The net asset value at an account boundary."""

    nav = InstanceDummy("nav", positive=True)


class Trade(Modello):
    """A signed share exchange at a price, before settlement charges."""

    quantity = InstanceDummy("quantity", real=True)
    price = InstanceDummy("price", positive=True)
    notional = quantity * price


class Allocation(Modello):
    """A share of capital expressed as an amount and a weight."""

    capital = InstanceDummy("capital", positive=True)
    weight = InstanceDummy("weight", nonnegative=True)
    value = capital * weight


class USDCash(Modello):
    """A nonnegative USD cash balance with no interest accrual."""

    currency: ClassVar[str] = "USD"
    amount = InstanceDummy("amount", nonnegative=True)


@dataclass(frozen=True)
class EquityPrice:
    """An identified positive scalar quote, ready for portfolio evaluation."""

    equity: Equity
    price: float

    def __post_init__(self) -> None:
        if not isfinite(self.price) or self.price <= 0:
            raise ValueError("An equity quote must be finite and strictly positive")


@dataclass(frozen=True)
class EquityPrices(Generic[Time]):
    """An instrument's positive prices on a date or timestamp observation axis."""

    equity: Equity
    close: Series[Time]

    def __post_init__(self) -> None:
        """Keep invalid quotes out of valuation and return calculations."""
        if any(value <= 0 for value in self.close.values):
            raise ValueError("Equity prices must be strictly positive")


class HoldingState(Modello):
    """A holding's shares and the cash balance used to settle its next fill.

    Signed balances also support virtual attribution sleeves. Funded-account
    policies are enforced by the chronological account adapter.
    """

    quantity = InstanceDummy("quantity", real=True)
    cash = InstanceDummy("cash", real=True)


class Execution(Trade):
    """The economic terms of a signed fill, separate from its ledger identity."""

    fee = InstanceDummy("fee", nonnegative=True)


class FillPosting(HoldingState, SymbolicModel):
    """A holding state resulting from a prior state and an execution."""

    before = component(HoldingState)
    fill = component(Execution)
    after = component(HoldingState)

    quantity = after.quantity
    quantity = before.quantity + fill.quantity
    cash = after.cash
    cash = before.cash - fill.notional - fill.fee


class AccountState(AccountValue):
    """A valued account and its participation units at one event boundary."""

    cash = InstanceDummy("cash", real=True)
    units = InstanceDummy("units", positive=True)
    unit_price = AccountValue.nav / units


class CashMovement(Modello):
    """Signed cash received or paid by an account."""

    amount = InstanceDummy("amount", real=True)


class AccountTransition(AccountState, SymbolicModel):
    """A resulting account state, with both event boundaries retained."""

    before = component(AccountState)
    after = component(AccountState)

    cash = after.cash
    nav = after.nav
    units = after.units


class CapitalFlow(AccountTransition):
    """Issue or redeem participation units at the pre-flow unit price."""

    flow = component(CashMovement)

    issued_units = flow.amount / AccountTransition.before.unit_price
    cash = AccountTransition.before.cash + flow.amount
    nav = AccountTransition.before.nav + flow.amount
    units = AccountTransition.before.units + issued_units


class AccountFill(AccountTransition):
    """Settle a fill and mark its shares without issuing participation units."""

    settlement = component(FillPosting)
    mark = component(Quote)

    opening_cash = AccountTransition.before.cash
    opening_cash = settlement.before.cash
    cash = settlement.cash
    nav = AccountTransition.before.nav + settlement.fill.quantity * mark.price + cash - AccountTransition.before.cash
    units = AccountTransition.before.units


class IncomePosting(AccountTransition):
    """Book investment income or an expense without changing account units."""

    income = component(CashMovement)

    cash = AccountTransition.before.cash + income.amount
    nav = AccountTransition.before.nav + income.amount
    units = AccountTransition.before.units


AccountPosting = CapitalFlow | AccountFill | IncomePosting


PROFIT_FIELDS = ("holding", "trading", "income", "fees", "gross", "net")


_Profit = TypeVar("_Profit", bound="Profit")


class Profit(SymbolicModel):
    """Currency P&L shared by instrument and account reporting models."""

    holding = InstanceDummy("holding", real=True)
    trading = InstanceDummy("trading", real=True)
    income = InstanceDummy("income", real=True)
    fees = InstanceDummy("fees", nonnegative=True)
    gross = holding + trading + income
    net = gross - fees

    @staticmethod
    def columns(path: ModelPath[_Profit, Time]) -> NamedMap[Series[Time]]:
        """Project the typed calculation to the six columns of a profit table."""
        model = path.view.source
        return NamedMap(
            {
                "holding": path.column(model.holding),
                "trading": path.column(model.trading),
                "income": path.column(model.income),
                "fees": path.column(model.fees),
                "gross": path.column(model.gross),
                "net": path.column(model.net),
            }
        )

    @classmethod
    def sum(
        cls, times: tuple[datetime, ...], paths: tuple[NamedMap[Series[datetime]], ...]
    ) -> NamedMap[Series[datetime]]:
        """Reduce aligned report columns; their formulas remain on the model."""
        if any(
            set(path) != set(PROFIT_FIELDS) or any(series.times != times for series in path.values()) for path in paths
        ):
            raise ValueError("Profit reports must share named columns and interval times")
        return NamedMap(
            (name, Series(times, tuple(sum(path[name].values[i] for path in paths) for i in range(len(times)))))
            for name in PROFIT_FIELDS
        )


class ProfitView(Profit):
    """Expose a shared profit component through the standard reporting fields."""

    profit = component(Profit)
    holding = profit.holding
    trading = profit.trading
    income = profit.income
    fees = profit.fees
    gross = profit.gross
    net = profit.net


class TradingActivity(Modello):
    """Signed interval totals from the external transaction log."""

    quantity = InstanceDummy("quantity", real=True)
    notional = InstanceDummy("notional", real=True)
    income = InstanceDummy("income", real=True)
    fees = InstanceDummy("fees", nonnegative=True)


class InstrumentPnL(ProfitView):
    """Exact interval mark-to-market profit, including signed virtual sleeves."""

    before = component(Position)
    after = component(Position)
    activity = component(TradingActivity)
    prices = component(PriceChange)

    opening_quantity = before.quantity
    closing_quantity = after.quantity
    closing_quantity = before.quantity + activity.quantity
    opening_price = before.instrument.price
    opening_price = prices.before.price
    closing_price = after.instrument.price
    closing_price = prices.after.price
    fill_quantity = activity.quantity
    fill_notional = activity.notional
    income = activity.income
    fees = activity.fees
    holding = before.quantity * prices.change
    trading = activity.quantity * after.instrument.price - activity.notional


class AccountPnL(ProfitView):
    """Relate investment profit and external capital to account NAV changes."""

    before = component(AccountValue)
    after = component(AccountValue)
    capital = component(CashMovement)

    opening_nav = before.nav
    closing_nav = after.nav
    external_flows = capital.amount
    nav_change = after.nav - before.nav
    residual = nav_change - ProfitView.profit.net - capital.amount

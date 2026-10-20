"""The P&L equations shared by account valuation and attribution reports."""

from datetime import datetime
from typing import TypeVar

from examples.compute.named_map import NamedMap
from examples.compute.series import Series, Time
from examples.model_plan import ModelPath
from examples.portfolio.models import AccountValue, Position, PriceChange
from examples.portfolio.postings import CashMovement
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, component

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

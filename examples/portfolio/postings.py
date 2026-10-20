"""Typed state transitions for externally supplied fills and capital movements."""

from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, component


class HoldingState(Modello):
    """A holding's shares and the cash balance used to settle its next fill.

    Signed balances also support virtual attribution sleeves. Funded-account
    policies are enforced by the chronological account adapter.
    """

    quantity = InstanceDummy("quantity", real=True)
    cash = InstanceDummy("cash", real=True)


class Execution(Modello):
    """The economic terms of a signed fill, separate from its ledger identity."""

    quantity = InstanceDummy("quantity", real=True)
    price = InstanceDummy("price", positive=True)
    fee = InstanceDummy("fee", nonnegative=True)


class FillPosting(SymbolicModel):
    """Relate a holding before and after an execution, in either solving direction."""

    before = component(HoldingState)
    fill = component(Execution)
    after = component(HoldingState)

    quantity = after.quantity
    quantity = before.quantity + fill.quantity
    cash = after.cash
    cash = before.cash - fill.quantity * fill.price - fill.fee


class AccountState(Modello):
    """A valued account and its participation units at one event boundary."""

    cash = InstanceDummy("cash", real=True)
    nav = InstanceDummy("nav", positive=True)
    units = InstanceDummy("units", positive=True)
    unit_price = nav / units


class CashMovement(Modello):
    """Signed external capital: positive deposits and negative withdrawals."""

    amount = InstanceDummy("amount", real=True)


class CapitalFlow(SymbolicModel):
    """Issue or redeem account units at the pre-flow price, preserving unit NAV."""

    before = component(AccountState)
    flow = component(CashMovement)
    after = component(AccountState)

    issued_units = flow.amount / before.unit_price
    cash = after.cash
    cash = before.cash + flow.amount
    nav = after.nav
    nav = before.nav + flow.amount
    units = after.units
    units = before.units + issued_units

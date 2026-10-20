"""Typed state transitions for externally supplied fills and capital movements."""

from examples.portfolio.models import AccountValue, Trade
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, component


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
    """Signed external capital: positive deposits and negative withdrawals."""

    amount = InstanceDummy("amount", real=True)


class CapitalFlow(AccountState, SymbolicModel):
    """An account state after issuing or redeeming units at the pre-flow price."""

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

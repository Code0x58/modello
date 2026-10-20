"""Typed state transitions for externally supplied account events."""

from examples.portfolio.models import AccountValue, Quote, Trade
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

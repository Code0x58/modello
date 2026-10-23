"""Typed account paths using the same FillPosting as a single execution."""

from __future__ import annotations

from examples.portfolio.models import Execution, FillPosting, HoldingState
from modello import InstanceDummy, Modello, component, model_recurrence, model_vector


class AccountSequence(Modello):
    """N executions connect N + 1 account boundaries in one system of equations."""

    N = InstanceDummy("N", integer=True, nonnegative=True)
    opening = component(HoldingState)
    fills = model_vector(Execution, size=N)
    states = model_recurrence(FillPosting, initial=opening, fill=fills)
    terminal = states[-1]

    fill_quantities = fills.project(lambda fill: fill.quantity)
    prices = fills.project(lambda fill: fill.price)
    fees = fills.project(lambda fill: fill.fee)
    quantities = states.project(lambda account: account.quantity)
    balances = states.project(lambda account: account.cash)
    total_fees = fees.sum()


def main() -> None:
    opening = HoldingState("opening", quantity=3, cash=1000)
    fills = [
        Execution("buy", quantity=2, price=100, fee=1),
        Execution("sell", quantity=-1, fee=1),
    ]
    target = HoldingState("target", cash=908)
    case = AccountSequence("settlement", opening=opening, fills=fills, terminal=target)
    print(f"{case.N} fills; quantities {case.quantities}; cash {case.balances}")
    print(f"Inferred sale price: {case.fills[1].price}")
    assert case.terminal is case.states[-1]
    assert case.states.steps[1].before is case.states[1]
    assert AccountSequence("empty", opening=opening, fills=[]).terminal.cash == 1000


if __name__ == "__main__":
    main()

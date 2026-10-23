"""Account and execution models related directly at successive indices."""

from __future__ import annotations

from examples.portfolio.models import Execution, HoldingState
from modello import Index, InstanceDummy, Modello, ModelObservations, model_recurrence, model_vector


class AccountSequence(Modello):
    """N executions connect N + 1 account boundaries in one system of equations."""

    N = InstanceDummy("N", integer=True, nonnegative=True)
    fills = model_vector(Execution, size=N)
    states = model_vector(HoldingState, size=N + 1)
    n = Index("n")
    law = model_recurrence(
        states[n],
        quantity=states[n - 1].quantity + fills[n - 1].quantity,
        cash=states[n - 1].cash - fills[n - 1].notional - fills[n - 1].fee,
    )
    terminal = states[-1]
    total_fees = fills.project(lambda fill: fill.fee).sum()


def main() -> None:
    opening = HoldingState("opening", quantity=3, cash=1000)
    fills = [
        Execution("buy", quantity=2, price=100, fee=1),
        Execution("sell", quantity=-1, fee=1),
    ]
    target = HoldingState("target", cash=908)
    case = AccountSequence("settlement", states=ModelObservations({0: opening}), fills=fills, terminal=target)
    quantities = case.states.project(lambda state: state.quantity)
    balances = case.states.project(lambda state: state.cash)
    print(f"{case.N} fills; quantities {quantities}; cash {balances}")
    print(f"Inferred sale price: {case.fills[1].price}")
    assert case.terminal is case.states[-1]
    assert AccountSequence("empty", states=ModelObservations({0: opening}), fills=[]).terminal.cash == 1000


if __name__ == "__main__":
    main()

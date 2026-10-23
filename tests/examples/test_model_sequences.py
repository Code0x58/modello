"""The same typed account transitions support symbolic and numerical paths."""

from examples.model_sequences import AccountSequence, main
from examples.portfolio.models import Execution, HoldingState
from examples.symbolic import Known, SymbolicModel


def test_model_sequence_console_walkthrough(capsys):
    main()
    assert "Inferred sale price: 110" in capsys.readouterr().out


def test_numerical_views_preserve_model_types_and_shared_state_objects():
    class NumericalSequence(AccountSequence, SymbolicModel):
        pass

    source = NumericalSequence(
        "source",
        opening=HoldingState("opening", quantity=3, cash=1000),
        fills=[Execution("buy", quantity=2, price=100, fee=1), Execution("sell", quantity=-1, fee=1)],
    )
    view = source.compile()
    result = view.evaluate(Known(source.fills[1].price, 110))
    assert isinstance(result, NumericalSequence)
    assert float(result.terminal.cash) == 908
    assert result.states[-1] is result.terminal
    assert result.states[0] is result.opening
    assert result.states.steps[1].before is result.states[1]
    assert result.states.steps[0].fill is result.fills[0]
    assert result.terminal is not source.terminal
    assert source.fills[1].price.free_symbols

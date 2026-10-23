"""The same typed account transitions support symbolic and numerical paths."""

from examples.model_sequences import AccountSequence, main
from examples.portfolio.models import Execution, HoldingState
from examples.symbolic import Known, SymbolicModel
from modello import ModelObservations


def test_model_sequence_console_walkthrough(capsys):
    main()
    assert "Inferred sale price: 110" in capsys.readouterr().out


def test_numerical_views_preserve_model_types_and_shared_state_objects():
    class NumericalSequence(AccountSequence, SymbolicModel):
        pass

    source = NumericalSequence(
        "source",
        states=ModelObservations({0: HoldingState("opening", quantity=3, cash=1000)}),
        fills=[Execution("buy", quantity=2, price=100, fee=1), Execution("sell", quantity=-1, fee=1)],
    )
    view = source.compile()
    result = view.evaluate(Known(source.fills[1].price, 110))
    assert isinstance(result, NumericalSequence)
    assert float(result.terminal.cash) == 908
    assert result.states[-1] is result.terminal
    assert float(result.states[0].cash) == 1000
    assert not hasattr(result.states, "steps")
    assert result.terminal is not source.terminal
    assert source.fills[1].price.free_symbols

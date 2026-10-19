"""Checks for the streaming example."""

from sympy import Rational

from examples.streaming import DoubleDataEntryFlow, ScalableFlow, SingleDataEntryFlow


def test_simple_system():
    """The wheels on the bus go round and round."""
    channel_input_rates = {"foo": 12, "bar": 3}
    sde = SingleDataEntryFlow(
        "SDE",
        input=sum(channel_input_rates.values()),
        entry_time=5,
        unit_cost=Rational(5, 10000),
        scale=10,
    )
    assert sde.fulfilment == Rational(104, 3285)


def test_scaling_system():
    """Hello from the other side.

    a1→b1→c1→d1
    a2↗

    """
    # imagining the input is in Hz, unit output is in Hz, and unit cost is in $imoleon Hz
    a1 = ScalableFlow("a1", input=100, unit_cost=2, unit_output=100, fulfilment=1)
    a2 = ScalableFlow("a2", input=150, unit_cost=3, unit_output=100, fulfilment=1)
    b1 = ScalableFlow("b1", input=a1.output + a2.output, unit_cost=5, unit_output=100, fulfilment=1)
    c1 = DoubleDataEntryFlow("c1", input=b1.output, unit_cost=7, unit_output=100, fulfilment=1)
    d1 = ScalableFlow("d1", input=c1.output, unit_cost=11, unit_output=100, fulfilment=1)

    verticies = (a1, a2, b1, c1, d1)
    complete_fulfilment_cost = sum(map((lambda v: v.cost), verticies))
    # cost is 1*2 + 2*3 + 3*5 + 3*7 + 3*11 = 77
    # without integer scaling cost is 1*2 + 1.5*3 + 2.5*5 + 2.5*7 + 2.5*11 = 64
    assert complete_fulfilment_cost == 77
    assert a1.utilization == 1
    assert a2.utilization == Rational(3, 4)

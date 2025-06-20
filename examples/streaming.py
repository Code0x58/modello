"""Symbolic models for a system which contains units which have inputs and outputs."""
from modello import InstanceDummy, Modello
from sympy import Rational, ceiling


class ScalableFlow(Modello):
    """
    Something that has input and output, which can be scaled and has associated costs.

    Input and output are assumed to be something like frequency (Hz)

    >>> sde = ScalableFlow("flow", fulfilment=2, scale=1, unit_output=1)
    >>> sde.fulfilment, sde.scale, sde.unit_output
    (2, 1, 1)
    >>> sde.input, sde.output
    (1/2, 1)
    >>> sde.unit_cost, sde.cost
    (_flow_unit_cost, _flow_unit_cost)
    """

    # latency calculations could also be included
    input = InstanceDummy("input", positive=True, rational=True)
    unit_output = InstanceDummy("unit_output", positive=True, rational=True)
    unit_cost = InstanceDummy("unit_cost", positive=True, rational=True)
    scale = InstanceDummy("scale", positive=True, rational=True)
    cost = unit_cost * ceiling(scale)
    output = unit_output * scale
    fulfilment = output / input
    utilization = output / (ceiling(scale) * unit_output)


class SingleDataEntryFlow(ScalableFlow):
    """Flow model for single data entry.

    >>> sde = SingleDataEntryFlow("SDE", scale=10, entry_time=7)
    >>> sde.output
    520/1533
    """

    entry_time = InstanceDummy("entry_time", positive=True, rational=True)
    unit_output = Rational(8 * 260, 24 * 365) / entry_time


class DoubleDataEntryFlow(ScalableFlow):
    """Flow model for double data entry.

    Currently has assumptions like the cost per data entry member is equal
    regardless of serving entry and resolution queues.

    >>> dde = DoubleDataEntryFlow("DDE")
    >>> dde.unit_output
    26/(219*_DDE_entry_time)
    """

    entry_time = InstanceDummy("entry_time", positive=True, rational=True)
    conflict_rate = InstanceDummy("resolution_rate", positive=True, rational=True)
    resolution_time = InstanceDummy("resolution_time", positive=True, rational=True)
    disagreement_rate = InstanceDummy("disagreement_rate", positive=True, rational=True)
    unit_output = Rational(8 * 260, 24 * 365) / entry_time / 2


class EntrySystem(Modello):
    """Simple system demonstrating nesting of flows.

    >>> sys = EntrySystem(
    ...     "SYS",
    ...     flow={"input": 20, "entry_time": 5, "unit_cost": Rational(1, 10), "scale": 2},
    ...     overhead=1,
    ... )
    >>> sys.total_cost
    6/5
    """

    flow = SingleDataEntryFlow
    overhead = InstanceDummy("overhead", positive=True, rational=True)
    total_cost = flow.cost + overhead



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
    b1 = ScalableFlow(
        "b1", input=a1.output + a2.output, unit_cost=5, unit_output=100, fulfilment=1
    )
    c1 = DoubleDataEntryFlow(
        "c1", input=b1.output, unit_cost=7, unit_output=100, fulfilment=1
    )
    d1 = ScalableFlow(
        "d1", input=c1.output, unit_cost=11, unit_output=100, fulfilment=1
    )

    verticies = (a1, a2, b1, c1, d1)
    complete_fulfilment_cost = sum(map((lambda v: v.cost), verticies))
    # cost is 1*2 + 2*3 + 3*5 + 3*7 + 3*11 = 77
    # without integer scaling cost is 1*2 + 1.5*3 + 2.5*5 + 2.5*7 + 2.5*11 = 64
    assert complete_fulfilment_cost == 77
    assert a1.utilization == 1
    assert a2.utilization == Rational(3, 4)

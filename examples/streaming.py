"""Symbolic models for a system which contains units which have inputs and outputs."""
from modello import InstanceDummy, Modello
from sympy import Rational


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
    cost = unit_cost * scale
    output = unit_output * scale
    fulfilment = output / input


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


def test_simple_system():
    """The wheels on the bus go round and round."""
    channel_input_rates = {
        "foo": 12,
        "bar": 3,
    }
    sde = SingleDataEntryFlow(
        "SDE",
        input=sum(channel_input_rates.values()),
        entry_time=5,
        unit_cost=Rational(5, 10000),
        scale=10,
    )
    assert sde.fulfilment == Rational(104, 3285)

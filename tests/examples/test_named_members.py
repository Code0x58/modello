# mypy: warn-unused-ignores
"""Named-member projection and joint solving, independent of finance."""

import pytest

from examples.compute.named_map import NamedMap
from examples.named_members import component_members
from modello import BoundInstanceDummy, InstanceDummy, Modello, component


class Reading(Modello):
    value = InstanceDummy("value", positive=True)


class Inlet(Reading):
    pass


class Outlet(Reading):
    pass


class Readings(Modello):
    inlet = component(Inlet)
    outlet = component(Outlet)


def test_component_view_retains_concrete_instances():
    source = Readings("measured", inlet=Inlet("in", value=3), outlet=Outlet("out", value=7))
    view = component_members(source, Reading)
    assert tuple(view) == ("inlet", "outlet")
    assert view["inlet"] is source.inlet
    assert isinstance(view["inlet"], Inlet)
    assert view.map(lambda reading: reading.value) == {"inlet": 3, "outlet": 7}


def test_component_view_checks_its_common_value_type():
    class Other(Modello):
        pass

    class Mixed(Modello):
        reading = component(Reading)
        other = component(Other)

    with pytest.raises(TypeError, match="other must be a Reading component"):
        component_members(Mixed("mixed"), Reading)
    with pytest.raises(TypeError, match="only model components"):
        component_members(Reading("reading"), Modello)


def test_named_reduction_keeps_joint_inverse_solving_outside_finance():
    class Total(Modello):
        readings = component(Readings)
        total = sum(reading.value for reading in component_members(readings, Reading).values())

    source = Readings("source", inlet=Inlet("in", value=3))
    result = Total("combined", readings=source, total=10)
    assert result.readings.outlet.value == 7
    assert component_members(result.readings, Reading).map(lambda reading: reading.value) == {"inlet": 3, "outlet": 7}
    assert isinstance(source.outlet.value, BoundInstanceDummy)


def _typing_contract(reading: Reading) -> None:
    """Keep projections typed and prevent accidentally introducing Any."""
    members = NamedMap({"reading": reading})
    typed_reading: Reading = members["reading"]
    values: NamedMap[str] = members.map(lambda member: str(member.value))
    numbers: NamedMap[int] = values.map(len)
    labels: NamedMap[str] = values.combine(numbers, lambda text, number: text * number)
    assert typed_reading is reading and labels["reading"]
    _wrong_member: Reading = labels["reading"]  # type: ignore[assignment]
    _wrong_projection: NamedMap[Reading] = values  # type: ignore[assignment]
    members["reading"] = reading  # type: ignore[index]

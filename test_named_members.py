# mypy: warn-unused-ignores
"""Named-domain semantics, independent of the financial example."""

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


def test_membership_is_owned_and_read_only():
    source = {"left": 1, "right": 2}
    members = NamedMap(source)
    source["left"] = 99
    source["new"] = 3
    assert members == {"left": 1, "right": 2}
    assert dict(members.items()) == {"left": 1, "right": 2}
    with pytest.raises(TypeError):
        members["left"] = 10
    with pytest.raises(KeyError):
        members["missing"]


def test_projection_keeps_identity_and_declaration_order():
    members = NamedMap({"second": 2, "first": 1})
    projected = members.map(lambda value: str(value * 10))
    assert tuple(projected) == ("second", "first")
    assert projected == {"second": "20", "first": "10"}


def test_combinations_align_by_name_not_position():
    left = NamedMap({"a": 10, "b": 20})
    right = NamedMap({"b": 2, "a": 1})
    combined = left.combine(right, lambda first, second: first - second)
    assert tuple(combined.items()) == (("a", 9), ("b", 18))
    assert combined == {"b": 18, "a": 9}


@pytest.mark.parametrize("other", [{"a": 1}, {"a": 1, "b": 2, "c": 3}, {"a": 1, "c": 3}])
def test_combinations_require_the_same_domain(other):
    with pytest.raises(ValueError, match="Member names differ"):
        NamedMap({"a": 1, "b": 2}).combine(other, lambda a, b: a + b)


def test_empty_domain_has_an_empty_projection_and_zero_sum():
    empty = NamedMap({})
    assert len(empty) == 0
    assert empty.map(str) == {}
    assert empty.combine({}, lambda a, b: (a, b)) == {}
    assert sum(empty.values()) == 0


@pytest.mark.parametrize("name", ["", 1, None])
def test_member_names_are_nonempty_strings(name):
    with pytest.raises(ValueError, match="nonempty strings"):
        NamedMap([(name, 1)])


def test_duplicate_names_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate member name 'same'"):
        NamedMap([("same", 1), ("same", 2)])


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

# mypy: warn-unused-ignores
"""Model-member projections preserve identity, type, and partial membership."""

from typing import TYPE_CHECKING

import pytest

from examples.compute.named_map import NamedMap
from examples.named_members import MemberValues, component_members
from modello import InstanceDummy, Modello, component


class Reading(Modello):
    value = InstanceDummy("value", real=True)


class Sensors(Modello):
    first = component(Reading)
    second = component(Reading)


def test_projection_selects_real_members_and_keeps_bindings_when_mapped():
    sensors = Sensors("sensors", first=Reading("one", value=2), second=Reading("two", value=3))
    values = MemberValues(component_members(sensors, Reading), NamedMap({"second": 3.0, "first": 2.0}))
    scaled = values.map(lambda value: value * 10)
    assert scaled.for_member(sensors.first) == 20
    assert scaled.for_member(sensors.second) == 30
    assert scaled.members is values.members
    assert tuple(scaled) == (30, 20)
    with pytest.raises(ValueError, match="this model case"):
        scaled.for_member(Sensors("other").first)


def test_partial_and_empty_projections_keep_the_original_member_domain():
    sensors = Sensors("sensors")
    members = component_members(sensors, Reading)
    values = MemberValues(members, NamedMap({"first": "selected"}))
    assert values.for_member(sensors.first) == "selected"
    with pytest.raises(KeyError, match="second"):
        values.for_member(sensors.second)
    empty = MemberValues(members, NamedMap({}))
    assert len(empty) == len(empty.map(str)) == 0
    assert len(empty.members) == 2


def test_projection_rejects_extra_names_and_ambiguous_member_aliases():
    sensors = Sensors("sensors")
    with pytest.raises(ValueError, match="declared members"):
        MemberValues(component_members(sensors, Reading), NamedMap({"unknown": 1}))
    with pytest.raises(ValueError, match="own model instance"):
        MemberValues(NamedMap({"one": sensors.first, "two": sensors.first}), NamedMap({}))


if TYPE_CHECKING:

    def typed_projection(sensors: Sensors, values: MemberValues[Reading, float]) -> None:
        number: float = values.for_member(sensors.first)
        text: MemberValues[Reading, str] = values.map(str)
        assert number and text
        values.for_member(sensors)  # type: ignore[arg-type]
        _wrong: str = values.for_member(sensors.first)  # type: ignore[assignment]
        values["first"]  # type: ignore[index]

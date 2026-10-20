# mypy: warn-unused-ignores
"""Named-member projection and joint solving, independent of finance."""

import pytest

from examples.compute.named_map import NamedMap
from examples.named_members import BoundMembers, ModelFactory, bound_component, component_members, member_bindings
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


def test_component_declaration_can_be_viewed_without_constructing_an_instance():
    declared = component_members(Readings, Reading)
    assert declared["inlet"].value == Readings.inlet.value
    assert declared["outlet"].value == Readings.outlet.value
    assert declared["inlet"].value != Readings("measured").inlet.value


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


class BoundReadings(BoundMembers[str]):
    inlet = bound_component(Reading, "source:in")
    outlet = bound_component(Reading, "source:out")


reading_factory = ModelFactory(Reading, str)


def test_model_factory_declarations_keep_bindings_and_typed_nested_construction():
    class Sensors(BoundMembers[str]):
        inlet = reading_factory.component("source:in")
        outlet = reading_factory.component("source:out")

    sensors = ModelFactory(Sensors, str)
    result = sensors("measured", inlet=reading_factory("in", value=3))
    assert type(result.inlet) is Reading and result.inlet.value == 3
    assert member_bindings(result) == {"inlet": "source:in", "outlet": "source:out"}
    with pytest.raises(TypeError, match="Reading needs a str binding"):
        reading_factory.component(3)
    with pytest.raises(TypeError, match="inlet must be a Reading instance"):
        sensors("invalid", inlet={"value": 3})


def test_bound_members_keep_bindings_with_classes_references_and_solved_instances():
    class Total(Modello):
        readings = component(BoundReadings)
        total = sum(reading.value for reading in component_members(readings, Reading).values())

    source = BoundReadings("source", inlet=Reading("in", value=3))
    result = Total("combined", readings=source, total=10)
    assert type(result.readings.inlet) is type(result.readings.outlet) is Reading
    assert result.readings.outlet.value == 7
    for group in (BoundReadings, Total.readings, result.readings):
        assert member_bindings(group) == {"inlet": "source:in", "outlet": "source:out"}


def test_bound_members_extend_and_replace_components_with_their_bindings():
    class Extended(BoundReadings):
        inlet = bound_component(Inlet, "other:in")
        extra = bound_component(Reading, "source:extra")

    readings = Extended("readings")
    assert (
        tuple(member_bindings(Extended)) == tuple(component_members(readings, Reading)) == ("inlet", "outlet", "extra")
    )
    assert member_bindings(readings) == {"inlet": "other:in", "outlet": "source:out", "extra": "source:extra"}
    assert isinstance(readings.inlet, Inlet)
    assert member_bindings(BoundReadings)["inlet"] == "source:in"


def test_bound_members_cannot_inherit_a_stale_binding_or_omit_one():
    with pytest.raises(ValueError, match="Member names differ"):

        class Missing(BoundMembers[str]):
            reading = component(Reading)

    with pytest.raises(ValueError, match="Member names differ"):

        class UnboundOverride(BoundReadings):
            inlet = component(Reading)

    with pytest.raises((TypeError, RuntimeError)) as error:

        class WrongBase(Modello):
            reading = bound_component(Reading, "source:in")

    cause = error.value.__cause__ or error.value
    assert "BoundMembers model" in str(cause)


def test_bound_members_follow_joint_model_inheritance_precedence():
    class Other(BoundMembers[str]):
        inlet = bound_component(Outlet, "later:in")

    class Combined(BoundReadings, Other):
        pass

    assert isinstance(Combined("readings").inlet, Outlet)
    assert member_bindings(Combined) == {"inlet": "later:in", "outlet": "source:out"}

    class Unbound(Modello):
        inlet = component(Reading)

    with pytest.raises(ValueError, match="Member names differ"):

        class Stale(BoundReadings, Unbound):
            pass


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
    declared: Reading = BoundReadings("readings").inlet
    bindings: NamedMap[str] = member_bindings(BoundReadings)
    assert declared.value.free_symbols and bindings["inlet"]
    _wrong_binding: int = bindings["inlet"]  # type: ignore[assignment]
    instance: Reading = reading_factory("reading", value=3)
    declaration: Reading = reading_factory.component("source:in")
    assert instance.value == 3 and declaration
    _wrong_factory_result: str = reading_factory("reading")  # type: ignore[assignment]
    reading_factory.component(3)  # type: ignore[arg-type]
    reading_factory("reading", value={"amount": 3})  # type: ignore[arg-type]

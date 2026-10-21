# mypy: warn-unused-ignores
"""Typed numerical evaluation, observation bindings, and symbolic plan reuse."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from functools import cached_property
from typing import TYPE_CHECKING

import pytest
from sympy import Symbol

from examples.data import NamedMap, Provider, Series
from examples.symbolic import Known, ModelData, ModelPath, ModelPlan, ModelSeries, ScalarPlan, SymbolicModel
from modello import InstanceDummy, Modello, component, inspect_model


class Reading(Modello):
    value = InstanceDummy("value", positive=True)


class Sensor(SymbolicModel):
    reading = component(Reading)
    scale = InstanceDummy("scale", positive=True)
    adjusted = reading.value * scale
    alias = adjusted

    @cached_property
    def doubled(self):
        return self.adjusted * 2


class Group(Modello):
    sensor = component(Sensor)


def test_evaluation_keeps_component_types_aliases_and_source_case(monkeypatch):
    source = Sensor("sensor", scale=2)
    assert source.doubled.free_symbols  # Prime a cached property before compilation.
    view = ModelPlan(source)

    def fail(*args, **kwargs):
        raise AssertionError("Evaluation must not invoke symbolic solving")

    monkeypatch.setattr("modello.solve", fail)
    result = view.evaluate(Known(source.reading.value, 3))
    assert type(result) is Sensor and type(result.reading) is Reading
    assert float(result.adjusted) == float(result.alias) == 6
    assert float(result.doubled) == 12
    assert result is not source and result.reading is not source.reading
    assert source.reading.value.free_symbols and source.adjusted.free_symbols
    fields = {field.path: field for field in view.walk(result)}
    assert fields[("reading", "value")].role == "input"
    assert fields[("reading", "value")].expression == source.reading.value
    assert fields[("reading", "value")].value == 3
    assert fields[("scale",)].role == "known"
    assert fields[("adjusted",)].role == "derived"
    copied = inspect_model(Group, "copy", sensor=result)
    kinds = {c.path: c.kind for c in copied.constraints if c.input}
    assert kinds[("sensor", "reading", "value")] == "input"
    assert kinds[("sensor", "adjusted")] == kinds[("sensor", "alias")] == "inferred"


def test_columns_preserve_model_references_and_materialize_only_requested_points(monkeypatch):
    source = Sensor("sensor", scale=2)
    view = ModelPlan(source)
    times = (date(2026, 1, 1), date(2026, 1, 2))
    data = Series(times, (3, 4))
    monkeypatch.setattr("modello.solve", lambda *args, **kwargs: pytest.fail("Repeated symbolic solve"))
    path = view.over(times, Known(source.reading.value, data))
    assert path.column(source.adjusted).values == (6, 8)
    assert path.column(source.doubled).values == (12, 16)
    assert isinstance(path.at(1), Sensor)
    assert float(path.at(1).reading.value) == 4
    assert float(path.at(1).doubled) == 16
    with pytest.raises(TypeError, match="not a string"):
        path.column("adjusted")
    with pytest.raises(ValueError, match="unspecified inputs"):
        path.column(Symbol("foreign"))


def test_input_references_are_owned_complete_and_unique():
    source = Sensor("source", scale=2)
    foreign = Sensor("source", scale=2)
    view = ModelPlan(source)
    for knowns in ((), (Known(foreign.reading.value, 3),), (Known(source.reading.value, 3),) * 2):
        with pytest.raises(ValueError):
            view.evaluate(*knowns)
    with pytest.raises(ValueError, match="model case"):
        ModelPlan(source, inputs=(foreign.reading.value,))
    with pytest.raises(ValueError, match="unspecified inputs"):
        ModelPlan(source, inputs=())
    with pytest.raises(ValueError, match="model assumptions"):
        view.evaluate(Known(source.reading.value, 0))
    with pytest.raises(ValueError, match="model assumptions"):
        view.over((date(2026, 1, 1),), Known(source.reading.value, Series((date(2026, 1, 1),), (0,))))


def test_the_compiled_case_is_a_snapshot_and_numeric_cases_can_be_composed():
    source = Sensor("source", scale=2)
    view = ModelPlan(source)
    source.scale = 20
    result = view.evaluate(Known(view.source.reading.value, 3))
    assert float(result.scale) == 2 and float(result.adjusted) == 6
    parent = Sensor("composed", scale=4, reading=result.reading)
    assert float(parent.adjusted) == 12


def test_cached_plans_are_reused_within_a_class_and_separate_for_subclasses():
    class Shifted(Sensor):
        adjusted = Sensor.reading.value * Sensor.scale + 1

    for model_type, expected in ((Sensor, 6), (Shifted, 7)):
        view = model_type.plan()
        model = view.source
        result = view.evaluate(Known(model.reading.value, 3), Known(model.scale, 2))
        assert type(result) is model_type and float(result.adjusted) == expected
        assert model_type.plan() is view
    assert Shifted.plan() is not Sensor.plan()


def test_a_model_without_scalar_fields_still_has_an_observation_axis():
    class Empty(Modello):
        pass

    view = ModelPlan(Empty("empty"))
    assert type(view.evaluate()) is Empty
    path = view.over(())
    with pytest.raises(IndexError):
        path.at(0)


if TYPE_CHECKING:

    def typed_results(source: Sensor, view: ModelPlan[Sensor], times: tuple[date, ...], values: Series[date]) -> None:
        compiled: ModelPlan[Sensor] = source.compile()
        cached: ModelPlan[Sensor] = Sensor.plan()
        result: Sensor = view.evaluate(Known(source.reading.value, 3.0))
        reading: Reading = result.reading
        path: ModelPath[Sensor, date] = view.over(times, Known(source.reading.value, values))
        column: Series[date] = path.column(source.adjusted)
        numeric: Sensor = path.at(0)
        assert reading and column and numeric and compiled and cached
        _wrong: Reading = view.evaluate(Known(source.reading.value, 3.0))  # type: ignore[assignment]
        view.evaluate(Known("reading.value", 3.0))  # type: ignore[arg-type]
        view.evaluate(Known(source.reading.value, "three"))  # type: ignore[arg-type]
        view.over(times, Known(source.reading.value, 3.0))  # type: ignore[arg-type]
        path.column("adjusted")  # type: ignore[arg-type]


@dataclass(frozen=True)
class Measurement:
    value: float

    def knowns(self, model: Sensor) -> tuple[Known[float], ...]:
        return (Known(model.reading.value, self.value),)


@dataclass(frozen=True)
class Measurements:
    times: tuple[date, ...]
    values: Series[date]

    def knowns(self, model: Sensor) -> tuple[Known[Series[date]], ...]:
        return (Known(model.reading.value, self.values),)


@dataclass(frozen=True)
class MeasurementProvider(Provider[Measurement]):
    value: float

    def load(self) -> Measurement:
        return Measurement(self.value)


def test_the_same_observation_binds_to_each_compiled_cases_own_fields(monkeypatch):
    first = Sensor("same", scale=2).compile()
    second = Sensor("same", scale=3).compile()
    observation = MeasurementProvider(4).load()
    monkeypatch.setattr("modello.solve", lambda *args, **kwargs: pytest.fail("Repeated symbolic solve"))
    assert float(first.evaluate(observation).adjusted) == 8
    assert float(second.evaluate(observation).adjusted) == 12
    assert first.source.reading.value != second.source.reading.value
    assert first.source.reading.value.free_symbols


def test_observation_records_and_field_bindings_can_supply_complementary_inputs():
    view = Sensor("scale_and_reading").compile()
    result = view.evaluate(Measurement(4), Known(view.source.scale, 3))
    assert float(result.adjusted) == 12
    with pytest.raises(ValueError, match="every input"):
        view.evaluate(Measurement(4))
    with pytest.raises(ValueError, match="more than once"):
        view.evaluate(Measurement(4), Measurement(5), Known(view.source.scale, 3))
    with pytest.raises(TypeError, match="typed ModelData"):
        view.evaluate({"reading": 4})


def test_structured_columns_supply_their_axis_and_keep_the_concrete_model_type():
    view = Sensor("sensor", scale=2).compile()
    times = (date(2026, 1, 1), date(2026, 1, 2))
    observations = Measurements(times, Series(times, (3, 4)))
    path = view.over(observations)
    assert path.times == times
    assert path.column(view.source.adjusted).values == (6, 8)
    assert type(path.at(1)) is Sensor and float(path.at(1).reading.value) == 4
    assert view.over(times, observations).column(view.source.adjusted) == path.column(view.source.adjusted)
    with pytest.raises(ValueError, match="same observation times"):
        view.over(Measurements(times, Series(times[:1], (3,))))
    with pytest.raises(TypeError, match="observation record or an axis"):
        view.over(observations, Known(view.source.reading.value, observations.values))


@dataclass(frozen=True)
class AdjustedMeasurement:
    value: float

    def knowns(self, model: Sensor) -> tuple[Known[float], ...]:
        return (Known(model.adjusted, self.value),)


def test_inverse_cases_use_the_same_data_protocol():
    source = Sensor("inverse", scale=2, adjusted=Symbol("observation", positive=True))
    result = source.compile().evaluate(AdjustedMeasurement(12))
    assert float(result.reading.value) == 6
    copied = inspect_model(Group, "inverse_copy", sensor=result)
    kinds = {c.path: c.kind for c in copied.constraints if c.input}
    assert kinds[("sensor", "adjusted")] == "input"
    assert kinds[("sensor", "reading", "value")] == "inferred"


if TYPE_CHECKING:

    class Other(SymbolicModel):
        value = InstanceDummy("value", real=True)

    def typed_observations(view: ModelPlan[Sensor], other: ModelPlan[Other], samples: Measurements) -> None:
        observation: ModelData[Sensor, float] = Measurement(3)
        observations: ModelSeries[Sensor, date] = samples
        provider: Provider[Measurement] = MeasurementProvider(3)
        result: Sensor = view.evaluate(observation)
        path: ModelPath[Sensor, date] = view.over(observations)
        assert result and path and provider
        other.evaluate(observation)  # type: ignore[arg-type]
        other.over(observations)  # type: ignore[arg-type]
        view.evaluate(observations)  # type: ignore[arg-type]
        view.over(observation)  # type: ignore[call-overload]


def test_named_plan_agrees_between_points_and_vectors():
    p, q = Symbol("p"), Symbol("q")
    plan = ScalarPlan(NamedMap({"price": p, "quantity": q}), NamedMap({"value": p * q, "ratio": p / q}))
    assert plan.evaluate(NamedMap({"quantity": 2, "price": 10})) == {"value": 20, "ratio": 5}
    times = (date(2026, 1, 1), date(2026, 1, 2))
    result = plan.over(times, NamedMap({"quantity": Series(times, (2, 4)), "price": Series(times, (10, 12))}))
    assert result["value"].values == (20, 48)
    assert result["ratio"].values == (5, 3)
    with pytest.raises(ValueError, match="same observation times"):
        plan.over(times, NamedMap({"quantity": Series(times, (2, 4)), "price": Series(times[:1], (10,))}))
    with pytest.raises(ValueError, match="Member names differ"):
        plan.evaluate(NamedMap({"price": 10}))
    with pytest.raises(ValueError, match="finite"):
        plan.evaluate(NamedMap({"price": float("nan"), "quantity": 2}))


def test_plan_handles_constants_and_rejects_unbound_or_aliased_inputs():
    p, q = Symbol("p"), Symbol("q")
    plan = ScalarPlan(NamedMap({}), NamedMap({"cash": 100}))
    assert plan.evaluate(NamedMap({})) == {"cash": 100}
    assert plan.over((date(2026, 1, 1),), NamedMap({}))["cash"].values == (100,)
    with pytest.raises(ValueError, match="independent symbols"):
        ScalarPlan(NamedMap({"p": p, "q": p}), NamedMap({"value": p}))
    with pytest.raises(ValueError, match="unspecified inputs"):
        ScalarPlan(NamedMap({"p": p}), NamedMap({"value": p + q}))

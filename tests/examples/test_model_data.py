# mypy: warn-unused-ignores
"""Typed observations reuse compiled views without application-specific evaluators."""

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pytest
from sympy import Symbol

from examples.compute.providers import Provider
from examples.compute.series import Series
from examples.model_plan import Known, ModelData, ModelPath, ModelPlan, ModelSeries
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, component


class Reading(Modello):
    value = InstanceDummy("value", positive=True)


class Sensor(SymbolicModel):
    reading = component(Reading)
    scale = InstanceDummy("scale", positive=True)
    adjusted = reading.value * scale


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

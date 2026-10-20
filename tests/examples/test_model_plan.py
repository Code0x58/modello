# mypy: warn-unused-ignores
"""Model-shaped evaluation, field ownership, provenance, and solver-free reuse."""

from datetime import date
from functools import cached_property
from typing import TYPE_CHECKING

import pytest
from sympy import Symbol

from examples.compute.series import Series
from examples.model_plan import Known, ModelPath, ModelPlan
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, component


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

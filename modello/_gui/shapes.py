"""Select an independent finite explorer context for each observation window."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from sympy import sympify

from modello._model import _specialize_model
from modello.errors import ModelInputError
from modello.vectors import _vector_input

if TYPE_CHECKING:
    from modello import Modello, ScalarValue
    from modello._gui.inputs import ExplorerValue


class ShapePlan:
    """Keep the reusable declaration and original observations across proposals."""

    def __init__(self, model: type[Modello], inputs: Mapping[str, ExplorerValue]) -> None:
        self.model = getattr(model, "_modello_shape_source", model)
        self.inputs = dict(inputs)
        self.vectors = frozenset(
            name for name, vector in self.model._modello_namespace.vectors.items() if not vector._finite
        )
        self.collections = frozenset(
            name for name, value in self.model._modello_namespace.collections.items() if not value._finite
        )
        self.axes = self.vectors | self.collections
        axes = {**self.model._modello_namespace.vectors, **self.model._modello_namespace.collections}
        symbols = set().union(*(sympify(value.shape[0]).free_symbols for value in axes.values()))
        self.dimensions = {name for name, symbol in self.model._modello_namespace.dummies.items() if symbol in symbols}
        known_sizes = {
            name: model._modello_namespace.attrs[name]
            for name in self.dimensions
            if model._modello_namespace.attrs[name].is_Integer
        }
        if not (self.axes | self.dimensions).isdisjoint(inputs):
            known_sizes = {}
        self.initial_model = _specialize_model(self.model, {**known_sizes, **inputs})
        self.fallback = {name: self.initial_model._modello_namespace.attrs[name] for name in self.dimensions}

    def select(self, bindings: Mapping[str, object]) -> tuple[type[Modello], dict[str, ExplorerValue]]:
        shapes: dict[str, object] = dict(self.inputs)
        for path, value in bindings.items():
            if path not in self.axes and path not in self.dimensions:
                continue
            shapes.pop(path, None)
            if value is not None:
                shapes[path] = value
        if any(path in bindings and bindings[path] is not None for path in self.axes):
            for path in self.axes - bindings.keys():
                value = shapes.get(path)
                if value is not None and not any(_observed(entry) for entry in _vector_input(value, None, path)):
                    shapes.pop(path)
        if not any(name in shapes for name in self.axes | self.dimensions):
            shapes.update(self.fallback)
        concrete = _specialize_model(self.model, shapes)
        baseline = dict(self.inputs)
        for path in self.axes & baseline.keys():
            value = _vector_input(baseline[path], None, path)
            size = (
                len(concrete._modello_namespace.nested_models[path])
                if path in self.collections
                else len(concrete._modello_namespace.vectors[path])
            )
            if len(value) != size:
                if path not in bindings and any(_observed(entry) for entry in value):
                    raise ModelInputError(f"Release the {path} observations before changing the related vector length")
                baseline[path] = cast("ExplorerValue", [*value[:size], *([None] * max(0, size - len(value)))])
        return concrete, baseline


def _entries(value: object) -> tuple[ScalarValue | None, ...]:
    return cast("tuple[ScalarValue | None, ...]", _vector_input(value, None, "Vector observations"))


def _observed(value: object) -> bool:
    return any(_observed(entry) for entry in value.values()) if isinstance(value, Mapping) else value is not None

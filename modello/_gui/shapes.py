"""Select an independent finite explorer context for each observation window."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

from modello._model import _specialize_model
from modello.errors import ModelInputError
from modello.vectors import Vector

if TYPE_CHECKING:
    from modello import Modello, ScalarValue, VectorValue


class ShapePlan:
    """Keep the reusable declaration and original observations across proposals."""

    def __init__(self, model: type[Modello], inputs: Mapping[str, ScalarValue | VectorValue]) -> None:
        self.model = getattr(model, "_modello_shape_source", model)
        self.inputs = dict(inputs)
        self.vectors = frozenset(
            name for name, vector in self.model._modello_namespace.vectors.items() if not vector._finite
        )
        self.symbols = set().union(*(self.model._modello_namespace.vectors[name].free_symbols for name in self.vectors))
        self.dimensions = {
            name for name, symbol in self.model._modello_namespace.dummies.items() if symbol in self.symbols
        }
        self.fallback = {
            name: model._modello_namespace.attrs[name]
            for name in self.dimensions
            if model._modello_namespace.attrs[name].is_Integer
        }
        concrete, _ = self.select({})
        self.fallback.update({name: concrete._modello_namespace.attrs[name] for name in self.dimensions})

    def select(self, bindings: Mapping[str, object]) -> tuple[type[Modello], dict[str, ScalarValue | VectorValue]]:
        shapes: dict[str, object] = dict(self.inputs)
        for path, value in bindings.items():
            if path in self.vectors or path in self.dimensions:
                if value is None:
                    shapes.pop(path, None)
                else:
                    shapes[path] = value
        if any(path in bindings and bindings[path] is not None for path in self.vectors):
            for path in self.vectors - bindings.keys():
                value = shapes.get(path)
                if value is not None and all(entry is None for entry in _entries(value)):
                    shapes.pop(path)
        if not any(name in shapes for name in self.vectors | self.dimensions):
            shapes.update(self.fallback)
        concrete = _specialize_model(self.model, shapes)
        baseline = dict(self.inputs)
        for path in self.vectors & baseline.keys():
            value = _entries(baseline[path])
            size = len(concrete._modello_namespace.vectors[path])
            if len(value) != size:
                if path not in bindings and any(entry is not None for entry in value):
                    raise ModelInputError(f"Release the {path} observations before changing the related vector length")
                baseline[path] = [*value[:size], *([None] * max(0, size - len(value)))]
        return concrete, baseline


def _entries(value: object) -> tuple[ScalarValue | None, ...]:
    if not isinstance(value, (Vector, Sequence)) or isinstance(value, (str, bytes)):
        raise ModelInputError("Vector observations must be a sequence of scalar entries")
    return cast("tuple[ScalarValue | None, ...]", tuple(value))

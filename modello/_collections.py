"""Deferred ordered component declarations, independent of model construction."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping

from sympy import Basic, Expr, Integer

from modello.errors import ModelInputError
from modello.vectors import Vector


class _UnresolvedSelection(Exception):
    """A collection or index still needs its finite structural parameters."""


@dataclass(frozen=True)
class _Location:
    path: tuple[str, ...]
    index: Expr | None = None
    suffix: tuple[str, ...] = ()

    def substitute(self, bindings: Mapping[Basic, Basic]) -> _Location:
        return replace(self, index=None if self.index is None else self.index.xreplace(bindings))

    def resolve(self, namespace: Any, known: Mapping[Basic, Basic]) -> Any:
        result = namespace.reference_at(self.path)
        if self.index is not None:
            index = self.index.xreplace(known)
            if not isinstance(index, Integer):
                raise _UnresolvedSelection
            result = result[index]
            if isinstance(result, _Selection):
                raise _UnresolvedSelection
        for name in self.suffix:
            result = getattr(result, name)
        return result


@dataclass(frozen=True)
class _Selection:
    model: type[Any]
    location: _Location
    columns: Mapping[tuple[str, ...], Vector] = field(default_factory=dict)

    @property
    def _modello_model(self) -> type[Any]:
        return self.model

    def substitute(self, bindings: Mapping[Basic, Basic]) -> _Selection:
        return replace(
            self,
            location=self.location.substitute(bindings),
            columns={path: column.xreplace(bindings) for path, column in self.columns.items()},
        )

    def __getitem__(self, index: object) -> _Selection:
        size = getattr(self.model, "_modello_collection_size", None)
        if isinstance(index, bool) or not isinstance(index, (int, Integer)) or not isinstance(size, Integer):
            raise ModelInputError("Nested model projections need a concrete integer collection index")
        position = int(index) if index >= 0 else int(size + index)
        if not 0 <= position < size:
            raise IndexError("Model vector index out of range")
        name = str(position)
        child = self.model._modello_namespace.nested_models[name]
        return _Selection(
            child._modello_model,
            replace(self.location, suffix=(*self.location.suffix, name)),
            {path[1:]: value for path, value in self.columns.items() if path[0] == name},
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_modello_"):
            raise AttributeError(name)
        if (name,) in self.columns:
            return self.columns[(name,)][self.location.index]
        child = self.model._modello_namespace.nested_models.get(name)
        if child is not None:
            return _Selection(
                child._modello_model,
                replace(self.location, suffix=(*self.location.suffix, name)),
                {path[1:]: value for path, value in self.columns.items() if path[0] == name},
            )
        vector = self.model._modello_namespace.vectors.get(name)
        if vector is not None:
            return Vector(*(self.columns[(name, str(i))][self.location.index] for i in range(len(vector))))
        raise AttributeError(f"{self.model.__name__} has no projected field {name!r}")


@dataclass(frozen=True)
class _CollectionSpec:
    model: type[Any]
    size: Expr
    factory: Callable[[Expr], type[Any]]
    initial: _Location | None = None
    inputs: Mapping[str, _Location] = field(default_factory=dict)
    indexed: frozenset[str] = frozenset()
    relations: tuple[Basic, ...] = ()
    wire: Callable[[Any, str, Any], None] | None = None

    @property
    def shape(self) -> tuple[int | Expr]:
        return (int(self.size) if isinstance(self.size, Integer) else self.size,)

    @property
    def _finite(self) -> bool:
        return isinstance(self.size, Integer)

    def substitute(self, bindings: Mapping[Basic, Basic]) -> _CollectionSpec:
        return replace(
            self,
            size=self.size.xreplace(bindings),
            initial=None if self.initial is None else self.initial.substitute(bindings),
            inputs={name: value.substitute(bindings) for name, value in self.inputs.items()},
            relations=tuple(value.xreplace(bindings) for value in self.relations),
        )


def _location(value: Any) -> _Location:
    if isinstance(value, _Selection):
        return value.location
    path = getattr(value, "_modello_path", None)
    if path is None:
        raise ModelInputError("Declare the source as a component or model vector in the model class")
    return _Location(path)

"""Typed ordered components and recurrences compiled into the flat model."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast, overload

from sympy import Basic, Eq, Expr, Integer, Symbol, Tuple, sympify

from modello._collections import (
    ModelObservations,
    _Location,
    _ModelVectorSpec,
    _Selection,
    _UnresolvedSelection,
)
from modello._evaluation import _indexed_value, _relative_expression, _substitute
from modello._model import ModelloMeta, ModelloMetaNamespace, _has_symbolic_shape, _known_integers, _ModelReference
from modello._model_laws import ModelRecurrence, _ModelLawSpec
from modello._operators import _definitely_undefined, _scalar_expression
from modello._paths import FieldPath, _path_label, _path_value
from modello._symbols import Index
from modello._values import ScalarValue, SymbolicScalar
from modello.errors import ModelDefinitionError, ModelInputError
from modello.vectors import Vector, _integer_index, _vector_size, _VectorElement, vector

if TYPE_CHECKING:
    from modello import Modello

__all__ = ["ModelVector", "ModelVectorValue", "ModelRecurrence", "model_vector", "model_recurrence"]

_Model = TypeVar("_Model", bound="Modello")


class ModelVector(Sequence[_Model], Generic[_Model]):
    """An immutable ordered group of typed model components.

    Declare it with ``model_vector(Model, size=N)`` in a model class. Supplied
    model instances retain their input provenance; omitted entries stay unknown.
    ``project(lambda item: item.field)`` returns an ordinary scalar Vector.
    """

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    def __len__(self) -> int:
        size = _collection_size(self)
        if not isinstance(size, Integer):
            raise TypeError("The model vector needs a known finite length")
        return int(size)

    @overload
    def __getitem__(self, index: int | SymbolicScalar) -> _Model: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[_Model, ...]: ...

    def __getitem__(self, index: object) -> Any:
        return _collection_item(self, index)

    def __iter__(self) -> Iterator[_Model]:
        return (self[i] for i in range(len(self)))

    def project(self, project: Callable[[_Model], ScalarValue]) -> Vector:
        """Build a scalar vector from expressions on symbolic typed members.

        For example, ``project(lambda fill: fill.quantity * fill.price)``
        combines fields while preserving their model scope and equations.
        """
        return _collection_project(self, project)


ModelVectorValue: TypeAlias = ModelVector[_Model] | Sequence[_Model | None] | ModelObservations[_Model]


class _CollectionMeta(ModelloMeta, ABCMeta):
    pass


def _model_type(value: Any) -> type[Any]:
    return value._modello_model if isinstance(value, _ModelReference) else type(value)


def _collection_size(value: Any) -> Expr:
    model = _model_type(value)
    size = model._modello_collection_size
    return size.xreplace(value._modello_bindings) if isinstance(value, _ModelReference) else size


def _collection_columns(value: Any) -> dict[FieldPath, Vector]:
    model = _model_type(value)
    return {path: getattr(value, name) for path, name in model._modello_collection_columns.items()}


def _collection_item(value: Any, index: object) -> Any:
    if isinstance(index, slice):
        return tuple(_collection_item(value, i) for i in range(len(value))[index])
    point = _integer_index(index, "A model vector index must be an integer")
    size = _collection_size(value)
    if isinstance(size, Integer) and isinstance(point, Integer):
        position = int(point) if point >= 0 else int(size + point)
        if not 0 <= position < size:
            raise IndexError("Model vector index out of range")
        member = getattr(value, str(position))
        return replace(member, _modello_view=True) if isinstance(member, _ModelReference) else member
    return _Selection(
        _model_type(value)._modello_collection_element,
        _Location(getattr(value, "_modello_path", ()), point),
        _collection_columns(value),
    )


def _collection_project(value: Any, project: Callable[[Any], ScalarValue]) -> Vector:
    i = Index("i")
    columns = _collection_columns(value)
    proxy = _Selection(_model_type(value)._modello_collection_element, _Location((), i), columns)
    expression = _scalar_expression(project(proxy), "A model projection must return a scalar expression")
    if isinstance(expression, _VectorElement) and expression.index == i:
        return expression.vector
    return vector(expression, (i, 0, _collection_size(value) - 1))


def _root_model(model: type[Any]) -> type[Any]:
    if not isinstance(model, ModelloMeta):
        raise ModelInputError("A model vector needs a Modello class")
    return next(base for base in reversed(model.__mro__) if isinstance(base, ModelloMeta))


def _metadata(namespace: ModelloMetaNamespace, element: type[Any], size: Expr) -> None:
    for name, value in {
        "_modello_collection_element": element,
        "_modello_collection_size": size,
        "_modello_collection_item": staticmethod(_collection_item),
        "_modello_collection_project": staticmethod(_collection_project),
    }.items():
        dict.__setitem__(namespace, name, value)


@lru_cache(maxsize=128)
def _vector_model(model: type[Modello], size: Expr) -> type[Any]:
    root = _root_model(model)
    if _has_symbolic_shape(model):
        raise ModelInputError("Model vector elements need a concrete scalar/component schema")
    bases = (root, ModelVector)
    namespace = ModelloMetaNamespace(f"{model.__name__}Vector", bases)
    unique = dict.fromkeys(model._modello_definition.fields.values())
    names = {}
    for i, symbol in enumerate(unique):
        name = f"column_{i}"
        namespace[name] = Vector(symbol.name, size=size, **symbol.assumptions0)
        names[symbol] = name
    columns = {path: names[symbol] for path, symbol in model._modello_definition.fields.items()}
    if isinstance(size, Integer):
        replacements = {}
        for i in range(size):
            namespace[str(i)] = model
            member = namespace[str(i)]
            replacements.update(
                {symbol: namespace[columns[path]][i] for path, symbol in member._modello_fields.items()}
            )
        namespace._replace(replacements)
    _metadata(namespace, model, size)
    dict.__setitem__(namespace, "_modello_collection_columns", columns)
    dict.__setitem__(namespace, "_modello_internal_vectors", frozenset(names.values()))
    return _CollectionMeta(f"{model.__name__}Vector", bases, namespace)


def model_vector(model: type[_Model], *, size: int | SymbolicScalar) -> ModelVector[_Model]:
    """Declare a homogeneous vector of independently scoped model components."""
    _root_model(model)
    return cast(ModelVector[_Model], _ModelVectorSpec(model, _vector_size(size), partial(_vector_model, model)))


def model_recurrence(target: _Model, *, start: int = 1, **updates: ScalarValue) -> ModelRecurrence:
    """Declare joint field equations over a finite model-vector window.

    For example, ``model_recurrence(states[n], cash=states[n-1].cash - fees[n-1])``.
    Members before ``start`` remain available for boundary observations. Equations
    may reference current, earlier, or later members within the window.
    All quantities participate in the same joint solve as every other field.
    """
    if not isinstance(target, _Selection):
        raise ModelInputError("Select states[n] from a declared model_vector for relative field equations")
    index = target.location.index
    if not isinstance(index, Index) or not target.location.path:
        raise ModelInputError("An indexed model recurrence targets a declared model vector at a local Index")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ModelInputError("Recurrence history length must be a nonnegative integer")
    if not updates:
        raise ModelInputError("Declare at least one recurrent model field")
    fields = []
    for name, value in updates.items():
        path = next((path for path in target.columns if _path_label(path) == name), None)
        if path is None:
            raise ModelInputError(f"{target.model.__name__} has no scalar field {name!r}")
        expression = _scalar_expression(value, "Recurrent field definitions must be scalar expressions")
        fields.append((path, expression))
    equations = Tuple(*(Eq(target.columns[path][index], expression, evaluate=False) for path, expression in fields))
    size = next(iter(target.columns.values())).shape[0]
    template = ModelRecurrence(equations, Tuple(index, Integer(start), sympify(size) - 1))
    spec = _ModelLawSpec(target.location, target.columns, template, tuple(fields), _compile_model_law)
    return cast(ModelRecurrence, spec)


def _compile_model_law(
    namespace: ModelloMetaNamespace,
    spec: _ModelLawSpec,
) -> tuple[tuple[FieldPath, Symbol, Basic], ...]:
    try:
        collection = namespace.reference_at(spec.target.path)
        size = _collection_size(collection)
    except (KeyError, AttributeError) as error:
        raise ModelDefinitionError("A model recurrence must target its own declared model vector") from error
    if not isinstance(size, Integer):
        raise _UnresolvedSelection
    columns = _collection_columns(collection)
    if spec.target.suffix:
        columns = {
            path[len(spec.target.suffix) :]: value
            for path, value in columns.items()
            if path[: len(spec.target.suffix)] == spec.target.suffix
        }
    if columns != spec.columns:
        raise ModelDefinitionError("A model recurrence must target its own declared model vector")
    law = spec.template
    start = int(law.start)
    if size < start:
        raise ModelInputError("The model vector must contain the recurrence's initial history")
    known = _known_integers(namespace)
    stop = law.stop.xreplace(known)
    if isinstance(stop, Integer) and stop != size - 1:
        raise ModelDefinitionError("The recurrence's domain must match its declared model-vector window")
    definitions = []
    for path, expression in spec.updates:
        body = _relative_expression(_substitute(expression, known), law.index)
        for point in range(start, int(size)):
            value = _indexed_value(body, law.index, Integer(point))
            if _definitely_undefined(value):
                raise ModelInputError("A recurrent field expression is undefined in its declared vector window")
            member = collection[point]
            target = _path_value(member, (*spec.target.suffix, *path))
            definitions.append(((*spec.target.path, str(point), *spec.target.suffix, *path), target, value))
    return tuple(definitions)

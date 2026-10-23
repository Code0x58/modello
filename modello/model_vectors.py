"""Typed ordered components and recurrences compiled into the flat model."""

from __future__ import annotations

from abc import ABCMeta
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar, cast, overload

from sympy import Basic, Dummy, Eq, Expr, Integer, Piecewise, S, Symbol, SympifyError, Tuple, simplify, sympify

from modello._collections import (
    ModelObservations,
    _CollectionSpec,
    _Location,
    _location,
    _Selection,
    _UnresolvedSelection,
)
from modello._expressions import _substitute
from modello._model import ModelloMeta, ModelloMetaNamespace, _has_symbolic_shape, _known_integers, _ModelReference
from modello._model_laws import ModelRecurrence, _ModelLawSpec
from modello._paths import _path_label, _path_value
from modello._symbols import Index
from modello._values import ScalarValue, SymbolicScalar
from modello.errors import ModelDefinitionError, ModelInputError
from modello.vectors import Vector, _invalid_index, _VectorElement, vector

if TYPE_CHECKING:
    from modello import Modello

__all__ = ["ModelVector", "ModelVectorValue", "ModelSequence", "ModelRecurrence", "model_vector", "model_recurrence"]

_Model = TypeVar("_Model", bound="Modello")
_Step = TypeVar("_Step", bound="Modello")


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
        """Project typed member expressions without losing their shared scope."""
        return _collection_project(self, project)


class ModelSequence(ModelVector[_Model], Generic[_Model, _Step]):
    """Opening state followed by the result of each typed transition."""

    initial: _Model
    steps: ModelVector[_Step]


ModelVectorValue: TypeAlias = ModelVector[_Model] | Sequence[_Model | None] | ModelObservations[_Model]


class _CollectionMeta(ModelloMeta, ABCMeta):
    pass


def _model_type(value: Any) -> type[Any]:
    return value._modello_model if isinstance(value, _ModelReference) else type(value)


def _collection_size(value: Any) -> Expr:
    model = _model_type(value)
    size = model._modello_collection_size
    return size.xreplace(value._modello_bindings) if isinstance(value, _ModelReference) else size


def _collection_columns(value: Any) -> dict[tuple[str, ...], Vector]:
    model = _model_type(value)
    if model._modello_collection_kind == "vector":
        return {path: getattr(value, name) for path, name in model._modello_collection_columns.items()}
    count = _collection_size(value)
    tail = _collection_columns(value.steps)
    i = Index("n")
    result = {}
    for path in model._modello_collection_element._modello_definition.fields:
        first = _path_value(value.initial, path)
        following = tail[(model._modello_result,) + path]
        result[path] = vector(Piecewise((first, Eq(i, 0)), (following[i - 1], True)), (i, 0, count - 1))
    return result


def _collection_item(value: Any, index: object) -> Any:
    if isinstance(index, slice):
        return tuple(_collection_item(value, i) for i in range(len(value))[index])
    if isinstance(index, bool):
        raise ModelInputError("A model vector index must be an integer")
    try:
        point = sympify(index)
    except (TypeError, ValueError, SympifyError) as error:
        raise ModelInputError("A model vector index must be an integer") from error
    if not isinstance(point, Expr) or _invalid_index(point):
        raise ModelInputError("A model vector index must be an integer")
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
    expression = sympify(project(proxy))
    if isinstance(expression, _VectorElement) and expression.index == i:
        return expression.vector
    return vector(expression, (i, 0, _collection_size(value) - 1))


def _size(value: int | SymbolicScalar) -> Expr:
    try:
        result = sympify(value)
    except (TypeError, ValueError, SympifyError) as error:
        raise ModelInputError("A model vector size must be a nonnegative integer expression") from error
    if (
        isinstance(value, bool)
        or not isinstance(result, Expr)
        or result.is_integer is not True
        or result.is_nonnegative is False
    ):
        raise ModelInputError("A model vector size must be a nonnegative integer expression")
    return result


def _root_model(model: type[Any]) -> type[Any]:
    if not isinstance(model, ModelloMeta):
        raise ModelInputError("A model vector needs a Modello class")
    return next(base for base in reversed(model.__mro__) if isinstance(base, ModelloMeta))


def _metadata(namespace: ModelloMetaNamespace, element: type[Any], size: Expr, kind: str) -> None:
    for name, value in {
        "_modello_collection_element": element,
        "_modello_collection_size": size,
        "_modello_collection_kind": kind,
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
    _metadata(namespace, model, size, "vector")
    dict.__setitem__(namespace, "_modello_collection_columns", columns)
    dict.__setitem__(namespace, "_modello_internal_vectors", frozenset(names.values()))
    return _CollectionMeta(f"{model.__name__}Vector", bases, namespace)


def model_vector(model: type[_Model], *, size: int | SymbolicScalar) -> ModelVector[_Model]:
    """Declare a homogeneous vector of independently scoped model components."""
    _root_model(model)
    return cast(ModelVector[_Model], _CollectionSpec(model, _size(size), partial(_vector_model, model)))


def _connect(namespace: ModelloMetaNamespace, target: _ModelReference, source: _ModelReference) -> None:
    if target._modello_model is not source._modello_model:
        raise ModelDefinitionError("Recurrence ports must declare the same model schema")
    replacements = {}
    for path, symbol in target._modello_fields.items():
        other = source._modello_fields[path]
        if symbol.assumptions0 != other.assumptions0:
            raise ModelDefinitionError("Component wiring cannot discard field assumptions")
        replacements[symbol] = other
    for symbol in target._modello_model._modello_definition.template_symbols:
        replacements[target._modello_bindings[symbol]] = source._modello_bindings[symbol]
    namespace._replace(replacements)
    namespace.aliases[target._modello_path] = source._modello_path


@lru_cache(maxsize=128)
def _sequence_model(
    transition: type[Modello], state: type[Modello], previous: str, result: str, size: Expr
) -> type[Any]:
    count = size - 1
    bases = (_root_model(transition), ModelSequence)
    namespace = ModelloMetaNamespace(f"{transition.__name__}Sequence", bases)
    namespace["initial"] = state
    namespace["steps"] = _vector_model(transition, count)
    if isinstance(count, Integer):
        for i in range(count):
            step = namespace["steps"][i]
            before = namespace["initial"] if i == 0 else getattr(namespace["steps"][i - 1], result)
            _connect(namespace, getattr(step, previous), before)
        namespace["0"] = replace(namespace["initial"], _modello_view=True)
        for i in range(count):
            namespace[str(i + 1)] = replace(getattr(namespace["steps"][i], result), _modello_view=True)
    _metadata(namespace, state, size, "sequence")
    dict.__setitem__(namespace, "_modello_previous", previous)
    dict.__setitem__(namespace, "_modello_result", result)
    dict.__setitem__(namespace, "_modello_transition", transition)
    return _CollectionMeta(f"{transition.__name__}Sequence", bases, namespace)


def _transition_recurrence(
    transition: type[_Step],
    *,
    initial: _Model,
    size: int | SymbolicScalar | None = None,
    previous: str = "before",
    result: str = "after",
    **inputs: ModelVector[Any] | Modello,
) -> ModelSequence[_Model, _Step]:
    """Repeat a transition, sharing its previous/result ports with adjacent states.

    The state ports must use the same declared schema. Model-vector inputs are
    consumed by index; ordinary component inputs are shared across steps.
    """
    _root_model(transition)
    ports = transition._modello_namespace.nested_models
    if previous == result or {previous, result} & inputs.keys():
        raise ModelInputError("Previous and result ports must be distinct and cannot also be recurrence inputs")
    if (
        previous not in ports
        or result not in ports
        or ports[previous]._modello_model is not ports[result]._modello_model
    ):
        raise ModelInputError("A model recurrence needs matching previous/result component schemas")
    state = ports[previous]._modello_model
    if getattr(initial, "_modello_model", None) is not state:
        raise ModelInputError("Declare the initial state as a component of the recurrence's state schema")
    sources = {}
    indexed = set()
    relations = []
    count = None if size is None else _size(size)
    for name, source in inputs.items():
        if name not in ports:
            raise ModelInputError(f"{transition.__name__} has no input component {name!r}")
        source_model = getattr(source, "_modello_model", None)
        member = getattr(source_model, "_modello_collection_element", source_model)
        if member is not ports[name]._modello_model:
            raise ModelInputError(f"Input {name!r} needs the declared component schema")
        sources[name] = _location(source)
        if hasattr(source_model, "_modello_collection_size"):
            indexed.add(name)
            length = _collection_size(source)
            count = length if count is None else count
            relation = Eq(count, length)
            if relation is S.false:
                raise ModelInputError("The step count and model-vector input lengths conflict")
            relations.append(relation)
    if count is None:
        raise ModelInputError("Supply a step count or a model-vector input")
    spec = _CollectionSpec(
        state,
        count + 1,
        partial(_sequence_model, transition, state, previous, result),
        _location(initial),
        sources,
        frozenset(indexed),
        tuple(relations),
        _wire_collection,
    )
    return cast(ModelSequence[_Model, _Step], spec)


@overload
def model_recurrence(
    transition: type[_Step],
    *,
    initial: _Model,
    size: int | SymbolicScalar | None = None,
    previous: str = "before",
    result: str = "after",
    **inputs: ModelVector[Any] | Modello,
) -> ModelSequence[_Model, _Step]: ...


@overload
def model_recurrence(transition: _Model, *, start: int = 1, **updates: ScalarValue) -> ModelRecurrence: ...


def model_recurrence(transition: Any, **arguments: Any) -> Any:
    """Declare relative field equations over indexed models, or repeat a transition.

    ``model_recurrence(states[n], cash=states[n-1].cash - fills[n-1].amount)``
    constrains the existing typed members. Members before ``start`` supply the
    history and may be observed or inferred through the same joint equations.
    """
    if isinstance(transition, _Selection):
        return _indexed_recurrence(transition, arguments.pop("start", 1), arguments)
    if "initial" not in arguments:
        raise ModelInputError("Select states[n] for field equations, or supply a transition's initial component")
    return _transition_recurrence(transition, **arguments)


def _indexed_recurrence(target: _Selection, start: int, updates: Mapping[str, ScalarValue]) -> ModelRecurrence:
    index = target.location.index
    if not isinstance(index, Index) or not target.location.path:
        raise ModelInputError("An indexed model recurrence targets a declared model vector at a local Index")
    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ModelInputError("Recurrence history length must be a nonnegative integer")
    if not updates:
        raise ModelInputError("Declare at least one recurrent model field")
    equations = []
    paths = []
    for name, value in updates.items():
        path = next((path for path in target.columns if _path_label(path) == name), None)
        if path is None:
            raise ModelInputError(f"{target.model.__name__} has no scalar field {name!r}")
        try:
            expression = sympify(value)
        except (TypeError, ValueError, SympifyError) as error:
            raise ModelInputError("Recurrent field definitions must be scalar expressions") from error
        if not isinstance(expression, Expr) or expression.is_Matrix:
            raise ModelInputError("Recurrent field definitions must be scalar expressions")
        _validate_model_rule(expression, index, start, set(target.columns.values()))
        equations.append(Eq(target.columns[path][index], expression, evaluate=False))
        paths.append(path)
    size = next(iter(target.columns.values())).shape[0]
    template = ModelRecurrence(Tuple(*equations), Tuple(index, Integer(start), sympify(size) - 1))
    spec = _ModelLawSpec(target.location, target.columns, tuple(paths), template, tuple(equations), _compile_model_law)
    return cast(ModelRecurrence, spec)


def _compile_model_law(
    namespace: ModelloMetaNamespace,
    name: str,
    spec: _ModelLawSpec,
) -> tuple[tuple[tuple[str, ...], Symbol, Basic], ...]:
    try:
        collection = namespace.reference_at(spec.target.path)
        size = _collection_size(collection)
    except (KeyError, AttributeError) as error:
        raise ModelDefinitionError("A model recurrence must target its own declared model vector") from error
    if getattr(collection._modello_model, "_modello_collection_kind", None) != "vector":
        raise ModelDefinitionError("Declare the indexed recurrence's state window with model_vector")
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
    targets = set(columns.values())
    known = _known_integers(namespace)
    stop = law.stop.xreplace(known)
    if isinstance(stop, Integer) and stop != size - 1:
        raise ModelDefinitionError("The recurrence's domain must match its declared model-vector window")
    definitions = []
    for path, equation in zip(spec.paths, spec.equations):
        expression = equation.rhs
        terms = _validate_model_rule(expression, law.index, start, targets)
        for point in range(start, int(size)):
            bindings = {**known, law.index: Integer(point)}
            for term in terms:
                if law.index in term.index.free_symbols:
                    position = term.index.xreplace(bindings)
                    count = sympify(term.vector.shape[0]).xreplace(known)
                    if not isinstance(count, Integer):
                        raise ModelDefinitionError("Recurrent inputs must use a finite declared vector window")
                    if not isinstance(position, Integer) or not 0 <= position < count:
                        raise ModelInputError("A recurrent input index falls outside its observed vector window")
            member = collection[point]
            target = _path_value(member, (*spec.target.suffix, *path))
            definitions.append(
                ((*spec.target.path, str(point), *spec.target.suffix, *path), target, _substitute(expression, bindings))
            )
    return tuple(definitions)


def _validate_model_rule(expression: Expr, index: Symbol, start: int, targets: set[Vector]) -> set[_VectorElement]:
    if any(index in getattr(part, "bound_symbols", ()) for part in expression.atoms(Basic)):
        raise ModelDefinitionError("Use a distinct local index inside a recurrence's nested operators")
    terms = expression.atoms(_VectorElement)
    for term in terms:
        if term.vector in targets:
            lag = simplify(index - term.index)
            history = isinstance(term.index, Integer) and 0 <= term.index < start
            if not history and (not isinstance(lag, Integer) or not 1 <= lag <= start):
                raise ModelDefinitionError("A causal model recurrence uses fixed positive lags covered by its history")
    unindexed = expression.xreplace({term: Dummy() for term in terms}).free_symbols
    if any(symbol in unindexed for column in targets if column._finite for symbol in column[start:]):
        raise ModelDefinitionError("Reference evolving states with a relative recurrence index")
    return terms


def _wire_collection(namespace: ModelloMetaNamespace, name: str, spec: _CollectionSpec) -> None:
    if spec.initial is None or not spec._finite:
        return
    sequence = namespace.nested_models[name]
    known = _known_integers(namespace)
    initial = spec.initial.resolve(namespace, known)
    sources = {port: location.resolve(namespace, known) for port, location in spec.inputs.items()}
    _connect(namespace, sequence.initial, initial)
    for port, source in sources.items():
        for i in range(len(sequence.steps)):
            target = getattr(namespace.nested_models[name].steps[i], port)
            _connect(namespace, target, source[i] if port in spec.indexed else source)

"""Private model declarations, field binding, and instance population."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, TypeVar, cast

from sympy import Basic, Expr, Integer, Symbol, sympify

from modello._collections import _Location, _ModelVectorSpec, _Selection, _UnresolvedSelection
from modello._expressions import _declaration, _DefinitionResolver, _substitute
from modello._model_laws import ModelRecurrence, _ModelLawSpec
from modello._operators import _FunctionSymbol, _SymbolicTemplate
from modello._paths import _path_label, _path_value
from modello._problem import ConstraintKind, FieldPath, _algebra
from modello._shapes import _dimensions
from modello._symbols import BoundInstanceDummy as BoundInstanceDummy, Coordinate, InstanceDummy as InstanceDummy
from modello._values import (
    ScalarValue as ScalarValue,
    SymbolicCondition as SymbolicCondition,
    SymbolicScalar as SymbolicScalar,
)
from modello.annotations import Annotation, _Annotated, _freeze
from modello.errors import ModelDefinitionError, ModelInputError
from modello.series import _project_series
from modello.vectors import Vector, VectorValue as VectorValue, _VectorSymbol

if TYPE_CHECKING:
    from modello import Modello
    from modello.diagnostics import ModelProblem


@dataclass(frozen=True, slots=True)
class _BoundField:
    """Keep a quantity's identity, construction value, and provenance together."""

    symbol: BoundInstanceDummy
    value: Basic
    kind: ConstraintKind

    def copied_kind(self, value: object) -> ConstraintKind:
        """An overwritten construction result becomes a supplied input."""
        return self.kind if isinstance(value, Basic) and value == self.value else ConstraintKind.INPUT


@dataclass(frozen=True, slots=True)
class _Definition:
    """Keep a defining equation with the path that explains where it came from."""

    path: FieldPath
    left: InstanceDummy
    right: Basic


@dataclass(frozen=True)
class _ModelDefinition:
    """A compiled declaration: all field paths and their scoped equations."""

    fields: Mapping[FieldPath, InstanceDummy]
    constraints: tuple[_Definition, ...]
    vectors: Mapping[FieldPath, Vector]
    vector_definitions: Mapping[FieldPath, Vector]
    templates: Mapping[FieldPath, _SymbolicTemplate]
    aliases: Mapping[FieldPath, FieldPath] = field(default_factory=dict)

    @property
    def template_symbols(self) -> frozenset[_FunctionSymbol]:
        return frozenset().union(
            *(
                value.atoms(_FunctionSymbol)
                for value in (*self.templates.values(), *self.vectors.values(), *(c.right for c in self.constraints))
            )
        )


@dataclass(frozen=True)
class _ModelReference:
    """Expose a child's complete field tree during class definition."""

    _modello_model: type[Modello]
    _modello_fields: dict[FieldPath, InstanceDummy]
    _modello_bindings: dict[Basic, Basic] = field(default_factory=dict)
    _modello_path: FieldPath = ()
    _modello_view: bool = False

    @classmethod
    def _modello_create(cls, model: type[Modello], name: str) -> _ModelReference:
        """Scope child symbols independently while preserving aliases."""
        replacements = {
            d: InstanceDummy(f"{name}_{d.name}", **d.assumptions0)
            for d in dict.fromkeys(model._modello_definition.fields.values())
        }
        replacements.update({symbol: symbol.bound(name) for symbol in model._modello_definition.template_symbols})
        axes = list(model._modello_definition.vectors.values())
        size = getattr(model, "_modello_collection_size", None)
        external = set().union(*(value.free_symbols for value in axes))
        if size is not None:
            external.update(size.free_symbols)
        replacements.update({symbol: symbol for symbol in external - replacements.keys()})
        return cls(
            model,
            {path: replacements[d] for path, d in model._modello_definition.fields.items()},
            replacements,
            (name,),
        )

    @cached_property
    def _modello_child_fields(self) -> dict[str, dict[FieldPath, InstanceDummy]]:
        result: dict[str, dict[FieldPath, InstanceDummy]] = {}
        for path, symbol in self._modello_fields.items():
            if len(path) > 1:
                result.setdefault(path[0], {})[path[1:]] = symbol
        return result

    def __len__(self) -> int:
        size = getattr(self._modello_model, "_modello_collection_size", None)
        if not isinstance(size, Integer):
            raise TypeError("The model vector needs a known finite length")
        return int(size)

    def __bool__(self) -> bool:
        return True if not hasattr(self._modello_model, "_modello_collection_size") else bool(len(self))

    def __iter__(self) -> Any:
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index: object) -> Any:
        select = getattr(self._modello_model, "_modello_collection_item", None)
        if select is None:
            raise TypeError("Only a model vector supports indexing")
        return select(self, index)

    @property
    def _modello_templates(self) -> dict[FieldPath, _SymbolicTemplate]:
        return {
            path: value.xreplace(self._modello_bindings)
            for path, value in self._modello_model._modello_definition.templates.items()
        }

    def __getattr__(self, name: str) -> Any:
        """Resolve a scalar or a reference to a deeper child."""
        if (name,) in self._modello_fields:
            return self._modello_fields[(name,)]
        vector = self._modello_model._modello_namespace.vectors.get(name)
        if vector is not None:
            if not vector._finite:
                return vector.xreplace(self._modello_bindings)
            return Vector(*(self._modello_fields[(name, str(i))] for i in range(len(vector))))
        if (name,) in self._modello_templates:
            return self._modello_templates[(name,)]
        if name == "project" and hasattr(self._modello_model, "_modello_collection_project"):
            return lambda project: self._modello_model._modello_collection_project(self, project)
        if name == "shape" and hasattr(self._modello_model, "_modello_collection_size"):
            size = self._modello_model._modello_collection_size.xreplace(self._modello_bindings)
            return (int(size) if isinstance(size, Integer) else size,)
        pending = self._modello_model._modello_namespace.views.get(name)
        if pending is not None and name not in self._modello_model._modello_namespace.nested_models:
            owner = self
            for part in pending.location.path:
                owner = getattr(owner, part)
            if pending.location.index is not None:
                owner = owner[pending.location.index.xreplace(self._modello_bindings)]
            for part in pending.location.suffix:
                owner = getattr(owner, part)
            return owner
        child = self._modello_model._modello_namespace.nested_models.get(name)
        if child is None:
            raise AttributeError(f"{self._modello_model.__name__} has no model attribute {name!r}")
        bindings = {
            original: self._modello_bindings.get(scoped, scoped) for original, scoped in child._modello_bindings.items()
        }
        external = set().union(
            *(value.free_symbols for value in child._modello_model._modello_definition.vectors.values())
        )
        external.difference_update(child._modello_model._modello_definition.fields.values())
        bindings.update(
            {symbol: self._modello_bindings[symbol] for symbol in external if symbol in self._modello_bindings}
        )
        return _ModelReference(
            child._modello_model,
            self._modello_child_fields.get(name, {}),
            bindings,
            (*self._modello_path, name),
        )


class ModelloMetaNamespace(dict):
    """Track scalar expressions and named children in a model class body."""

    def __init__(self, name: str, bases: tuple[type, ...]) -> None:
        """Inherit model fields without copying ordinary methods or class cells."""
        super().__init__()
        self.name = name
        self.attrs: dict[str, Basic] = {}
        self.dummies: dict[str, InstanceDummy] = {}
        self.vectors: dict[str, Vector] = {}
        self.vector_attrs: dict[str, Vector] = {}
        self.templates: dict[str, _SymbolicTemplate] = {}
        self.nested_models: dict[str, _ModelReference] = {}
        self.collections: dict[str, _ModelVectorSpec] = {}
        self.views: dict[str, _Selection] = {}
        self.aliases: dict[FieldPath, FieldPath] = {}
        self.structural_indices: dict[Symbol, Basic] = {}
        self.pending_laws: set[str] = set()
        self.model_laws: dict[str, _ModelLawSpec] = {}
        self.annotations: dict[type[Annotation], dict[str, Annotation]] = {}
        self._declared_annotations: set[tuple[type[Annotation], str]] = set()
        self._replacements: dict[Basic, Basic] = {}
        self._missing: dict[Basic, str] = {}
        # Preserve the existing precedence: later model bases override earlier ones.
        for base in bases:
            parent = getattr(base, "_modello_namespace", None)
            if not isinstance(parent, ModelloMetaNamespace):
                continue
            for key in self.collections.keys() | self.views.keys() | parent.collections.keys() | parent.views.keys():
                own, inherited = self._field_kind(key), parent._field_kind(key)
                if own is not None and inherited is not None and own != inherited:
                    raise ModelDefinitionError(f"Cannot merge incompatible field kinds for {key}")
                if key in self.views and key in parent.views and self.views[key].location != parent.views[key].location:
                    raise ModelDefinitionError(f"Cannot merge different component views for {key}")
                if (
                    key in self.collections
                    and key in parent.collections
                    and self.collections[key].model is not parent.collections[key].model
                ):
                    raise ModelDefinitionError(f"Cannot merge different collection schemas for {key}")
            for kind, fields in parent.annotations.items():
                self.annotations.setdefault(kind, {}).update(fields)
            conflicts = (
                self.dummies.keys() & (parent.nested_models.keys() | parent.vectors.keys())
                | self.vectors.keys() & (parent.nested_models.keys() | parent.dummies.keys())
                | self.nested_models.keys() & (parent.dummies.keys() | parent.vectors.keys())
                | self.templates.keys() & (parent.dummies.keys() | parent.vectors.keys() | parent.nested_models.keys())
                | parent.templates.keys() & (self.dummies.keys() | self.vectors.keys() | self.nested_models.keys())
                | self.views.keys()
                & (parent.dummies.keys() | parent.vectors.keys() | parent.templates.keys() | parent.collections.keys())
                | parent.views.keys()
                & (self.dummies.keys() | self.vectors.keys() | self.templates.keys() | self.collections.keys())
            )
            if conflicts:
                kinds = (
                    "symbolic law and quantity"
                    if conflicts & (self.templates.keys() | parent.templates.keys())
                    else "vector and scalar or nested model"
                    if conflicts & (self.vectors.keys() | parent.vectors.keys())
                    else "scalar and nested model"
                )
                raise ModelDefinitionError(f"Cannot merge {kinds} fields: {', '.join(sorted(conflicts))}")
            replacements = {self.dummies[key]: dummy for key, dummy in parent.dummies.items() if key in self.dummies}
            for key in self.templates.keys() & parent.templates.keys():
                if self.templates[key]._binding_signature != parent.templates[key]._binding_signature:
                    raise ModelDefinitionError(f"Cannot merge incompatible symbolic declarations: {key}")
            replacements.update(
                {self.templates[key]: value for key, value in parent.templates.items() if key in self.templates}
            )
            for key, vector in parent.vectors.items():
                if key in self.vectors:
                    self._require_shape(key, self.vectors[key], vector)
                    if vector._finite:
                        replacements.update(zip(self.vectors[key], vector))
                    else:
                        replacements[self.vectors[key]] = vector
            for key, child in parent.nested_models.items():
                replacements.update(self._child_replacements(key, child))
            self.attrs.update(parent.attrs)
            self.dummies.update(parent.dummies)
            self.vectors.update(parent.vectors)
            self.vector_attrs.update(parent.vector_attrs)
            self.templates.update(parent.templates)
            self.nested_models.update(parent.nested_models)
            self.collections.update(parent.collections)
            self.views.update(parent.views)
            self.aliases.update(parent.aliases)
            self.structural_indices.update(parent.structural_indices)
            self.model_laws.update(parent.model_laws)
            self._replacements.update(parent._replacements)
            self._missing.update(parent._missing)
            self._replace(replacements)

    def _field_kind(self, name: str) -> str | None:
        for kind in ("collections", "views", "dummies", "vectors", "templates", "nested_models"):
            if name in getattr(self, kind):
                return kind
        return None

    def _child_replacements(self, name: str, child: _ModelReference) -> dict[Basic, Basic]:
        """Match inherited child symbols by their attribute paths."""
        old = self.nested_models.get(name)
        replacements: dict[Basic, Basic] = {}
        if old is not None:
            for path, template in old._modello_templates.items():
                replacement = child._modello_templates.get(path)
                if replacement is not None and template._binding_signature == replacement._binding_signature:
                    replacements[template] = replacement
                else:
                    self._missing[template] = _path_label((name, *path))
            for path, vector in old._modello_model._modello_definition.vectors.items():
                target = child._modello_model._modello_definition.vectors.get(path)
                collection = name in self.collections
                if (
                    target is not None
                    and vector._finite
                    and target._finite
                    and len(target) != len(vector)
                    and not collection
                ):
                    raise ModelDefinitionError(
                        f"Cannot change the size of {_path_label((name, *path))}; declare a new model shape"
                    )
                if target is not None and not vector._finite:
                    replacements[vector.xreplace(old._modello_bindings)] = target.xreplace(child._modello_bindings)
            for path, dummy in old._modello_fields.items():
                target = child._modello_fields.get(path)
                if target is None:
                    self._missing[dummy] = _path_label((name, *path))
                elif dummy in replacements and replacements[dummy] != target:
                    self._missing[dummy] = f"{name} (changed field aliases)"
                else:
                    replacements[dummy] = target
            for dummy in self._missing:
                replacements.pop(dummy, None)
        return replacements

    def _replace(self, replacements: dict[Basic, Basic]) -> None:
        """Apply symbol changes simultaneously, including inherited aliases."""
        self._replacements = {d: replacements.get(value, value) for d, value in self._replacements.items()}
        self._replacements.update(replacements)
        self._missing = {value.xreplace(replacements): path for value, path in self._missing.items()}
        self.attrs = {key: expr.xreplace(replacements) for key, expr in self.attrs.items()}
        self.dummies = {key: replacements.get(d, d) for key, d in self.dummies.items()}
        self.vectors = {key: value.xreplace(replacements) for key, value in self.vectors.items()}
        self.vector_attrs = {key: value.xreplace(replacements) for key, value in self.vector_attrs.items()}
        self.templates = {key: value.xreplace(replacements) for key, value in self.templates.items()}
        self.collections = {key: value.substitute(replacements) for key, value in self.collections.items()}
        self.views = {key: value.substitute(replacements) for key, value in self.views.items()}
        self.model_laws = {key: value.substitute(replacements) for key, value in self.model_laws.items()}
        for key, law in self.model_laws.items():
            self.templates[key] = law.template
        self.structural_indices = {
            replacements.get(symbol, symbol): value.xreplace(replacements)
            for symbol, value in self.structural_indices.items()
        }
        self.nested_models = {
            key: _ModelReference(
                child._modello_model,
                {path: replacements.get(d, d) for path, d in child._modello_fields.items()},
                {symbol: value.xreplace(replacements) for symbol, value in child._modello_bindings.items()},
                child._modello_path,
                child._modello_view,
            )
            for key, child in self.nested_models.items()
        }
        for key, dummy in self.dummies.items():
            super().__setitem__(key, dummy)
        for key, vector in self.vectors.items():
            super().__setitem__(key, vector)
        for key, template in self.templates.items():
            super().__setitem__(key, template)
        for key, child in self.nested_models.items():
            super().__setitem__(key, child)

    def __setitem__(self, key: str, value: object) -> None:
        """Record expressions or create a scoped reference for a model class."""
        if key == "__modello_annotations__":
            raise ModelDefinitionError("__modello_annotations__ is managed by Modello")
        annotations: tuple[Annotation, ...] = ()
        if isinstance(value, _Annotated):
            annotations = value.annotations
            value = value.value
        self.add_annotations(key, annotations)
        if isinstance(value, _ModelLawSpec):
            value = value.substitute(self._replacements)
            self.model_laws[key] = value
            value = value.template
        elif isinstance(value, ModelRecurrence) or key in self.model_laws:
            raise ModelDefinitionError("Declare indexed model equations with model_recurrence(states[n], ...)")
        if key in self.collections and not isinstance(value, _ModelVectorSpec):
            raise ModelDefinitionError(f"Cannot replace model collection {self.name}.{key} with another field kind")
        is_view = isinstance(value, _Selection) or isinstance(value, _ModelReference) and value._modello_view
        if key in self.views and not is_view:
            raise ModelDefinitionError(f"Cannot replace component view {self.name}.{key} with another field kind")
        is_model = isinstance(value, ModelloMeta)
        if (isinstance(value, (Basic, _ModelVectorSpec, _Selection, _ModelReference)) or is_model) and key.startswith(
            "_modello_"
        ):
            raise ModelDefinitionError("The _modello_ prefix is reserved for model metadata")
        if isinstance(value, _ModelVectorSpec):
            value = value.substitute(self._replacements)
            if (
                key in self.dummies
                or key in self.vectors
                or key in self.templates
                or key in self.views
                or (key in self.nested_models and key not in self.collections)
            ):
                raise ModelDefinitionError(f"Cannot replace model field {self.name}.{key} with a model collection")
            previous = self.collections.get(key)
            if previous is not None and previous.model is not value.model:
                raise ModelDefinitionError(f"Cannot change the component schema of {self.name}.{key}")
            if previous is not None and previous.size != value.size:
                raise ModelDefinitionError(f"Cannot change the size of {self.name}.{key}; declare a new model shape")
            self.collections[key] = value
            value = self._declare_component(key, value.factory(value.size))
        elif isinstance(value, _Selection) or (isinstance(value, _ModelReference) and value._modello_view):
            if key in self.dummies or key in self.vectors or key in self.templates or key in self.collections:
                raise ModelDefinitionError(f"Cannot replace model quantity {self.name}.{key} with a component view")
            if key in self.nested_models and key not in self.views:
                raise ModelDefinitionError(f"Cannot replace component {self.name}.{key} with a component view")
            self.nested_models.pop(key, None)
            self.aliases.pop((key,), None)
            if isinstance(value, _ModelReference):
                selection = _Selection(value._modello_model, _Location(value._modello_path))
                self.nested_models[key] = value
                self.aliases[(key,)] = value._modello_path
            else:
                selection = value
            if key in self.views and selection.location != self.views[key].location:
                raise ModelDefinitionError(
                    f"Declare a new name instead of retargeting component view {self.name}.{key}"
                )
            self.views[key] = selection
        elif isinstance(value, Coordinate):
            if key in self.dummies or key in self.vectors or key in self.nested_models or key in self.templates:
                raise ModelDefinitionError(f"Cannot replace model field {self.name}.{key} with a local coordinate")
        elif isinstance(value, _SymbolicTemplate):
            if key in self.dummies or key in self.vectors or key in self.nested_models:
                raise ModelDefinitionError(f"Cannot replace a model quantity {self.name}.{key} with a symbolic law")
            template = cast(_SymbolicTemplate, value.xreplace(self._replacements))
            if key in self.templates:
                if self.templates[key]._binding_signature != template._binding_signature:
                    raise ModelDefinitionError(f"Cannot change the kind of symbolic declaration {self.name}.{key}")
                if self.templates[key]._same_declaration(template):
                    template = self.templates[key]
                self._replace({self.templates[key]: template})
            self.templates[key] = template
            value = template
        elif isinstance(value, Vector):
            value = self._declare_vector(key, value)
        elif isinstance(value, Basic):
            value = self._declare_scalar(key, value)
        elif is_model:
            value = self._declare_component(key, cast("type[Modello]", value))
        elif isinstance(value, _ModelReference):
            raise ModelInputError("Declare a component using its model class, not another component reference")
        elif key in self.dummies or key in self.vectors or key in self.nested_models or key in self.templates:
            raise ModelDefinitionError(f"Cannot assign {self.name}.{key} to a non-model value")
        super().__setitem__(key, value)

    def _declare_scalar(self, key: str, value: Basic) -> InstanceDummy:
        """Bind an expression and propagate changes to inherited aliases."""
        if key in self.nested_models or key in self.vectors or key in self.templates:
            raise ModelDefinitionError(f"Cannot replace nested model {self.name}.{key} with a scalar")
        expr = _declaration(value).xreplace(self._replacements)
        dummy = expr if isinstance(expr, InstanceDummy) else self.dummies.get(key)
        if dummy is None:
            dummy = InstanceDummy(key, **expr.assumptions0)
        if key in self.dummies:
            self._replace({self.dummies[key]: dummy})
        self.attrs[key] = expr
        self.dummies[key] = dummy
        return dummy

    def _require_shape(self, key: str, previous: Vector, replacement: Vector) -> None:
        if previous.shape != replacement.shape:
            raise ModelDefinitionError(f"Cannot change the size of {self.name}.{key}; declare a new model shape")

    def _declare_vector(self, key: str, value: Vector) -> Vector:
        """Keep the indexed formula while giving each coordinate a scalar identity."""
        if key in self.dummies or key in self.nested_models or key in self.templates:
            raise ModelDefinitionError(f"Cannot replace scalar or nested model {self.name}.{key} with a vector")
        expression = value.xreplace(self._replacements)
        previous = self.vectors.get(key)
        if previous is not None:
            self._require_shape(key, previous, expression)
        if not expression._finite:
            fields = (
                previous
                if previous is not None
                else (expression if isinstance(expression, _VectorSymbol) else Vector(key, size=expression.shape[0]))
            )
            self.vector_attrs[key] = expression
            self.vectors[key] = fields
            return fields
        if previous is not None and (
            not all(isinstance(entry, InstanceDummy) for entry in expression)
            or expression.free_symbols & set(self.fields().values())
        ):
            # A formula constrains the declared coordinates, including their
            # domains. Merging a boundary alias here could leave later entries
            # referring to an old, unbound coordinate of a recurrence.
            fields = previous
        else:
            fields = Vector(
                *(
                    entry if isinstance(entry, InstanceDummy) else InstanceDummy(f"{key}[{i}]", **entry.assumptions0)
                    for i, entry in enumerate(expression)
                )
            )
        if previous is not None and fields != previous:
            self._replace(dict(zip(previous, fields)))
        self.vector_attrs[key] = expression
        self.vectors[key] = fields
        return fields

    def _declare_component(self, key: str, model: type[Modello]) -> _ModelReference:
        """Scope a component and rebind any inherited references to its fields."""
        if key in self.dummies or key in self.vectors or key in self.templates:
            raise ModelDefinitionError(f"Cannot replace scalar {self.name}.{key} with a nested model")
        child = _ModelReference._modello_create(model, key)
        replacements = self._child_replacements(key, child)
        if replacements:
            self._replace(replacements)
        self.nested_models[key] = child
        return child

    def add_annotations(self, field: str, annotations: tuple[Annotation, ...]) -> None:
        """Reject duplicate declarations while allowing inherited metadata to be refined."""
        for annotation in annotations:
            if (type(annotation), field) in self._declared_annotations:
                path = f"{self.name}.{field}" if field else self.name
                raise ModelDefinitionError(f"Duplicate {type(annotation).__name__} annotation on {path}")
        for annotation in annotations:
            self.annotations.setdefault(type(annotation), {})[field] = annotation
            self._declared_annotations.add((type(annotation), field))

    def fields(self) -> dict[FieldPath, InstanceDummy]:
        """Flatten scalar fields while retaining their attribute paths."""
        fields: dict[FieldPath, InstanceDummy] = {(key,): dummy for key, dummy in self.dummies.items()}
        internal = self.get("_modello_internal_vectors", frozenset())
        fields.update(
            ((key, str(i)), dummy)
            for key, vector in self.vectors.items()
            if vector._finite and key not in internal
            for i, dummy in enumerate(vector)
        )
        fields.update(
            ((key, *path), dummy)
            for key, child in self.nested_models.items()
            for path, dummy in child._modello_fields.items()
        )
        fields.update(
            ((key, str(i)), dummy)
            for key, vector in self.vectors.items()
            if vector._finite and key in internal
            for i, dummy in enumerate(vector)
        )
        return fields

    def reference_at(self, path: FieldPath) -> _ModelReference:
        result = self.nested_models[path[0]]
        for name in path[1:]:
            result = getattr(result, name)
        return result

    def _resolve_views(self) -> None:
        known = _known_integers(self)
        for name, selection in self.views.items():
            try:
                target = selection.location.resolve(self, known)
            except (_UnresolvedSelection, KeyError):
                continue
            if not isinstance(target, _ModelReference):
                raise ModelDefinitionError("A model view must select a declared component")
            self.nested_models[name] = target
            self.aliases[(name,)] = target._modello_path
            dict.__setitem__(self, name, target)

    def compile(self) -> _ModelDefinition:
        """Include descendants' equations, preserving their paths and aliases."""
        self.pending_laws.clear()
        self._resolve_views()
        constraints = [
            _Definition((attr,), self.dummies[attr], expr)
            for attr, expr in self.attrs.items()
            if expr != self.dummies[attr]
        ]
        constraints.extend(
            _Definition((name,), symbol, self.structural_indices[symbol])
            for name, symbol in self.dummies.items()
            if symbol in self.structural_indices
        )
        constraints.extend(
            _Definition((attr, str(i)), self.vectors[attr][i], expression)
            for attr, value in self.vector_attrs.items()
            if value._finite
            for i, expression in enumerate(value)
            if expression != self.vectors[attr][i]
        )
        for name, law in self.model_laws.items():
            try:
                constraints.extend(_Definition(*definition) for definition in law.compile(self, law))
            except _UnresolvedSelection:
                self.pending_laws.add(name)
        for attr, child in self.nested_models.items():
            if attr in self.views:
                continue
            definition = child._modello_model._modello_definition
            replacements = child._modello_bindings
            constraints.extend(
                _Definition(
                    (attr, *constraint.path),
                    replacements[constraint.left],
                    constraint.right.xreplace(replacements).xreplace(self._replacements),
                )
                for constraint in definition.constraints
                if not any((attr, *constraint.path)[: len(path)] == path for path in self.aliases)
            )
        expressions = [
            *self.attrs.values(),
            *self.vector_attrs.values(),
            *self.templates.values(),
            *(constraint.right for constraint in constraints),
        ]
        used = set().union(*(expr.free_symbols for expr in expressions))
        used.update(template for expression in expressions for template in expression.atoms(_SymbolicTemplate))
        for dummy in used & self._missing.keys():
            raise ModelDefinitionError(
                f"{self.name} references removed or ambiguous child field {self._missing[dummy]}"
            )
        vectors: dict[FieldPath, Vector] = {(name,): vector for name, vector in self.vectors.items()}
        vector_definitions: dict[FieldPath, Vector] = {
            (name,): expression for name, expression in self.vector_attrs.items() if expression != self.vectors[name]
        }
        templates: dict[FieldPath, _SymbolicTemplate] = {(name,): value for name, value in self.templates.items()}
        for name, child in self.nested_models.items():
            definition = child._modello_model._modello_definition
            replacements = child._modello_bindings
            vectors.update(((name, *path), value.xreplace(replacements)) for path, value in definition.vectors.items())
            vector_definitions.update(
                ((name, *path), value.xreplace(replacements).xreplace(self._replacements))
                for path, value in definition.vector_definitions.items()
            )
            templates.update(
                ((name, *path), value.xreplace(replacements).xreplace(self._replacements))
                for path, value in definition.templates.items()
            )
        constraints = [_Definition(c.path, c.left, _project_series(c.right, templates.values())) for c in constraints]
        vector_definitions = {
            path: _project_series(value, templates.values()) for path, value in vector_definitions.items()
        }
        return _ModelDefinition(
            self.fields(), tuple(constraints), vectors, vector_definitions, templates, dict(self.aliases)
        )

    def compiled_annotations(self) -> Mapping[type[Annotation], Mapping[str, Annotation]]:
        """Scope child annotations before applying this class's refinements."""
        annotations: dict[type[Annotation], dict[str, Annotation]] = {}
        for attr, child in self.nested_models.items():
            prefix = _path_label((attr,))
            for kind, fields in child._modello_model.__modello_annotations__.items():
                annotations.setdefault(kind, {}).update(
                    (prefix + ("" if path.startswith("[") else ".") + path if path else prefix, annotation)
                    for path, annotation in fields.items()
                )
        for kind, fields in self.annotations.items():
            annotations.setdefault(kind, {}).update(fields)
        return _freeze(annotations)


class ModelloMeta(type):
    """Compile a model tree's field paths and equations once per class."""

    @classmethod
    def __prepare__(metacls, __name: str, __bases: tuple[type, ...], **kwds: Any) -> MutableMapping[str, Any]:
        """Collect model declarations in a specialized class namespace."""
        return ModelloMetaNamespace(__name, __bases)

    def __new__(mcs, name: str, bases: tuple[type, ...], meta_namespace: ModelloMetaNamespace) -> Any:
        """Attach the compiled declaration and annotations to an ordinary class."""
        definition = meta_namespace.compile()
        namespace = dict(meta_namespace)
        namespace["_modello_namespace"] = meta_namespace
        namespace["_modello_definition"] = definition
        namespace["__modello_annotations__"] = meta_namespace.compiled_annotations()
        return super().__new__(mcs, name, bases, namespace)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        concrete = _specialize_model(cast("type[Modello]", cls), kwargs)
        if concrete is not cls:
            return concrete(*args, **kwargs)
        return super().__call__(*args, **kwargs)


ShapeModel = TypeVar("ShapeModel", bound="Modello")


def _known_integers(
    namespace: ModelloMetaNamespace,
    observations: Mapping[str, object] | None = None,
    bindings: Mapping[Symbol, Basic] | None = None,
) -> dict[Symbol, Integer]:
    """Resolve declared integer expressions used to select component structure."""
    known = dict(bindings or {})
    for name, value in (observations or {}).items():
        if name in namespace.dummies:
            try:
                scalar = sympify(value)
            except (TypeError, ValueError):
                continue
            if isinstance(scalar, Integer):
                known[namespace.dummies[name]] = scalar
    for _ in range(len(namespace.attrs) + 1):
        additions = {}
        for name, value in namespace.attrs.items():
            symbol = namespace.dummies[name]
            if symbol not in known:
                resolved = _substitute(value, known)
                if isinstance(resolved, Integer):
                    additions[symbol] = resolved
        if not additions:
            break
        known.update(additions)
    return {symbol: value for symbol, value in known.items() if isinstance(value, Integer)}


def _has_symbolic_shape(model: type[Modello]) -> bool:
    namespace = model._modello_namespace
    return (
        bool(namespace.views.keys() - namespace.nested_models.keys())
        or bool(namespace.pending_laws)
        or any(not value._finite for value in namespace.vectors.values())
        or any(not value._finite for value in namespace.collections.values())
        or any(
            _has_symbolic_shape(child._modello_model)
            for name, child in namespace.nested_models.items()
            if name not in namespace.views
        )
    )


def _specialize_model(model: type[ShapeModel], values: Mapping[str, object]) -> type[ShapeModel]:
    """Choose a finite declaration without changing the reusable model class."""
    namespace = model._modello_namespace
    children = []
    for name, child in namespace.nested_models.items():
        if name in namespace.collections or name in namespace.views:
            continue
        if _has_symbolic_shape(child._modello_model):
            value = values.get(name)
            if not isinstance(value, child._modello_model):
                raise ModelInputError(
                    f"Supply a {child._modello_model.__name__} instance to determine {name}'s vector sizes"
                )
            children.append((name, type(value)))
    axes: dict[str, Vector | _ModelVectorSpec] = {**namespace.vectors, **namespace.collections}
    finite = all(vector._finite for vector in axes.values())
    pending = namespace.views.keys() - namespace.nested_models.keys()
    if finite and not children and not pending and not namespace.pending_laws:
        return model
    dimensions = {} if finite else _dimensions(axes, namespace.dummies, namespace.attrs, values)
    known = _known_integers(namespace, values, dimensions)
    selected = {}
    locations = [namespace.views[name].location for name in pending]
    for location in locations:
        index = location.index
        if index is not None:
            if not isinstance(index.xreplace(known), Integer):
                raise ModelInputError("A model selection needs a known finite integer index")
            selected.update({symbol: known[symbol] for symbol in index.free_symbols if symbol in known})
    return cast(
        type[ShapeModel],
        _concrete_model(
            model,
            tuple(sorted(dimensions.items(), key=lambda item: str(item[0]))),
            tuple(children),
            tuple(sorted(selected.items(), key=lambda item: str(item[0]))),
        ),
    )


@lru_cache(maxsize=128)
def _concrete_model(
    model: type[Modello],
    dimensions: tuple[tuple[Symbol, Basic], ...],
    children: tuple[tuple[str, type[Modello]], ...],
    selected: tuple[tuple[Symbol, Basic], ...] = (),
) -> type[Modello]:
    bindings = dict(dimensions)
    namespace = ModelloMetaNamespace(model.__name__, (model,))
    for name, child in children:
        namespace[name] = child
    for name, spec in tuple(namespace.collections.items()):
        if not spec._finite:
            dict.__setitem__(
                namespace, name, namespace._declare_component(name, spec.factory(spec.size.xreplace(bindings)))
            )
    fields = {vector: vector.xreplace(bindings) for vector in namespace.vectors.values() if not vector._finite}
    namespace._replace(fields)
    namespace.collections = {
        name: spec.substitute({**bindings, **dict(selected)}) for name, spec in namespace.collections.items()
    }
    namespace.views = {
        name: selection.substitute({**bindings, **dict(selected)}) for name, selection in namespace.views.items()
    }
    namespace.structural_indices.update(selected)
    namespace.vector_attrs = {
        name: expression.xreplace(bindings) for name, expression in namespace.vector_attrs.items()
    }
    for name, symbol in namespace.dummies.items():
        if symbol in bindings:
            namespace[name] = bindings[symbol]
    dict.__setitem__(namespace, "_modello_shape_source", model)
    namespace["__module__"] = model.__module__
    return ModelloMeta(model.__name__, (model,), namespace)


def _populate(
    instance: Modello, problem: ModelProblem, solution: dict[Symbol, Expr], inputs: frozenset[Symbol] = frozenset()
) -> None:
    """Materialize the model tree from one shared solution, without solving."""
    fields = {field.path: cast(BoundInstanceDummy, field.symbol) for field in problem.fields}
    input_symbols = inputs | {c.left for c in problem.constraints if c.kind is ConstraintKind.INPUT}
    symbols = problem.symbols
    aliases: dict[FieldPath, FieldPath] = {}
    resolve = (
        _algebra(problem.constraints, problem.templates.values(), solution).resolve.symbolic()
        if problem.templates
        else None
    )

    def collect_aliases(model: type[Modello], prefix: FieldPath) -> None:
        aliases.update({prefix + path: prefix + target for path, target in model._modello_definition.aliases.items()})
        for attr, child in model._modello_namespace.nested_models.items():
            path = (*prefix, attr)
            if path not in aliases:
                collect_aliases(child._modello_model, path)

    collect_aliases(type(instance), ())

    def canonical(path: FieldPath) -> FieldPath:
        seen = set()
        while path not in seen:
            seen.add(path)
            match = next((part for part in sorted(aliases, key=len, reverse=True) if path[: len(part)] == part), None)
            if match is None:
                return path
            path = (*aliases[match], *path[len(match) :])
        raise ModelDefinitionError("Cyclic component views")

    instances: dict[FieldPath, Modello] = {}

    def populate(instance: Modello, model: type[Modello], prefix: FieldPath) -> None:
        instances[prefix] = instance
        reference = _path_value(symbols, prefix)
        instance._modello_bound_functions = {
            symbol: reference._modello_bindings[symbol] for symbol in model._modello_definition.template_symbols
        }
        bound_fields = {d: fields[prefix + path] for path, d in model._modello_definition.fields.items()}
        instance._modello_bound_fields = {
            d: _BoundField(
                bound, solution[bound], ConstraintKind.INPUT if bound in input_symbols else ConstraintKind.INFERRED
            )
            for d, bound in bound_fields.items()
        }
        for attr in model._modello_namespace.dummies:
            setattr(instance, attr, solution[fields[prefix + (attr,)]])
        for attr, vector in model._modello_namespace.vectors.items():
            setattr(instance, attr, Vector(*(solution[fields[prefix + (attr, str(i))]] for i in range(len(vector)))))
        _populate_templates(
            instance,
            {attr: problem.templates[prefix + (attr,)] for attr in model._modello_namespace.templates},
            solution,
            resolve,
        )
        for attr, child in model._modello_namespace.nested_models.items():
            path = (*prefix, attr)
            if path in aliases:
                continue
            child_instance = object.__new__(child._modello_model)
            populate(child_instance, child._modello_model, path)
            setattr(instance, attr, child_instance)

    populate(instance, type(instance), ())
    for path in aliases:
        setattr(instances[canonical(path[:-1])], path[-1], instances[canonical(path)])


def _populate_templates(
    instance: Modello,
    templates: Mapping[str, _SymbolicTemplate],
    values: Mapping[Symbol, Basic],
    resolve: _DefinitionResolver | None = None,
) -> None:
    """Keep scoped formulas reusable while presenting their evaluated parameters."""
    expanded = {name: resolve(template) if resolve is not None else template for name, template in templates.items()}
    instance._modello_bound_templates = expanded
    for name, template in expanded.items():
        setattr(instance, name, _substitute(template, values))

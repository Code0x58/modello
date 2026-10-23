"""Private model declarations, field binding, and instance population."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, TypeVar, cast

from sympy import Basic, Expr, Symbol

from modello._expressions import _declaration, _substitute
from modello._operators import _FunctionSymbol, _SymbolicTemplate
from modello._paths import _path_label
from modello._problem import ConstraintKind, FieldPath
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

    @classmethod
    def _modello_create(cls, model: type[Modello], name: str) -> _ModelReference:
        """Scope child symbols independently while preserving aliases."""
        replacements = {
            d: InstanceDummy(f"{name}_{d.name}", **d.assumptions0)
            for d in dict.fromkeys(model._modello_definition.fields.values())
        }
        replacements.update({symbol: symbol.bound(name) for symbol in model._modello_definition.template_symbols})
        return cls(model, {path: replacements[d] for path, d in model._modello_definition.fields.items()}, replacements)

    @property
    def _modello_templates(self) -> dict[FieldPath, _SymbolicTemplate]:
        return {
            path: value.xreplace(self._modello_bindings)
            for path, value in self._modello_model._modello_definition.templates.items()
        }

    def __getattr__(self, name: str) -> InstanceDummy | Vector | _ModelReference | _SymbolicTemplate:
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
        child = self._modello_model._modello_namespace.nested_models.get(name)
        if child is None:
            raise AttributeError(f"{self._modello_model.__name__} has no model attribute {name!r}")
        return _ModelReference(
            child._modello_model,
            {path[1:]: d for path, d in self._modello_fields.items() if path[0] == name},
            {
                original: self._modello_bindings.get(scoped, scoped)
                for original, scoped in child._modello_bindings.items()
            },
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
        self.annotations: dict[type[Annotation], dict[str, Annotation]] = {}
        self._declared_annotations: set[tuple[type[Annotation], str]] = set()
        self._replacements: dict[Basic, Basic] = {}
        self._missing: dict[Basic, str] = {}
        # Preserve the existing precedence: later model bases override earlier ones.
        for base in bases:
            parent = getattr(base, "_modello_namespace", None)
            if not isinstance(parent, ModelloMetaNamespace):
                continue
            for kind, fields in parent.annotations.items():
                self.annotations.setdefault(kind, {}).update(fields)
            conflicts = (
                self.dummies.keys() & (parent.nested_models.keys() | parent.vectors.keys())
                | self.vectors.keys() & (parent.nested_models.keys() | parent.dummies.keys())
                | self.nested_models.keys() & (parent.dummies.keys() | parent.vectors.keys())
                | self.templates.keys() & (parent.dummies.keys() | parent.vectors.keys() | parent.nested_models.keys())
                | parent.templates.keys() & (self.dummies.keys() | self.vectors.keys() | self.nested_models.keys())
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
            self._replacements.update(parent._replacements)
            self._missing.update(parent._missing)
            self._replace(replacements)

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
                if target is not None and vector._finite and target._finite and len(target) != len(vector):
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
        self.nested_models = {
            key: _ModelReference(
                child._modello_model,
                {path: replacements.get(d, d) for path, d in child._modello_fields.items()},
                {symbol: value.xreplace(replacements) for symbol, value in child._modello_bindings.items()},
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
        is_model = isinstance(value, ModelloMeta)
        if (isinstance(value, Basic) or is_model) and key.startswith("_modello_"):
            raise ModelDefinitionError("The _modello_ prefix is reserved for model metadata")
        if isinstance(value, Coordinate):
            if key in self.dummies or key in self.vectors or key in self.nested_models or key in self.templates:
                raise ModelDefinitionError(f"Cannot replace model field {self.name}.{key} with a local coordinate")
        elif isinstance(value, _SymbolicTemplate):
            if key in self.dummies or key in self.vectors or key in self.nested_models:
                raise ModelDefinitionError(f"Cannot replace a model quantity {self.name}.{key} with a symbolic law")
            template = cast(_SymbolicTemplate, value.xreplace(self._replacements))
            if key in self.templates:
                if self.templates[key]._binding_signature != template._binding_signature:
                    raise ModelDefinitionError(f"Cannot change the kind of symbolic declaration {self.name}.{key}")
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
        fields.update(
            ((key, str(i)), dummy)
            for key, vector in self.vectors.items()
            if vector._finite
            for i, dummy in enumerate(vector)
        )
        fields.update(
            ((key, *path), dummy)
            for key, child in self.nested_models.items()
            for path, dummy in child._modello_fields.items()
        )
        return fields

    def compile(self) -> _ModelDefinition:
        """Include descendants' equations, preserving their paths and aliases."""
        constraints = [
            _Definition((attr,), self.dummies[attr], expr)
            for attr, expr in self.attrs.items()
            if expr != self.dummies[attr]
        ]
        constraints.extend(
            _Definition((attr, str(i)), self.vectors[attr][i], expression)
            for attr, value in self.vector_attrs.items()
            if value._finite
            for i, expression in enumerate(value)
            if expression != self.vectors[attr][i]
        )
        for attr, child in self.nested_models.items():
            definition = child._modello_model._modello_definition
            replacements = child._modello_bindings
            constraints.extend(
                _Definition(
                    (attr, *constraint.path),
                    replacements[constraint.left],
                    constraint.right.xreplace(replacements).xreplace(self._replacements),
                )
                for constraint in definition.constraints
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
        return _ModelDefinition(self.fields(), tuple(constraints), vectors, vector_definitions, templates)

    def compiled_annotations(self) -> Mapping[type[Annotation], Mapping[str, Annotation]]:
        """Scope child annotations before applying this class's refinements."""
        annotations: dict[type[Annotation], dict[str, Annotation]] = {}
        for attr, child in self.nested_models.items():
            for kind, fields in child._modello_model.__modello_annotations__.items():
                annotations.setdefault(kind, {}).update(
                    (f"{attr}.{path}" if path else attr, annotation) for path, annotation in fields.items()
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
        namespace = dict(meta_namespace)
        namespace["_modello_namespace"] = meta_namespace
        namespace["_modello_definition"] = meta_namespace.compile()
        namespace["__modello_annotations__"] = meta_namespace.compiled_annotations()
        return super().__new__(mcs, name, bases, namespace)

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        concrete = _specialize_model(cast("type[Modello]", cls), kwargs)
        if concrete is not cls:
            return concrete(*args, **kwargs)
        return super().__call__(*args, **kwargs)


ShapeModel = TypeVar("ShapeModel", bound="Modello")


def _specialize_model(model: type[ShapeModel], values: Mapping[str, object]) -> type[ShapeModel]:
    """Choose a finite declaration without changing the reusable model class."""
    namespace = model._modello_namespace
    children = []
    for name, child in namespace.nested_models.items():
        if any(not vector._finite for vector in child._modello_model._modello_definition.vectors.values()):
            value = values.get(name)
            if not isinstance(value, child._modello_model):
                raise ModelInputError(
                    f"Supply a {child._modello_model.__name__} instance to determine {name}'s vector sizes"
                )
            children.append((name, type(value)))
    if all(vector._finite for vector in namespace.vectors.values()) and not children:
        return model
    dimensions = _dimensions(namespace.vectors, namespace.dummies, namespace.attrs, values)
    return cast(
        type[ShapeModel],
        _concrete_model(model, tuple(sorted(dimensions.items(), key=lambda item: str(item[0]))), tuple(children)),
    )


@lru_cache(maxsize=128)
def _concrete_model(
    model: type[Modello], dimensions: tuple[tuple[Symbol, Basic], ...], children: tuple[tuple[str, type[Modello]], ...]
) -> type[Modello]:
    bindings = dict(dimensions)
    namespace = ModelloMetaNamespace(model.__name__, (model,))
    for name, child in children:
        namespace[name] = child
    fields = {vector: vector.xreplace(bindings) for vector in namespace.vectors.values() if not vector._finite}
    namespace._replace(fields)
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

    def populate(instance: Modello, model: type[Modello], prefix: FieldPath) -> None:
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
        )
        for attr, child in model._modello_namespace.nested_models.items():
            child_instance = object.__new__(child._modello_model)
            populate(child_instance, child._modello_model, (*prefix, attr))
            setattr(instance, attr, child_instance)

    populate(instance, type(instance), ())


def _populate_templates(
    instance: Modello, templates: Mapping[str, _SymbolicTemplate], values: Mapping[Symbol, Basic]
) -> None:
    """Keep original scoped laws while presenting their evaluated parameters."""
    instance._modello_bound_templates = dict(templates)
    for name, template in templates.items():
        setattr(instance, name, _substitute(template, values))

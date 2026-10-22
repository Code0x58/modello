"""Private model declarations, field binding, and instance population."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, Protocol, TypeAlias, cast

from sympy import Basic, Dummy, Expr, Symbol
from sympy.simplify.simplify import simplify

from modello._problem import ConstraintKind, FieldPath
from modello.annotations import Annotation, _Annotated, _freeze
from modello.errors import ModelDefinitionError, ModelInputError

if TYPE_CHECKING:
    from modello import Modello
    from modello.diagnostics import ModelProblem


class SymbolicScalar(Protocol):
    """A scalar expression accepted by the constructor's input annotations."""

    @property
    def free_symbols(self) -> set[Any]:
        """Return the expression's free symbols."""
        ...

    @property
    def is_commutative(self) -> bool | None:
        """Return the expression's commutativity assumption."""
        ...


class SymbolicCondition(Protocol):
    """A symbolic relation or Boolean accepted by solution-family refinement."""

    @property
    def free_symbols(self) -> set[Any]:
        """Return the condition's free symbols."""
        ...

    def as_set(self) -> object:
        """Return the condition's set representation, when available."""
        ...


ScalarValue: TypeAlias = SymbolicScalar | int | float | complex | str | Decimal | Fraction


class InstanceDummy(Dummy):
    """A symbol that is rebound independently for each model instance."""

    def bound(self, model_name: str) -> BoundInstanceDummy:
        """Return a fresh symbol with this symbol's assumptions."""
        return BoundInstanceDummy(model_name + "_" + self.name, **self.assumptions0)


class BoundInstanceDummy(InstanceDummy):
    """A symbol belonging to a particular model instance."""


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


@dataclass(frozen=True)
class _ModelReference:
    """Expose a child's complete field tree during class definition."""

    _modello_model: type[Modello]
    _modello_fields: dict[FieldPath, InstanceDummy]

    @classmethod
    def _modello_create(cls, model: type[Modello], name: str) -> _ModelReference:
        """Scope child symbols independently while preserving aliases."""
        replacements = {
            d: InstanceDummy(f"{name}_{d.name}", **d.assumptions0)
            for d in dict.fromkeys(model._modello_definition.fields.values())
        }
        return cls(model, {path: replacements[d] for path, d in model._modello_definition.fields.items()})

    def __getattr__(self, name: str) -> InstanceDummy | _ModelReference:
        """Resolve a scalar or a reference to a deeper child."""
        if (name,) in self._modello_fields:
            return self._modello_fields[(name,)]
        child = self._modello_model._modello_namespace.nested_models.get(name)
        if child is None:
            raise AttributeError(f"{self._modello_model.__name__} has no model attribute {name!r}")
        return _ModelReference(
            child._modello_model,
            {path[1:]: d for path, d in self._modello_fields.items() if path[0] == name},
        )


class ModelloMetaNamespace(dict):
    """Track scalar expressions and named children in a model class body."""

    def __init__(self, name: str, bases: tuple[type, ...]) -> None:
        """Inherit model fields without copying ordinary methods or class cells."""
        super().__init__()
        self.name = name
        self.attrs: dict[str, Basic] = {}
        self.dummies: dict[str, InstanceDummy] = {}
        self.nested_models: dict[str, _ModelReference] = {}
        self.annotations: dict[type[Annotation], dict[str, Annotation]] = {}
        self._declared_annotations: set[tuple[type[Annotation], str]] = set()
        self._replacements: dict[InstanceDummy, InstanceDummy] = {}
        self._missing: dict[InstanceDummy, str] = {}
        # Preserve the existing precedence: later model bases override earlier ones.
        for base in bases:
            parent = getattr(base, "_modello_namespace", None)
            if not isinstance(parent, ModelloMetaNamespace):
                continue
            for kind, fields in parent.annotations.items():
                self.annotations.setdefault(kind, {}).update(fields)
            conflicts = (self.dummies.keys() & parent.nested_models.keys()) | (
                self.nested_models.keys() & parent.dummies.keys()
            )
            if conflicts:
                raise ModelDefinitionError(
                    f"Cannot merge scalar and nested model fields: {', '.join(sorted(conflicts))}"
                )
            replacements = {self.dummies[key]: dummy for key, dummy in parent.dummies.items() if key in self.dummies}
            for key, child in parent.nested_models.items():
                replacements.update(self._child_replacements(key, child))
            self.attrs.update(parent.attrs)
            self.dummies.update(parent.dummies)
            self.nested_models.update(parent.nested_models)
            self._replacements.update(parent._replacements)
            self._missing.update(parent._missing)
            self._replace(replacements)

    def _child_replacements(self, name: str, child: _ModelReference) -> dict[InstanceDummy, InstanceDummy]:
        """Match inherited child symbols by their attribute paths."""
        old = self.nested_models.get(name)
        replacements: dict[InstanceDummy, InstanceDummy] = {}
        if old is not None:
            for path, dummy in old._modello_fields.items():
                target = child._modello_fields.get(path)
                if target is None:
                    self._missing[dummy] = ".".join((name, *path))
                elif dummy in replacements and replacements[dummy] != target:
                    self._missing[dummy] = f"{name} (changed field aliases)"
                else:
                    replacements[dummy] = target
            for dummy in self._missing:
                replacements.pop(dummy, None)
        return replacements

    def _replace(self, replacements: dict[InstanceDummy, InstanceDummy]) -> None:
        """Apply symbol changes simultaneously, including inherited aliases."""
        self._replacements = {d: replacements.get(value, value) for d, value in self._replacements.items()}
        self._replacements.update(replacements)
        self.attrs = {key: expr.xreplace(replacements) for key, expr in self.attrs.items()}
        self.dummies = {key: replacements.get(d, d) for key, d in self.dummies.items()}
        self.nested_models = {
            key: _ModelReference(
                child._modello_model,
                {path: replacements.get(d, d) for path, d in child._modello_fields.items()},
            )
            for key, child in self.nested_models.items()
        }
        for key, dummy in self.dummies.items():
            super().__setitem__(key, dummy)
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
        if isinstance(value, Basic):
            value = self._declare_scalar(key, value)
        elif is_model:
            value = self._declare_component(key, cast("type[Modello]", value))
        elif isinstance(value, _ModelReference):
            raise ModelInputError("Declare a component using its model class, not another component reference")
        elif key in self.dummies or key in self.nested_models:
            raise ModelDefinitionError(f"Cannot assign {self.name}.{key} to a non-model value")
        super().__setitem__(key, value)

    def _declare_scalar(self, key: str, value: Basic) -> InstanceDummy:
        """Bind an expression and propagate changes to inherited aliases."""
        if key in self.nested_models:
            raise ModelDefinitionError(f"Cannot replace nested model {self.name}.{key} with a scalar")
        expr = simplify(value).xreplace(self._replacements)
        dummy = expr if isinstance(expr, InstanceDummy) else self.dummies.get(key)
        if dummy is None:
            dummy = InstanceDummy(key, **expr.assumptions0)
        if key in self.dummies:
            self._replace({self.dummies[key]: dummy})
        self.attrs[key] = expr
        self.dummies[key] = dummy
        return dummy

    def _declare_component(self, key: str, model: type[Modello]) -> _ModelReference:
        """Scope a component and rebind any inherited references to its fields."""
        if key in self.dummies:
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
        for attr, child in self.nested_models.items():
            definition = child._modello_model._modello_definition
            replacements = {d: child._modello_fields[path] for path, d in definition.fields.items()}
            constraints.extend(
                _Definition(
                    (attr, *constraint.path),
                    replacements[constraint.left],
                    constraint.right.xreplace(replacements).xreplace(self._replacements),
                )
                for constraint in definition.constraints
            )
        expressions = [*self.attrs.values(), *(constraint.right for constraint in constraints)]
        used = set().union(*(expr.free_symbols for expr in expressions))
        for dummy in used & self._missing.keys():
            raise ModelDefinitionError(
                f"{self.name} references removed or ambiguous child field {self._missing[dummy]}"
            )
        return _ModelDefinition(self.fields(), tuple(constraints))

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
        for attr, child in model._modello_namespace.nested_models.items():
            child_instance = object.__new__(child._modello_model)
            populate(child_instance, child._modello_model, (*prefix, attr))
            setattr(instance, attr, child_instance)

    populate(instance, type(instance), ())

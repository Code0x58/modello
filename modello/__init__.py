"""Symbolic models whose named children form a single system of equations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Any, ClassVar, Mapping, MutableMapping, Protocol, TypeAlias, TypeVar, cast

from sympy import Basic, Dummy, Eq, Expr, S, Symbol, SympifyError, solve
from sympy.simplify.simplify import simplify

from modello.annotations import Annotation as Annotation
from modello.annotations import _Annotated, _freeze
from modello.annotations import annotate as annotate
from modello.annotations import annotate_model as annotate_model
from modello.diagnostics import (
    Candidate,
    ConstraintKind,
    FieldPath,
    FieldValue,
    ModelConstraint,
    ModelField,
    ModelProblem,
)
from modello.errors import (
    ModelDefinitionError,
    ModelInputError,
    SolutionCountError,
    SolverBackendError,
)
from modello.errors import (
    ModelloError as ModelloError,
)
from modello.errors import (
    ModelSolveError as ModelSolveError,
)
from modello.errors import (
    ModelValidationError as ModelValidationError,
)


# A structural protocol keeps untyped SymPy imports out of the input union.
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
            for d in dict.fromkeys(model._modello_fields.values())
        }
        return cls(model, {path: replacements[d] for path, d in model._modello_fields.items()})

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
        self._replacements: dict[InstanceDummy, InstanceDummy] = {}
        self._missing: dict[InstanceDummy, str] = {}
        # Preserve the existing precedence: later model bases override earlier ones.
        for base in bases:
            parent = getattr(base, "_modello_namespace", None)
            if not isinstance(parent, ModelloMetaNamespace):
                continue
            for kind, fields in parent.annotations.items():
                self.annotations.setdefault(kind, {}).update(fields)
            # Model decorators run after namespace compilation.
            for kind, fields in getattr(base, "__modello_annotations__", {}).items():
                if "" in fields:
                    self.annotations.setdefault(kind, {})[""] = fields[""]
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
        is_model = isinstance(value, ModelloMeta)
        if (isinstance(value, Basic) or is_model) and key.startswith("_modello_"):
            raise ModelDefinitionError("The _modello_ prefix is reserved for model metadata")
        if isinstance(value, Basic):
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
            value = dummy
        elif is_model:
            if key in self.dummies:
                raise ModelDefinitionError(f"Cannot replace scalar {self.name}.{key} with a nested model")
            child = _ModelReference._modello_create(cast(type[Modello], value), key)
            replacements = self._child_replacements(key, child)
            if replacements:
                self._replace(replacements)
            self.nested_models[key] = child
            value = child
        elif isinstance(value, _ModelReference):
            raise ModelInputError("Declare a component using its model class, not another component reference")
        elif key in self.dummies or key in self.nested_models:
            raise ModelDefinitionError(f"Cannot assign {self.name}.{key} to a non-model value")
        for annotation in annotations:
            self.annotations.setdefault(type(annotation), {})[key] = annotation
        super().__setitem__(key, value)

    def fields(self) -> dict[FieldPath, InstanceDummy]:
        """Flatten scalar fields while retaining their attribute paths."""
        fields: dict[FieldPath, InstanceDummy] = {(key,): dummy for key, dummy in self.dummies.items()}
        fields.update(
            ((key, *path), dummy)
            for key, child in self.nested_models.items()
            for path, dummy in child._modello_fields.items()
        )
        return fields


class ModelloMeta(type):
    """Compile a model tree's field paths and equations once per class."""

    @classmethod
    def __prepare__(metacls, __name: str, __bases: tuple[type, ...], **kwds: Any) -> MutableMapping[str, Any]:
        """Collect model declarations in a specialized class namespace."""
        return ModelloMetaNamespace(__name, __bases)

    def __new__(mcs, name: str, bases: tuple[type, ...], meta_namespace: ModelloMetaNamespace) -> Any:
        """Include every descendant's constraints with its own scoped symbols."""
        namespace = dict(meta_namespace)
        constraints = [
            (meta_namespace.dummies[attr], expr)
            for attr, expr in meta_namespace.attrs.items()
            if expr != meta_namespace.dummies[attr]
        ]
        paths: list[FieldPath] = [
            (attr,) for attr, expr in meta_namespace.attrs.items() if expr != meta_namespace.dummies[attr]
        ]
        for attr, child in meta_namespace.nested_models.items():
            model = child._modello_model
            replacements = {d: child._modello_fields[path] for path, d in model._modello_fields.items()}
            constraints.extend(
                (replacements[d], expr.xreplace(replacements).xreplace(meta_namespace._replacements))
                for d, expr in model._modello_class_constraints
            )
            paths.extend((attr, *path) for path in model._modello_constraint_paths)
        expressions = [*meta_namespace.attrs.values(), *(expr for _, expr in constraints)]
        used = set().union(*(expr.free_symbols for expr in expressions))
        for dummy in used & meta_namespace._missing.keys():
            raise ModelDefinitionError(
                f"{name} references removed or ambiguous child field {meta_namespace._missing[dummy]}"
            )
        namespace["_modello_namespace"] = meta_namespace
        namespace["_modello_fields"] = meta_namespace.fields()
        namespace["_modello_class_constraints"] = tuple(constraints)
        namespace["_modello_constraint_paths"] = tuple(paths)
        annotations: dict[type[Annotation], dict[str, Annotation]] = {}
        for attr, child in meta_namespace.nested_models.items():
            for kind, fields in child._modello_model.__modello_annotations__.items():
                annotations.setdefault(kind, {}).update(
                    (f"{attr}.{path}" if path else attr, annotation) for path, annotation in fields.items()
                )
        for kind, fields in meta_namespace.annotations.items():
            annotations.setdefault(kind, {}).update(fields)
        namespace["__modello_annotations__"] = _freeze(annotations)
        return super().__new__(mcs, name, bases, namespace)


class Modello(metaclass=ModelloMeta):
    """Build a symbolic model, solving its named children together with it."""

    _modello_namespace: ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace("", ())
    _modello_fields: ClassVar[dict[FieldPath, InstanceDummy]] = {}
    _modello_class_constraints: ClassVar[tuple[tuple[InstanceDummy, Basic], ...]] = ()
    _modello_constraint_paths: ClassVar[tuple[FieldPath, ...]] = ()
    _modello_bound_fields: dict[InstanceDummy, _BoundField]
    __modello_annotations__: ClassVar[Mapping[type[Annotation], Mapping[str, Annotation]]]

    def __init__(self, name: str, **value_map: ScalarValue | Modello) -> None:
        """Accept scalar values and compatible child model instances."""
        problem = inspect_model(type(self), name, **value_map)
        fields = {field.path: cast(BoundInstanceDummy, field.symbol) for field in problem.fields}
        reported = _solve_problem(problem)
        candidates = []
        for resolved in reported:
            if any(value.has(S.NaN, S.ComplexInfinity) for value in resolved.values()):
                continue
            if all(
                getattr(value, f"is_{assumption}", None) in (None, expected)
                for dummy, value in resolved.items()
                for assumption, expected in dummy.assumptions0.items()
            ):
                candidates.append(resolved)
        if len(candidates) != 1:
            snapshots = tuple(
                Candidate(tuple(FieldValue(field, values[field.symbol]) for field in problem.fields))
                for values in reported
            )
            raise SolutionCountError(problem, len(candidates), snapshots)
        solution = candidates[0]
        input_symbols = {c.left for c in problem.constraints if c.kind is ConstraintKind.INPUT}

        def populate(instance: Modello, model: type[Modello], prefix: FieldPath) -> None:
            """Materialize descendants directly from the one shared solution."""
            bound_fields = {d: fields[prefix + path] for path, d in model._modello_fields.items()}
            instance._modello_bound_fields = {
                d: _BoundField(
                    bound, solution[bound], ConstraintKind.INPUT if bound in input_symbols else ConstraintKind.INFERRED
                )
                for d, bound in bound_fields.items()
            }
            for attr in model._modello_namespace.dummies:
                bound = fields[prefix + (attr,)]
                setattr(instance, attr, solution[bound])
            for attr, child in model._modello_namespace.nested_models.items():
                child_instance = object.__new__(child._modello_model)
                populate(child_instance, child._modello_model, (*prefix, attr))
                setattr(instance, attr, child_instance)

        populate(self, type(self), ())


_ModelT = TypeVar("_ModelT", bound=Modello)


def component(model: type[_ModelT]) -> _ModelT:
    """Declare a named child with its instance type in a model class body.

    Use ``child = component(Child)``, then supply a ``Child`` instance when
    constructing the parent model.
    """
    # The class namespace turns this declaration into a scoped reference;
    # populated models expose an instance of this type instead.
    return cast(_ModelT, model)


def inspect_model(model: type[Modello], name: str, /, **value_map: ScalarValue | Modello) -> ModelProblem:
    """Prepare the constructor's actual problem without invoking the solver.

    Supply the same scalar values and typed component instances as a normal
    construction. The snapshot retains all aliases, constraint origins, and
    whole-argument observation groups, including copied component values.
    """
    if not isinstance(model, ModelloMeta):
        raise ModelInputError("inspect_model requires a Modello class and constructor inputs")
    bindings = {d: d.bound(name) for d in dict.fromkeys(model._modello_fields.values())}
    fields = {path: bindings[d] for path, d in model._modello_fields.items()}
    constraints = [
        ModelConstraint(i, path, ConstraintKind.DEFINITION, bindings[d], expr.xreplace(bindings))
        for i, (path, (d, expr)) in enumerate(
            zip(model._modello_constraint_paths, model._modello_class_constraints, strict=True)
        )
    ]

    def add_value(
        path: FieldPath,
        raw: object,
        scope: dict[InstanceDummy, BoundInstanceDummy],
        group: str,
        source: _BoundField | None = None,
    ) -> None:
        try:
            value = simplify(raw)
        except (SympifyError, TypeError, AttributeError) as error:
            raise ModelInputError(f"{'.'.join(path)} must be a SymPy-compatible scalar") from error
        if not isinstance(value, Expr):
            raise ModelInputError(f"{'.'.join(path)} must be a SymPy-compatible scalar")
        kind = ConstraintKind.INPUT if source is None else source.copied_kind(raw)
        constraints.append(ModelConstraint(len(constraints), path, kind, fields[path], value.xreplace(scope), group))

    for attr, value in value_map.items():
        if attr in model._modello_namespace.dummies:
            add_value((attr,), value, bindings, attr)
            continue
        child = model._modello_namespace.nested_models.get(attr)
        if child is None:
            raise ModelInputError(f"Unknown model attribute {attr!r}")
        child_model = child._modello_model
        if not isinstance(value, child_model):
            raise ModelInputError(f"{attr} must be a {child_model.__name__} instance")
        prefix = (attr,)
        source_fields = type(value)._modello_fields
        for path in child_model._modello_fields:
            if path not in source_fields:
                raise ModelInputError(f"Child instance is missing {'.'.join((attr, *path))}")
        scope = {
            **bindings,
            **{d: fields[prefix + path] for path, d in child_model._modello_fields.items()},
            **{
                value._modello_bound_fields[source_fields[path]].symbol: fields[prefix + path]
                for path in child_model._modello_fields
            },
        }
        for path in child_model._modello_fields:
            scalar: object = value
            for part in path:
                scalar = getattr(scalar, part)
            add_value(
                (attr, *path),
                scalar,
                scope,
                attr,
                value._modello_bound_fields[source_fields[path]],
            )
    return ModelProblem(
        model, name, tuple(ModelField(path, symbol) for path, symbol in fields.items()), tuple(constraints)
    )


def _solve_problem(problem: ModelProblem, *, check: bool = True) -> tuple[dict[Symbol, Expr], ...]:
    """Generate joint candidates after expanding directed model definitions.

    Construction and interactive previews share the same equations and ordering.
    Callers choose their candidate checks; an empty result is not itself a proof
    of inconsistency.
    """
    constraints = [Eq(c.left, c.right) for c in problem.constraints]
    if S.false in constraints:
        return ()
    constraints = [eq for eq in constraints if eq is not S.true]
    fields = tuple(dict.fromkeys(field.symbol for field in problem.fields))
    expressions = {symbol: symbol for symbol in fields}
    pending = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.DEFINITION}
    while pending:
        ready = {left: right for left, right in pending.items() if not right.free_symbols & pending.keys()}
        if not ready:
            break  # Cyclic definitions belong in the joint solve.
        for left, right in ready.items():
            expressions[left] = right.xreplace(expressions)
            del pending[left]
    equations = [eq.xreplace(expressions) for eq in constraints]
    if S.false in equations:
        return ()
    equations = [eq for eq in equations if eq is not S.true]
    symbols = list(dict.fromkeys([*pending, *fields]))
    try:
        solutions = solve(equations, symbols, dict=True, check=check) if equations else [{}]
    except Exception as error:
        raise SolverBackendError(problem, error) from error
    return tuple(
        {symbol: expression.xreplace(result) for symbol, expression in expressions.items()} for result in solutions
    )

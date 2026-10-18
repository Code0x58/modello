"""Symbolic models whose named children form a single system of equations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, MutableMapping, cast

from sympy import Basic, Dummy, Eq, Expr, S, SympifyError, solve
from sympy.simplify.simplify import simplify

FieldPath = tuple[str, ...]


class ModelloSentinelClass:
    """Identify models while the metaclass is constructing Modello itself."""


class InstanceDummy(Dummy):
    """A symbol that is rebound independently for each model instance."""

    def bound(self, model_name: str) -> BoundInstanceDummy:
        """Return a fresh symbol with this symbol's assumptions."""
        return BoundInstanceDummy(model_name + "_" + self.name, **self.assumptions0)


class BoundInstanceDummy(InstanceDummy):
    """A symbol belonging to a particular model instance."""


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
        self._replacements: dict[InstanceDummy, InstanceDummy] = {}
        self._missing: dict[InstanceDummy, str] = {}
        # Preserve the existing precedence: later model bases override earlier ones.
        for base in bases:
            parent = getattr(base, "_modello_namespace", None)
            if not isinstance(parent, ModelloMetaNamespace):
                continue
            conflicts = (self.dummies.keys() & parent.nested_models.keys()) | (
                self.nested_models.keys() & parent.dummies.keys()
            )
            if conflicts:
                raise ValueError(f"Cannot merge scalar and nested model fields: {', '.join(sorted(conflicts))}")
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
        is_model = isinstance(value, type) and issubclass(value, ModelloSentinelClass)
        if (isinstance(value, Basic) or is_model) and key.startswith("_modello_"):
            raise ValueError("The _modello_ prefix is reserved for model metadata")
        if isinstance(value, Basic):
            if key in self.nested_models:
                raise ValueError(f"Cannot replace nested model {self.name}.{key} with a scalar")
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
                raise ValueError(f"Cannot replace scalar {self.name}.{key} with a nested model")
            child = _ModelReference._modello_create(cast(type[Modello], value), key)
            self._replace(self._child_replacements(key, child))
            self.nested_models[key] = child
            value = child
        elif isinstance(value, _ModelReference):
            raise TypeError("Declare a component using its model class, not another component reference")
        elif key in self.dummies or key in self.nested_models:
            raise ValueError(f"Cannot assign {self.name}.{key} to a non-model value")
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
        for child in meta_namespace.nested_models.values():
            model = child._modello_model
            replacements = {d: child._modello_fields[path] for path, d in model._modello_fields.items()}
            constraints.extend(
                (replacements[d], expr.xreplace(replacements).xreplace(meta_namespace._replacements))
                for d, expr in model._modello_class_constraints
            )
        expressions = [*meta_namespace.attrs.values(), *(expr for _, expr in constraints)]
        used = set().union(*(expr.free_symbols for expr in expressions))
        for dummy in used & meta_namespace._missing.keys():
            raise ValueError(f"{name} references removed or ambiguous child field {meta_namespace._missing[dummy]}")
        namespace["_modello_namespace"] = meta_namespace
        namespace["_modello_fields"] = meta_namespace.fields()
        namespace["_modello_class_constraints"] = tuple(constraints)
        return super().__new__(mcs, name, bases, namespace)


class Modello(ModelloSentinelClass, metaclass=ModelloMeta):
    """Build a symbolic model, solving its named children together with it."""

    _modello_namespace: ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace("", ())
    _modello_fields: ClassVar[dict[FieldPath, InstanceDummy]] = {}
    _modello_class_constraints: ClassVar[tuple[tuple[InstanceDummy, Basic], ...]] = ()
    _modello_instance_dummies: dict[InstanceDummy, BoundInstanceDummy]
    _modello_instance_constraints: dict[BoundInstanceDummy, Basic]
    _modello_constraints: list[Basic]

    def __init__(self, name: str, **value_map: object) -> None:
        """Accept scalar values and child mappings or compatible model instances."""
        bindings = {d: d.bound(name) for d in dict.fromkeys(self._modello_fields.values())}
        fields = {path: bindings[d] for path, d in self._modello_fields.items()}
        constraints = [Eq(bindings[d], expr.xreplace(bindings)) for d, expr in self._modello_class_constraints]
        inputs: dict[BoundInstanceDummy, Basic] = {}

        def add_value(path: FieldPath, raw: object, scope: dict[InstanceDummy, BoundInstanceDummy]) -> None:
            """Translate a supplied scalar into the shared instance scope."""
            try:
                value = simplify(raw)
            except (SympifyError, TypeError, AttributeError) as error:
                raise TypeError(f"{'.'.join(path)} must be a SymPy-compatible scalar") from error
            if not isinstance(value, Expr):
                raise TypeError(f"{'.'.join(path)} must be a SymPy-compatible scalar")
            value = value.xreplace(scope)
            inputs[fields[path]] = value
            constraints.append(Eq(fields[path], value))

        def collect(model: type[Modello], values: Mapping[str, object] | Modello, prefix: FieldPath) -> None:
            """Validate nested inputs and rebind snapshots of existing children."""
            scope = {**bindings, **{d: fields[prefix + path] for path, d in model._modello_fields.items()}}
            if isinstance(values, Modello):
                if not isinstance(values, model):
                    raise TypeError(f"{'.'.join(prefix)} must be a {model.__name__} instance or a mapping")
                source_fields = type(values)._modello_fields
                for path in model._modello_fields:
                    if path not in source_fields:
                        raise TypeError(f"Child instance is missing {'.'.join((*prefix, *path))}")
                scope.update(
                    (values._modello_instance_dummies[source_fields[path]], fields[prefix + path])
                    for path in model._modello_fields
                )
                for path in model._modello_fields:
                    value: object = values
                    for attr in path:
                        value = getattr(value, attr)
                    add_value((*prefix, *path), value, scope)
                return
            for attr, value in values.items():
                if not isinstance(attr, str):
                    raise TypeError(f"{'.'.join(prefix)} keys must be attribute names")
                path = (*prefix, attr)
                if attr in model._modello_namespace.dummies:
                    add_value(path, value, scope)
                elif attr in model._modello_namespace.nested_models:
                    child_model = model._modello_namespace.nested_models[attr]._modello_model
                    if not isinstance(value, (Mapping, Modello)):
                        raise TypeError(f"{'.'.join(path)} must be a {child_model.__name__} instance or a mapping")
                    collect(child_model, value, path)
                else:
                    raise TypeError(f"Unknown model attribute {'.'.join(path)!r}")

        collect(type(self), value_map, ())
        if S.false in constraints:
            raise ValueError("0 solutions (inconsistent constraints)")
        constraints = [eq for eq in constraints if eq is not S.true]
        # Expand directed definitions first. Otherwise a general solver may use
        # an output as a free parameter and unnecessarily solve for its inputs.
        expressions = {d: d for d in bindings.values()}
        pending = {bindings[d]: expr.xreplace(bindings) for d, expr in self._modello_class_constraints}
        while pending:
            ready = {d: expr for d, expr in pending.items() if not expr.free_symbols & pending.keys()}
            if not ready:
                break  # A cyclic set of definitions still belongs in the joint solve.
            for dummy, expr in ready.items():
                expressions[dummy] = expr.xreplace(expressions)
                del pending[dummy]
        equations = [eq.xreplace(expressions) for eq in constraints]
        if S.false in equations:
            raise ValueError("0 solutions (inconsistent constraints)")
        equations = [eq for eq in equations if eq is not S.true]
        symbols = list(dict.fromkeys([*pending, *bindings.values()]))
        solutions = solve(equations, symbols, dict=True) if equations else [{}]
        candidates = []
        for result in solutions:
            resolved = {d: expr.xreplace(result) for d, expr in expressions.items()}
            if any(value.has(S.NaN, S.ComplexInfinity) for value in resolved.values()):
                continue
            if all(
                getattr(value, f"is_{assumption}", None) in (None, expected)
                for dummy, value in resolved.items()
                for assumption, expected in dummy.assumptions0.items()
            ):
                candidates.append(resolved)
        if len(candidates) != 1:
            raise ValueError(f"{len(candidates)} solutions")
        solution = candidates[0]

        def populate(instance: Modello, model: type[Modello], prefix: FieldPath) -> None:
            """Materialize descendants directly from the one shared solution."""
            instance._modello_instance_dummies = {d: fields[prefix + path] for path, d in model._modello_fields.items()}
            instance._modello_instance_constraints = {
                d: inputs[d] for d in instance._modello_instance_dummies.values() if d in inputs
            }
            instance._modello_constraints = constraints
            for attr in model._modello_namespace.dummies:
                bound = fields[prefix + (attr,)]
                setattr(instance, attr, solution[bound])
            for attr, child in model._modello_namespace.nested_models.items():
                child_instance = object.__new__(child._modello_model)
                populate(child_instance, child._modello_model, (*prefix, attr))
                setattr(instance, attr, child_instance)

        populate(self, type(self), ())

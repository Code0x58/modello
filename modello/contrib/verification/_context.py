"""Bind declarations, assumptions, contracts, and computations to fresh contexts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field as dataclass_field, replace
from functools import cached_property
from itertools import chain
from types import MappingProxyType, new_class
from typing import ClassVar, Generic, TypeVar, cast

from sympy import (
    And,
    Basic,
    Contains,
    Derivative,
    Expr,
    Integral,
    Limit,
    Piecewise,
    Product,
    S,
    Subs,
    Sum,
    Symbol,
    SympifyError,
    default_sort_key,
    preorder_traversal,
    sympify,
)
from sympy.assumptions import global_assumptions
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError

from modello import (
    ConstraintKind,
    Coordinate,
    Modello,
    ModelVectorValue,
    ScalarValue,
    VectorValue,
    annotate,
    annotate_model,
    component,
    inspect_model,
)
from modello._collections import _Selection
from modello._evaluation import _evaluate
from modello._expressions import _definition_resolver, _DefinitionResolver
from modello._model import _ModelReference
from modello._operators import _bound_coordinates, _empty_reduction, _scalar_atoms, _ScopedExpression, _SymbolicTemplate
from modello._problem import ModelConstraint
from modello._reconstruction import _BINDERS, _expression_key, _predicate_parts, _rebind
from modello._symbols import BoundInstanceDummy
from modello.diagnostics import ModelProblem
from modello.errors import ModelDefinitionError, ModelInputError, ModelValidationError
from modello.series import Recurrence, _project_series, _SeriesTerm
from modello.solutions import SolutionFamily, _field_domain, _membership

from ._proof import (
    Attempt,
    _defined,
    _DerivativePremises,
    attempt,
    captures_coordinate,
    conjunction,
    consistency,
    coordinate_domain,
    derivative_premises,
    evaluate_native,
    guard_source,
    native_domains,
    scoped_source,
    substitution_points,
    truth,
)
from ._records import (
    Evidence,
    InferenceRule,
    Method,
    Obligation,
    ResultT,
    Sample,
    VerificationError,
    VerificationReport,
    VerificationResult,
    _Obligations,
    differentiable,
    nonzero,
    predicate,
)

ModelT = TypeVar("ModelT", bound=Modello)
ValueT = TypeVar("ValueT")
InputKeyT = TypeVar("InputKeyT")
_InputValue = ScalarValue | VectorValue | Modello | ModelVectorValue[Modello]


@dataclass(frozen=True)
class _ModelValues:
    """Resolve the existing typed field namespace without selecting a solution."""

    _reference: _ModelReference | _Selection
    _expression: Callable[[Basic], Basic]

    def _resolved(self, value):
        if isinstance(value, (_ModelReference, _Selection)):
            return _ModelValues(value, self._expression)
        if isinstance(value, tuple):
            return tuple(self._resolved(item) for item in value)
        if isinstance(value, Basic):
            return self._expression(value)
        if callable(value):
            return lambda *args, **kwargs: self._resolved(value(*args, **kwargs))
        return value

    def __getattr__(self, name: str):
        reference = self.__dict__.get("_reference")
        if reference is None:
            raise AttributeError(name)
        return self._resolved(getattr(reference, name))

    def __getitem__(self, index: object):
        return self._resolved(self._reference[index])

    def __len__(self) -> int:
        return len(self._reference)

    def __bool__(self) -> bool:
        return bool(self._reference)

    def __iter__(self):
        return (self[index] for index in range(len(self)))

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._reference._modello_model._modello_namespace.keys())


def _unique_expressions(values: Iterable[Basic]) -> tuple[Basic, ...]:
    """Deduplicate verification premises without merging substitution scopes."""
    return tuple({_expression_key(value): value for value in values}.values())


@dataclass
class _Context:
    """Resolved work for one context under one set of global assumptions."""

    problem: ModelProblem
    assumptions: tuple[Basic, ...]
    parameters: dict[Symbol, Expr] = dataclass_field(default_factory=dict)
    expressions: dict[tuple, tuple[Basic, tuple[Basic, ...]]] = dataclass_field(default_factory=dict)
    attempts: dict[tuple, Attempt] = dataclass_field(default_factory=dict)

    @cached_property
    def law(self) -> _DefinitionResolver:
        definitions: dict[Symbol, Basic] = {}
        ambiguous: set[Symbol] = set()
        for item in self.problem.constraints:
            if item.kind is ConstraintKind.DEFINITION:
                value = guard_source(item.right)
                if item.left in definitions and _expression_key(definitions[item.left]) != _expression_key(value):
                    ambiguous.add(item.left)
                definitions[item.left] = value
        return _definition_resolver(
            definitions,
            frozenset(ambiguous),
            templates=self.problem.templates.values(),
            evaluate_operators=False,
        )

    @cached_property
    def needs_solutions(self) -> bool:
        if (
            self.law.cyclic
            or self.law.ambiguous
            or any(item.kind is not ConstraintKind.DEFINITION for item in self.problem.constraints)
        ):
            return True
        parameters = tuple(dict.fromkeys(field.symbol for field in self.problem.fields))
        try:
            for field in self.problem.fields:
                domain = _field_domain(field.symbol)
                value = self.law(field.symbol)
                intrinsic = domain is not S.Complexes or field.symbol.is_complex is True
                if not intrinsic and value.has(*_BINDERS):
                    if value.free_symbols:
                        continue
                    value = _evaluate(value)
                    if value.has(*_BINDERS):
                        return True
                membership = _membership(value, domain, parameters)
                if membership is S.false or intrinsic and membership is not S.true:
                    return True
            return False
        except (NotImplementedError, TypeError, ValueError):
            return True

    @cached_property
    def bindings(self) -> _DefinitionResolver:
        """Keep directed observation formulas until their operations are checked."""
        defined = {item.left for item in self.problem.constraints if item.kind is ConstraintKind.DEFINITION}
        inputs: dict[Symbol, list[Basic]] = {}
        for item in self.problem.constraints:
            if item.kind is not ConstraintKind.DEFINITION and item.left not in defined:
                inputs.setdefault(item.left, []).append(item.right)
        direct = {}
        for symbol, observations in inputs.items():
            values = _unique_expressions(observations)
            if len(values) == 1:
                direct[symbol] = guard_source(values[0])
        if not direct:
            return self.law
        source = _definition_resolver(
            {**self.law.definitions, **direct},
            self.law.ambiguous,
            templates=self.law.templates,
            evaluate_operators=False,
        )
        # Cyclic and conflicting inputs remain equations for the joint solver.
        # Retain the original laws even when they depend on such an input.
        return _DefinitionResolver(
            {
                **self.law.definitions,
                **{symbol: value for symbol, value in source.definitions.items() if symbol in direct},
            },
            self.law.cyclic,
            self.law.ambiguous,
            {},
            source.templates,
            evaluate_operators=False,
        )

    @cached_property
    def residual_sources(self) -> tuple[ModelConstraint, ...]:
        """Sources the directed bindings leave for joint solving."""
        direct = self.bindings.definitions.keys() - self.law.definitions.keys()
        return tuple(
            item
            for item in self.problem.constraints
            if item.kind is not ConstraintKind.DEFINITION
            and item.left not in direct
            or item.kind is ConstraintKind.DEFINITION
            and item.left in self.law.cyclic | self.law.ambiguous
        )

    @cached_property
    def family(self) -> SolutionFamily:
        return self.problem.solutions().where(*self.assumptions)


def _schema(model: type[Modello], path: tuple[str, ...]) -> type[Modello]:
    for part in path:
        reference = model._modello_namespace.nested_models.get(part)
        if reference is None:
            raise ModelDefinitionError(f"Unknown component path: {'.'.join(path)}")
        model = reference._modello_model
    return model


def _coordinates(model: type[Modello], prefix=""):
    attributes = {name: value for base in reversed(model.__mro__) for name, value in vars(base).items()}
    result = {prefix + name: value for name, value in sorted(attributes.items()) if isinstance(value, Coordinate)}
    for name, child in model._modello_namespace.nested_models.items():
        result.update(_coordinates(child._modello_model, prefix + name + "."))
    return result


def _scoped_symbols(value: Basic, *, include_declarations: bool = False) -> set[Basic]:
    """Track scalar scopes without borrowing a named declaration's binders."""
    symbols = set(value.free_symbols)
    operators = preorder_traversal(value)
    for operator in operators:
        if not include_declarations and isinstance(operator, _SymbolicTemplate):
            operators.skip()
            continue
        if isinstance(operator, _BINDERS) and (
            include_declarations or not isinstance(operator, _ScopedExpression) or operator.scoped_arguments
        ):
            for variable in _bound_coordinates(operator):
                symbols.update(variable.free_symbols | {variable})
    return symbols


def _replace_component(model: type[Modello], path: tuple[str, ...], replacement: type[Modello]) -> type[Modello]:
    if not path or not isinstance(replacement, type) or not issubclass(replacement, Modello):
        raise ModelDefinitionError("Replacement needs a component path and a Modello class")
    old_child = _schema(model, path[:1])
    if len(path) > 1:
        replacement = _replace_component(old_child, path[1:], replacement)
    old_coordinates, new_coordinates = _coordinates(old_child), _coordinates(replacement)
    changed_kind = old_coordinates.keys() & {
        ".".join(field_path) for field_path in replacement._modello_definition.fields
    }
    if changed_kind:
        raise ModelDefinitionError(
            "Replacement changes coordinate paths to model fields: " + ", ".join(sorted(changed_kind))
        )
    coordinate_scopes: dict[Coordinate, dict[str, Coordinate | None]] = {}
    for name, original in old_coordinates.items():
        by_scope = coordinate_scopes.setdefault(original, {})
        scope = name.rpartition(".")[0]
        by_scope.setdefault(scope, None)
        if name not in new_coordinates:
            continue
        target = new_coordinates[name]
        if by_scope[scope] is not None and by_scope[scope] != target:
            raise ModelDefinitionError("Replacement splits aliased coordinates; declare a new model and context")
        by_scope[scope] = target
    own_coordinates = {name: value for name, value in _coordinates(model).items() if "." not in name}
    namespace = model._modello_namespace
    sources = (*namespace.attrs.values(), *namespace.vector_attrs.values(), *namespace.templates.values())
    used = set().union(*(_scoped_symbols(value) for value in sources))
    for annotation in namespace.annotations.get(_Obligations, {}).values():
        if isinstance(annotation, _Obligations):
            for claim in annotation.obligations:
                for value in (claim.subject, claim.wrt, claim.domain, claim.bound, claim.expected, claim.at):
                    if isinstance(value, Basic):
                        used.update(_scoped_symbols(value))
    used.update(own_coordinates.values())
    needed = {}
    for original, scopes in coordinate_scopes.items():
        if original not in used:
            continue
        for scope, target in scopes.items():
            if target is None:
                if not any(
                    value == original and name.rpartition(".")[0] == scope for name, value in new_coordinates.items()
                ):
                    raise ModelDefinitionError(
                        "Replacement removed a coordinate used by inherited expressions; "
                        "declare a new model and context"
                    )
                scopes[scope] = original
        targets = set(scopes.values())
        if original in new_coordinates.values():
            targets.add(original)
        if len(targets) != 1:
            raise ModelDefinitionError(
                "Replacement changes a shared parent coordinate; declare its argument join explicitly"
            )
        target = next(iter(targets))
        if target != original:
            needed[original] = target
    other_coordinates = {
        value
        for name, child in model._modello_namespace.nested_models.items()
        if name != path[0]
        for value in _coordinates(child._modello_model).values()
    }
    if set(needed) & other_coordinates:
        raise ModelDefinitionError(
            "Replacement changes a shared parent coordinate; declare its argument join explicitly"
        )

    def declare(namespace):
        if needed:
            namespace._replace(needed)
            for name, value in own_coordinates.items():
                if value in needed:
                    namespace[name] = needed[value]
        namespace[path[0]] = component(replacement)

    return new_class(
        f"{model.__name__}With{replacement.__name__}",
        (model,),
        exec_body=declare,
    )


@dataclass(frozen=True)
class VerificationContext(Generic[ModelT]):
    """Immutable definitions, observations, assumptions, and proof obligations.

    Construction does not solve the model or establish its obligations.
    """

    problem: ModelProblem[ModelT]
    assumptions: tuple[Basic, ...] = ()
    obligations: tuple[Obligation, ...] = ()
    bound_hints: tuple[tuple[str, Expr], ...] = ()
    rules: tuple[InferenceRule, ...] = ()
    changes: tuple[str, ...] = ()

    @classmethod
    def for_model(cls, model: type[ModelT], name: str = "verification", **observations) -> VerificationContext[ModelT]:
        return cls(inspect_model(model, name, **observations))

    @property
    def symbols(self) -> ModelT:
        return self.problem.symbols

    @cached_property
    def model(self) -> ModelT:
        """Read resolved fields through their declared model and component types.

        This read-only namespace has the same fields as ``symbols``. It retains
        unresolved parameters and held operations, without selecting a solution
        or establishing proof obligations. Use ``symbols`` to select inputs.
        """
        return cast(ModelT, _ModelValues(cast(_ModelReference, self.symbols), self.expression))

    @property
    def family(self) -> SolutionFamily[ModelT]:
        """The existing exact solution family, evaluated only on explicit demand."""
        return self._context.family

    @property
    def _context(self) -> _Context:
        key = _expression_key(frozenset(global_assumptions))
        if key not in self._contexts:
            self._contexts[key] = _Context(self.problem, self.assumptions)
        return self._contexts[key]

    @cached_property
    def _contexts(self) -> dict[frozenset[object], _Context]:
        return {}

    @property
    def _law(self):
        """Expand definitions without observations for source-domain guards."""
        return self._context.law

    @cached_property
    def _fields(self):
        return {".".join(field.path): field.symbol for field in self.problem.fields}

    @cached_property
    def _references(self) -> dict[str, Expr]:
        return {**_coordinates(self.problem.model), **self._fields}

    @cached_property
    def _field_paths(self) -> dict[Symbol, str]:
        """Choose one observable path for each identity, including aliases."""
        labels: dict[Symbol, str] = {}
        for path, symbol in self._fields.items():
            previous = labels.get(symbol)
            if previous is None or (path.count("."), len(path)) < (previous.count("."), len(previous)):
                labels[symbol] = path
        return labels

    @cached_property
    def _labels(self) -> tuple[tuple[Symbol, str], ...]:
        labels = dict(self._field_paths)
        paths: dict[Coordinate, str] = {}
        for path, coordinate in self._references.items():
            if not isinstance(coordinate, Coordinate):
                continue
            previous = paths.get(coordinate)
            if previous is None or (path.count("."), len(path), path) < (previous.count("."), len(previous), previous):
                paths[coordinate] = path
        for constraint in self.problem.constraints:
            for coordinate in sorted(
                constraint.right.atoms(Coordinate), key=lambda value: (value.name, value.dummy_index)
            ):
                paths.setdefault(coordinate, ".".join((*constraint.path, coordinate.name)))
        counts = Counter(coordinate.name for coordinate in paths)
        occupied = set(labels.values())
        reserved = set(paths.values())
        for coordinate, path in sorted(paths.items(), key=lambda item: (item[1], item[0].dummy_index)):
            if coordinate in labels:
                continue
            label = coordinate.name
            if counts[label] > 1 or label in occupied or label in reserved and label != path:
                label = path
            base, suffix = label, 2
            while label in occupied:
                label = f"{base}#{suffix}"
                suffix += 1
            labels[coordinate] = label
            occupied.add(label)
        return tuple(labels.items())

    def _reference(self, path: str) -> Expr:
        try:
            return self._references[path]
        except KeyError:
            raise ModelDefinitionError(f"Unknown scalar or coordinate path: {path}") from None

    @cached_property
    def _known_symbols(self) -> frozenset[Basic]:
        allowed: set[Basic] = set(self._references.values())
        for item in self.problem.constraints:
            allowed.update(item.symbols | _scoped_symbols(item.right, include_declarations=True))
        return frozenset(allowed)

    def _require_symbols(self, value: Basic) -> None:
        symbols = value.free_symbols | {
            symbol for symbol in _scoped_symbols(value) if isinstance(symbol, BoundInstanceDummy)
        }
        if symbols - self._known_symbols:
            raise ModelValidationError("Use this verification context's symbols or coordinates from its declaration")

    @cached_property
    def _changed_coordinate_scopes(self) -> frozenset[Coordinate]:
        """Coordinates captured from argument paths changed by model inheritance."""
        current = _coordinates(self.problem.model)
        changed: set[Coordinate] = set()
        pending: list[tuple[str, type[Modello]]] = [("", self.problem.model)]
        while pending:
            prefix, model = pending.pop()
            for base in model.__mro__[1:]:
                if base is Modello or not issubclass(base, Modello):
                    continue
                for path, coordinate in _coordinates(base).items():
                    path = prefix + path
                    if current.get(path) == coordinate:
                        continue
                    if path not in current and any(
                        value == coordinate and name.rpartition(".")[0] == path.rpartition(".")[0]
                        for name, value in current.items()
                    ):
                        continue
                    changed.add(coordinate)
            pending.extend(
                (prefix + name + ".", child._modello_model)
                for name, child in model._modello_namespace.nested_models.items()
            )
        return frozenset(changed)

    def _coordinate(self, value: str | Symbol) -> Symbol:
        """Validate against definitions without specializing an observed state."""
        coordinate = self._reference(value) if isinstance(value, str) else value
        if not isinstance(coordinate, Symbol):
            raise ModelDefinitionError("wrt must select an independent model symbol or declared coordinate")
        self._require_symbols(coordinate)
        law = self._law
        if coordinate in law.cyclic or law(coordinate) != coordinate:
            raise ModelDefinitionError(
                "wrt must select an independent model symbol or declared coordinate; "
                "reparameterize derived quantities explicitly"
            )
        return coordinate

    def _annotation_bindings(self, prefix: str):
        """Bind native metadata expressions in their declaring component scope."""
        scope = _schema(self.problem.model, tuple(prefix.split(".")) if prefix else ())
        path_prefix = prefix + "." if prefix else ""
        fields = {
            dummy: self._reference(path_prefix + ".".join(path))
            for path, dummy in scope._modello_definition.fields.items()
        }
        mapping = {old: value.xreplace(fields) for old, value in scope._modello_namespace._replacements.items()}
        mapping.update(fields)
        for base in scope.__mro__:
            for name, coordinate in vars(base).items():
                target = self._references.get(path_prefix + name)
                if isinstance(coordinate, Coordinate) and target is not None:
                    mapping[coordinate] = target
        return mapping

    def _bind(self, value: Basic, protected: frozenset[Basic] = frozenset()) -> Basic:
        context = self._context
        key = (_expression_key(value), protected)
        if key in context.expressions:
            return context.expressions[key][0]
        result = context.bindings(guard_source(value), protected)
        sources: tuple[Basic, ...] = ()
        if context.needs_solutions:
            dependencies = {self._fields[path] for path in self._dependencies(value)}
            sources = tuple(
                context.law(guard_source(item.right), protected)
                for item in context.residual_sources
                if item.left in dependencies
            )
            if protected:
                scoped_sources = tuple(context.bindings(source, protected) for source in sources)
                if any(protected & source.free_symbols for source in scoped_sources):
                    # A joint state solution must not freeze an implicit curve,
                    # including one reached through observation equations.
                    for source in (result, *scoped_sources):
                        context.law._check_calculus(source, protected)
            # Bind free parameters individually. Resolving the entire curve
            # through a specialized family would also substitute its coordinate.
            replacements = {}
            for symbol in set().union(*(item.free_symbols for item in (result, *sources))) - protected:
                if symbol in self._fields.values():
                    if symbol not in context.parameters:
                        context.parameters[symbol] = context.family.expression(symbol)
                    resolved = context.parameters[symbol]
                    if resolved != symbol:
                        replacements[symbol] = guard_source(resolved)
            # Preserve held operations even when observed coefficients are zero;
            # the proof walker must check their original scopes before evaluation.
            resolve = _DefinitionResolver({}, frozenset(), frozenset(), replacements, evaluate_operators=False)

            def bind(expression):
                expression = resolve(expression, protected)
                points = sorted(
                    set(replacements)
                    & expression.free_symbols
                    & {symbol for derivative in expression.atoms(Derivative) for symbol in derivative.variables},
                    key=default_sort_key,
                )
                if points:
                    expression = Subs(expression, tuple(points), tuple(replacements[symbol] for symbol in points))
                return expression

            result = bind(result)
            sources = tuple(bind(source) for source in sources)
        context.expressions[key] = result, sources
        return result

    def _independent_binding(
        self, value: Basic, protected: frozenset[Basic], patterns: frozenset[Basic]
    ) -> Basic | None:
        """Resolve a value only when native patterns cannot affect it or its sources."""
        scoped = protected | patterns
        resolved = self._bind(value, scoped)
        _, operands = self._context.expressions[_expression_key(value), scoped]
        dependencies = resolved.free_symbols | set().union(*(operand.free_symbols for operand in operands))
        return None if patterns & dependencies else resolved

    def expression(self, subject: str | Basic, *, wrt: str | Symbol | None = None) -> Expr:
        """Resolve a law, preserving an independent ``wrt`` symbol from observations."""
        variable = self._coordinate(wrt) if wrt is not None else None
        value = self._reference(subject) if isinstance(subject, str) else subject
        self._require_symbols(value)
        return cast(Expr, self._bind(value, frozenset() if variable is None else frozenset({variable})))

    def assume(self, *conditions) -> VerificationContext[ModelT]:
        """Add logical premises on a new context, retaining their native scopes."""
        checked = []
        for value in conditions:
            condition = sympify(value)
            if not isinstance(condition, Boolean) or isinstance(condition, Expr):
                raise ModelInputError("Assumptions must be symbolic predicates; use Eq for equalities")
            self._require_symbols(condition)
            checked.append(condition)
        return replace(self, assumptions=(*self.assumptions, *checked))

    def _component_path(self, reference: str | Modello) -> str:
        if isinstance(reference, str):
            return reference
        if not isinstance(reference, _ModelReference) or not reference._modello_path:
            raise ModelDefinitionError("Select a component through this verification context's symbols")
        if reference._modello_scope != self.problem._scope:
            raise ModelValidationError("Use this verification context's component references")
        return ".".join(reference._modello_path)

    def _input_path(self, reference: object) -> str:
        if isinstance(reference, str):
            return reference
        if isinstance(reference, _ModelReference):
            return self._component_path(cast(Modello, reference))
        if isinstance(reference, Symbol):
            path = self._field_paths.get(reference)
            if path is not None:
                return path
            self._require_symbols(reference)
        raise ModelInputError("Observe a scalar or component selected through this verification context's symbols")

    def observe(
        self,
        observations: Mapping[InputKeyT, _InputValue] | None = None,
        /,
        **inputs: _InputValue,
    ) -> VerificationContext[ModelT]:
        """Bind reference-keyed observations or constructor-style inputs immutably."""
        if observations is not None and not isinstance(observations, Mapping):
            raise ModelInputError("Pass a mapping from model references to observed values")
        values: dict[str, _InputValue] = {}
        for reference, value in chain((observations or {}).items(), inputs.items()):
            path = self._input_path(reference)
            if path in values:
                raise ModelInputError(f"More than one observation selects {path!r}")
            values[path] = value
        return replace(self, problem=self.problem._capture_inputs(**values))

    def without_assumptions(self, *conditions: Basic, reason: str) -> VerificationContext[ModelT]:
        """Remove explicitly recorded assumptions, retaining why they changed.

        Declaration domains and observations are unaffected. Each new report
        checks every dependent conclusion again under the remaining premises.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ModelInputError("Record a reason when removing assumptions")
        removed = _unique_expressions(sympify(condition) for condition in conditions)
        recorded = {_expression_key(condition) for condition in self.assumptions}
        missing = tuple(condition for condition in removed if _expression_key(condition) not in recorded)
        if missing:
            raise ModelInputError(f"These assumptions were not recorded in this context: {missing}")
        removed_keys = {_expression_key(condition) for condition in removed}
        return replace(
            self,
            assumptions=tuple(
                condition for condition in self.assumptions if _expression_key(condition) not in removed_keys
            ),
            changes=(*self.changes, *(f"Removed assumption {condition}: {reason.strip()}" for condition in removed)),
        )

    def with_obligations(self, *obligations: Obligation) -> VerificationContext[ModelT]:
        """Add proof goals without treating them as assumptions."""
        if any(not isinstance(item, Obligation) for item in obligations):
            raise ModelDefinitionError("with_obligations accepts Obligation records")
        return replace(self, obligations=(*self.obligations, *obligations))

    def without_obligations(self, *obligation_ids: str, reason: str) -> VerificationContext[ModelT]:
        """Remove context-added goals; method and declaration obligations remain.

        Change a declaration or component to change its annotations. Automatic
        obligations and a method's own preconditions cannot be waived here.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ModelInputError("Record a reason when removing obligations")
        removed = tuple(dict.fromkeys(obligation_ids))
        available = {item.name for item in self.obligations}
        missing = tuple(key for key in removed if key not in available)
        if missing:
            raise ModelInputError(f"Only context-added obligations can be removed; unknown context IDs: {missing}")
        return replace(
            self,
            obligations=tuple(item for item in self.obligations if item.name not in removed),
            changes=(*self.changes, *(f"Removed obligation {key}: {reason.strip()}" for key in removed)),
        )

    def with_bound_hint(self, obligation_id: str, *, upper) -> VerificationContext[ModelT]:
        """Propose a uniform upper bound; every check must establish it."""
        try:
            value = sympify(upper)
        except (SympifyError, TypeError) as error:
            raise ModelInputError("An upper-bound hint must be a scalar expression") from error
        if not isinstance(value, Expr):
            raise ModelInputError("An upper-bound hint must be a scalar expression")
        return replace(self, bound_hints=tuple({**dict(self.bound_hints), obligation_id: value}.items()))

    def with_rule(self, rule: InferenceRule) -> VerificationContext[ModelT]:
        """Register a trusted inference rule whose premises must be verified."""
        if not rule.source or any(item.name == rule.name for item in self.rules):
            raise ModelDefinitionError("Inference rules need a source and a unique rule name")
        return replace(self, rules=(*self.rules, rule))

    def replace(self, component_path: str | Modello, model: type[Modello]) -> VerificationContext[ModelT]:
        component_path = self._component_path(component_path)
        declaration = _replace_component(self.problem.model, tuple(component_path.split(".")), model)
        problem = inspect_model(cast(type[ModelT], declaration), self.problem.name)
        mapping: dict[Basic, Basic] = {}
        for originals, paths, kind in (
            (
                {field.path: field.symbol for field in self.problem.fields},
                {field.path: field.symbol for field in problem.fields},
                "field",
            ),
            (self.problem.templates, problem.templates, "declaration"),
        ):
            for path, original in originals.items():
                if path not in paths:
                    raise ModelDefinitionError(f"Replacement removed a recorded {kind}: {'.'.join(path)}")
                target = paths[path]
                if isinstance(original, _SymbolicTemplate) and (
                    not isinstance(target, _SymbolicTemplate)
                    or original._binding_signature != target._binding_signature
                ):
                    raise ModelDefinitionError(
                        f"Replacement changes a declaration's argument signature: {'.'.join(path)}"
                    )
                if isinstance(original, _SymbolicTemplate) and isinstance(target, _SymbolicTemplate):
                    split = any(
                        isinstance(previous, _SymbolicTemplate)
                        and original._same_declaration(previous)
                        and (not isinstance(mapped, _SymbolicTemplate) or not target._same_declaration(mapped))
                        for previous, mapped in mapping.items()
                    )
                else:
                    split = original in mapping and mapping[original] != target
                if split:
                    raise ModelDefinitionError(
                        "Replacement splits aliased fields or declarations; declare a new model and context"
                    )
                mapping[original] = target
        old_coordinates = _coordinates(self.problem.model)
        new_coordinates = _coordinates(declaration)
        changed_coordinates = {
            name: (old, new_coordinates.get(name))
            for name, old in old_coordinates.items()
            if old != new_coordinates.get(name)
        }
        changed_symbols = {old for old, _ in changed_coordinates.values()}
        removed_paths = old_coordinates.keys() - new_coordinates.keys()
        unmapped = (
            set(old_coordinates.values())
            - {old for name, old in old_coordinates.items() if name in new_coordinates}
            - set(new_coordinates.values())
        )

        def coordinate_references(*subjects):
            referenced = set()
            for value in subjects:
                if isinstance(value, str) and value in removed_paths:
                    raise ModelDefinitionError(f"Replacement removed a referenced coordinate path: {value}")
                if isinstance(value, Basic):
                    referenced.update(_scoped_symbols(value))
            if referenced & unmapped:
                raise ModelDefinitionError(
                    "Replacement removed a referenced coordinate; declare its argument scope explicitly"
                )
            return referenced

        def coordinate_bindings(*subjects):
            referenced = coordinate_references(*subjects)
            if not referenced & changed_symbols:
                return {}
            dependencies = tuple(
                dict.fromkeys(
                    path
                    for subject in subjects
                    if isinstance(subject, (str, Basic))
                    for path in self._dependencies(self._reference(subject) if isinstance(subject, str) else subject)
                )
            )
            selected = {}
            for name, (old, new) in changed_coordinates.items():
                if old not in referenced:
                    continue
                coordinate_dependencies = [
                    path
                    for path in dependencies
                    if old in _scoped_symbols(self._law(self._reference(path), frozenset({old})))
                ]
                scope = name.rpartition(".")[0]
                unscoped = old in referenced and not coordinate_dependencies
                relevant = (
                    not dependencies
                    or unscoped
                    or any(path == scope or path.startswith(scope + ".") for path in coordinate_dependencies)
                )
                if relevant:
                    if new is None:
                        targets = {
                            new_coordinates[path]
                            for path, value in old_coordinates.items()
                            if value == old and path in new_coordinates and path.rpartition(".")[0] == scope
                        }
                        if any(
                            value == old and path.rpartition(".")[0] == scope for path, value in new_coordinates.items()
                        ):
                            targets.add(old)
                        if not targets:
                            raise ModelDefinitionError(
                                "Replacement removed a referenced coordinate; redeclare its argument scope"
                            )
                        new = next(iter(targets))
                    if new == old:
                        continue
                    unchanged = [path for path, value in new_coordinates.items() if value == old]
                    if (
                        unchanged
                        and unscoped
                        or any(
                            any(
                                dependency.startswith(path.rpartition(".")[0] + ".")
                                for dependency in coordinate_dependencies
                            )
                            for path in unchanged
                            if path.rpartition(".")[0]
                        )
                    ):
                        raise ModelDefinitionError(
                            "A native claim joins renamed and unchanged coordinates; redeclare its scope"
                        )
                    if old in selected and selected[old] != new:
                        raise ModelDefinitionError(
                            "Replacement splits a shared coordinate; redeclare the argument scopes"
                        )
                    selected[old] = new
            return selected

        for annotation_path, annotation in _Obligations.for_model(declaration).items():
            prefix = annotation_path if annotation.model_level else annotation_path.rpartition(".")[0]
            for claim in annotation.obligations:
                coordinate_references(
                    *(
                        f"{prefix}.{value}" if prefix and isinstance(value, str) else value
                        for value in (claim.subject, claim.wrt, claim.domain, claim.bound, claim.expected, claim.at)
                    )
                )
        assumptions = []
        for value in self.assumptions:
            for old in coordinate_references(value) & changed_symbols:
                if old in new_coordinates.values():
                    raise ModelDefinitionError(
                        "An assumption uses a shared renamed coordinate; refine its scope explicitly"
                    )
            bindings = coordinate_bindings(value)
            assumptions.append(_rebind(value, {**mapping, **bindings}))
        observations: list[ModelConstraint] = []
        invalidated = []
        for constraint in self.problem.constraints:
            if constraint.kind is not ConstraintKind.DEFINITION:
                # Inverse-inferred values depend on the constructor's whole
                # equation system, rather than only on their directed laws.
                if constraint.kind is ConstraintKind.INFERRED and (
                    constraint.input is not None
                    and (component_path == constraint.input or component_path.startswith(constraint.input + "."))
                    or any(
                        path == component_path or path.startswith(component_path + ".")
                        for path in self._dependencies(constraint.left, include_declarations=True)
                    )
                ):
                    invalidated.append(
                        "Invalidated inferred observation "
                        + ".".join(constraint.path)
                        + " after replacing "
                        + component_path
                        + "."
                    )
                    continue
                observations.append(
                    replace(
                        constraint,
                        index=len(problem.constraints) + len(observations),
                        left=mapping[constraint.left],
                        right=_rebind(
                            constraint.right,
                            {**mapping, **coordinate_bindings(constraint.left, constraint.right)},
                        ),
                    )
                )
        problem = replace(problem, constraints=(*problem.constraints, *observations))

        def rebound(claim):
            payloads = (claim.subject, claim.wrt, claim.domain, claim.bound, claim.expected, claim.at)
            bindings = {**mapping, **coordinate_bindings(*payloads)}

            def value(item):
                return _rebind(item, bindings) if isinstance(item, Basic) else item

            return replace(
                claim,
                subject=value(claim.subject),
                wrt=value(claim.wrt),
                expected=value(claim.expected),
                domain=value(claim.domain),
                bound=value(claim.bound),
                at=value(claim.at),
            )

        return VerificationContext(
            problem,
            tuple(assumptions),
            tuple(rebound(item) for item in self.obligations),
            tuple((key, _rebind(hint, {**mapping, **coordinate_bindings(hint)})) for key, hint in self.bound_hints),
            tuple(
                replace(
                    rule,
                    conclusion=rebound(rule.conclusion),
                    premises=tuple(rebound(item) for item in rule.premises),
                )
                for rule in self.rules
            ),
            (*self.changes, *invalidated),
        )

    def _dependencies(self, value: Basic, *, include_declarations: bool = False) -> tuple[str, ...]:
        templates = tuple(self.problem.templates.values())

        def references(expression, laws=templates):
            result = expression.free_symbols | expression.atoms(_SymbolicTemplate)
            for term in _scalar_atoms(expression, _SeriesTerm):
                try:
                    projected = _project_series(term, laws)
                except ModelInputError:
                    # Dependency inspection must not choose an ambiguous law
                    # or replace the proof engine's unresolved evidence.
                    continue
                result.update(projected.free_symbols | projected.atoms(_SymbolicTemplate))
            return result

        symbols: set[Basic] = set()
        pending = list(references(value))
        sources: dict[Basic, list[Basic]] = {}
        for item in self.problem.constraints:
            sources.setdefault(item.left, []).append(item.right)
        while pending:
            symbol = pending.pop()
            if symbol in symbols:
                continue
            symbols.add(symbol)
            if isinstance(symbol, _SymbolicTemplate):
                laws = templates
                if isinstance(symbol, Recurrence):
                    # An explicit recurrence selects itself; only foreign
                    # series queries use the registry's ambiguity policy.
                    laws = (
                        *(
                            law
                            for law in templates
                            if not isinstance(law, Recurrence)
                            or law.target.series.symbol != symbol.target.series.symbol
                        ),
                        symbol,
                    )
                for expression in (symbol, *symbol.equations):
                    pending.extend(references(expression, laws) - set(symbol.protected_symbols))
            else:
                for expression in sources.get(symbol, ()):
                    pending.extend(references(expression))
        paths = [name for name, symbol in self._fields.items() if symbol in symbols]
        if include_declarations:
            reached = tuple(symbol for symbol in symbols if isinstance(symbol, _SymbolicTemplate))
            paths.extend(
                ".".join(path)
                for path, template in self.problem.templates.items()
                if any(template._same_declaration(symbol) for symbol in reached)
            )
        return tuple(paths)

    def _declared(self):
        records = []
        for path, annotation in _Obligations.for_model(self.problem.model).items():
            prefix = path if annotation.model_level else path.rpartition(".")[0]
            for obligation in annotation.obligations:
                key = f"{path}/{obligation.name}" if path else obligation.name
                records.append((key, obligation, prefix))
        records.extend((item.name, item, "") for item in self.obligations)
        return records

    def _automatic(self, assumptions: tuple[Basic, ...]) -> tuple[list[tuple[str, Obligation, str]], frozenset[str]]:
        records: list[tuple[str, Obligation, str]] = []
        conditional: set[str] = set()
        sources: dict[Symbol, list[Basic]] = {}
        for constraint in self.problem.constraints:
            sources.setdefault(constraint.left, []).append(constraint.right)
        quantified: set[Basic] = set().union(
            *(value.atoms(Coordinate) for values in sources.values() for value in values)
        )
        quantified.update(
            variable
            for values in sources.values()
            for value in values
            for operator in value.atoms(*_BINDERS)
            for variable in _bound_coordinates(operator)
        )
        seen_by_path: dict[str, dict[tuple, str]] = {}
        for constraint in self.problem.constraints:
            path = ".".join(constraint.path)
            seen = seen_by_path.setdefault(path, {})

            def record(claim, reason, uncertain):
                proposition = claim._proposition()
                key = seen.get(proposition)
                if key is None:
                    key = f"{path}/auto:{len(seen) + 1}"
                    seen[proposition] = key
                    records.append((key, replace(claim, consumer=path, reason=reason), ""))
                    if uncertain:
                        conditional.add(key)
                elif not uncertain:
                    conditional.discard(key)

            def visit(value, points=None, bound=(), active=(), uncertain=False, scopes=None, native=()):
                points = points or {}
                scopes = {} if scopes is None else scopes
                compound = any(
                    isinstance(scope, Subs) and any(not isinstance(symbol, Symbol) for symbol in scope.variables)
                    for scope in native
                )
                if isinstance(value, (Integral, Sum, Product, Limit)):
                    variables = tuple(_bound_coordinates(value))
                    protected = frozenset((*quantified, *bound))
                    if compound:
                        domains = dict.fromkeys(variables, S.Complexes)
                        targets = tuple(
                            symbol for scope in native if isinstance(scope, Subs) for symbol in scope.variables
                        )
                        patterns = frozenset().union(*(symbol.free_symbols for symbol in targets))
                        if all(symbol.free_symbols for symbol in targets) and not any(
                            patterns & variable.free_symbols for variable in variables
                        ):
                            endpoints = (
                                (value.args[2],)
                                if isinstance(value, Limit)
                                else tuple(endpoint for limit in value.limits for endpoint in limit[1:])
                            )
                            resolved = {
                                _expression_key(endpoint): self._independent_binding(endpoint, protected, patterns)
                                for endpoint in endpoints
                            }
                            if all(endpoint is not None for endpoint in resolved.values()):
                                domains = native_domains(
                                    value, lambda endpoint: resolved[_expression_key(endpoint)], protected
                                )
                    else:
                        domains = native_domains(
                            value,
                            lambda endpoint: self._bind(endpoint, protected).subs(points, simultaneous=True),
                            protected,
                        )
                    local = {**scopes, **domains}
                    if isinstance(value, Limit):
                        visit(value.args[2], points, bound, active, uncertain, scopes, native)
                        visit(value.args[0], points, (*bound, *variables), active, True, local, (*native, value))
                        return
                    for index, limit in enumerate(value.limits):
                        outer = tuple(item[0] for item in value.limits[index + 1 :])
                        for endpoint in limit[1:]:
                            visit(
                                endpoint,
                                {symbol: point for symbol, point in points.items() if symbol not in outer},
                                (*bound, *outer),
                                active,
                                uncertain,
                                {**scopes, **{symbol: local[symbol] for symbol in outer}},
                                native,
                            )
                    if any(local[variable].is_empty is True for variable in variables) or (
                        compound and _empty_reduction(value) is not None
                    ):
                        return
                    visit(
                        value.function,
                        {symbol: point for symbol, point in points.items() if symbol not in variables},
                        (*bound, *variables),
                        active,
                        uncertain or isinstance(value, Integral),
                        local,
                        (*native, value),
                    )
                    return
                if isinstance(value, Piecewise):
                    protected = frozenset((*quantified, *bound))
                    targets = tuple(symbol for scope in native if isinstance(scope, Subs) for symbol in scope.variables)
                    patterns = frozenset().union(*(symbol.free_symbols for symbol in targets))
                    symbolic_patterns = all(symbol.free_symbols for symbol in targets)
                    for branch, condition in value.args:
                        visit(condition, points, bound, active, uncertain, scopes, native)
                        _, source_domain = _predicate_parts(guard_source(self._law(condition, protected)))
                        domain_independent = (
                            not compound
                            or source_domain is S.true
                            or (
                                symbolic_patterns
                                and self._independent_binding(source_domain, protected, patterns) is not None
                            )
                        )
                        if domain_independent:
                            source_domain = source_domain.subs(points, simultaneous=True)
                        elif source_domain is not S.true:
                            record(
                                Obligation("selector_defined", scoped_source(condition, native), "rule"),
                                "The compound substitution leaves the selector's scalar domain unresolved.",
                                True,
                            )
                        if domain_independent and source_domain is not S.true:
                            coordinates = set(source_domain.atoms(Coordinate)) | (
                                set(bound) & source_domain.free_symbols
                            )
                            claim = predicate(source_domain, name="selector_defined")
                            if coordinates:
                                variable = next(iter(coordinates)) if len(coordinates) == 1 else None
                                claim = Obligation(
                                    "selector_defined",
                                    Piecewise((S.One, source_domain), (S.NaN, S.true), evaluate=False),
                                    "equal_to" if variable is not None else "rule",
                                    wrt=variable,
                                    domain=scopes.get(variable, coordinate_domain(variable)),
                                    expected=S.One,
                                )
                            record(
                                claim,
                                "A branch selector requires defined scalar operands before choosing its value.",
                                uncertain,
                            )
                        if compound:
                            selected = None
                            if symbolic_patterns:
                                try:
                                    candidate = self._independent_binding(condition, protected, patterns)
                                    if candidate is not None:
                                        selected = candidate
                                    # Apply the native patterns before observing
                                    # their component symbols (p**4 can match p**2).
                                    elif not any(
                                        isinstance(scope, (Integral, Sum, Product, Limit)) for scope in native
                                    ):
                                        held = self._bind(scoped_source(condition, native), protected | patterns)
                                        if not held.has(Integral, Sum, Product, Limit, Derivative):
                                            selected = self._bind(held.doit(), protected)
                                except (NotImplementedError, ModelInputError, TypeError, ValueError, RecursionError):
                                    pass
                        else:
                            selected = self._bind(condition, protected).subs(points, simultaneous=True)
                        if selected is not None:
                            selected, source_domain = _predicate_parts(selected)
                            if source_domain is S.false:
                                return
                            if source_domain is not S.true:
                                selected = None
                        if selected is None or selected.free_symbols & protected or selected.has(*_BINDERS):
                            selected = None
                        else:
                            selected = truth(
                                selected,
                                tuple(condition for condition in assumptions if not condition.free_symbols & protected),
                            )
                        if selected is S.true:
                            visit(branch, points, bound, active, uncertain, scopes, native)
                            return
                        if selected is not S.false:
                            visit(branch, points, bound, active, True, scopes, native)
                            uncertain = True
                    return
                if isinstance(value, Subs):
                    visit(
                        value.expr,
                        points
                        if any(not isinstance(symbol, Symbol) for symbol in value.variables)
                        else substitution_points(value, points),
                        bound,
                        active,
                        uncertain,
                        {symbol: domain for symbol, domain in scopes.items() if symbol not in value.variables},
                        (*native, value),
                    )
                    for point in value.point:
                        visit(point, points, bound, active, uncertain, scopes, native)
                    return
                if value in sources:
                    if value not in active:
                        for source in sources[value]:
                            visit(source, points, bound, (*active, value), uncertain, scopes, native)
                    return
                if isinstance(value, Derivative):
                    variables = tuple(dict.fromkeys(value.variables))
                    if len(variables) == 1 and isinstance(variables[0], Symbol):
                        variable = variables[0]
                        curve = self._law(guard_source(value.expr), frozenset((*bound, *variables)))
                        domain = scopes.get(variable, coordinate_domain(variable))
                        if compound or captures_coordinate(variable, points, source=curve):
                            # Keep the native scope intact. The shared premise
                            # checker separates the derivative axis from values
                            # that happen to use the same coordinate identity.
                            claim = Obligation(
                                "derivative",
                                _DerivativePremises(
                                    guard_source(
                                        scoped_source(Derivative(curve, *value.variable_count, evaluate=False), native)
                                    )
                                ),
                                "predicate",
                                variable,
                                domain,
                                order=len(value.variables),
                                at=points.get(variable),
                            )
                        else:
                            curve = curve.subs(
                                {symbol: point for symbol, point in points.items() if symbol not in variables},
                                simultaneous=True,
                            )
                            claim = differentiable(
                                curve,
                                wrt=variable,
                                domain=domain,
                                order=len(value.variables),
                                at=points.get(variable),
                                name="derivative",
                            )
                            if (curve.free_symbols | (points.get(variable, S.Zero).free_symbols)) & (
                                set(bound) - {variable}
                            ):
                                claim = replace(claim, property="rule")
                                uncertain = True
                    else:
                        claim = Obligation("mixed_derivative", value, "rule")
                    record(
                        claim,
                        "The native derivative requires differentiability, not necessarily C2.",
                        uncertain,
                    )
                    bound = (*bound, *variables)
                if isinstance(value, Expr) and value.is_Pow and value.exp.is_negative is True:
                    base = self._law(guard_source(value.base), frozenset(bound))
                    captured_range = any(captures_coordinate(symbol, points, source=base) for symbol in scopes)
                    independent = None
                    if compound:
                        targets = tuple(
                            symbol for scope in native if isinstance(scope, Subs) for symbol in scope.variables
                        )
                        if all(symbol.free_symbols for symbol in targets):
                            patterns = frozenset().union(*(symbol.free_symbols for symbol in targets))
                            independent = self._independent_binding(base, frozenset(bound), patterns)
                    if (compound and independent is None) or captured_range:
                        operand = scoped_source(value, native)
                        if any(isinstance(scope, (Integral, Sum, Product, Limit)) for scope in native):
                            record(
                                Obligation("denominator", operand, "rule"),
                                "Overlapping or compound substitutions in a ranged denominator remain unresolved.",
                                True,
                            )
                        else:
                            coordinates = operand.atoms(Coordinate) | (set(bound) & operand.free_symbols)
                            for variable in sorted(coordinates, key=default_sort_key) or (None,):
                                record(
                                    Obligation(
                                        "denominator",
                                        Contains(operand, S.Complexes, evaluate=False),
                                        "predicate",
                                        variable,
                                        scopes.get(variable, coordinate_domain(variable)),
                                    ),
                                    "The original scoped division must be defined before cancellation.",
                                    uncertain,
                                )
                    else:
                        denominator = independent if independent is not None else base.subs(points, simultaneous=True)
                        coordinates = set(denominator.atoms(Coordinate)) | (set(bound) & denominator.free_symbols)
                        if len(coordinates) <= 1:
                            variable = next(iter(coordinates), None)
                            claim = nonzero(
                                denominator,
                                wrt=variable,
                                domain=scopes.get(variable, coordinate_domain(variable)),
                                name="denominator",
                            )
                        else:
                            claim = Obligation("multivariate_denominator", denominator, "rule")
                        record(claim, "The original denominator must be nonzero before cancellation.", uncertain)
                for arg in value.args:
                    visit(arg, points, bound, active, uncertain, scopes, native)

            # Follow source equations under operator scopes before algebra
            # can cancel an original division or erase a local derivative.
            try:
                visit(constraint.right)
            except (NotImplementedError, ModelInputError, RecursionError) as error:
                record(Obligation("unsupported_algebra", constraint.left, "rule"), str(error), False)
        return records, frozenset(conditional)

    def verify(self, *, max_steps: int = 100, obligations: tuple[Obligation, ...] = ()) -> VerificationReport:
        """Try bounded scalar rules; unknown does not mean a false proposition.

        The budget counts proof attempts, not wall-clock time inside the CAS.
        """
        return self._verify(max_steps=max_steps, obligations=obligations)[0]

    def _verify(
        self, *, max_steps: int, obligations: tuple[Obligation, ...]
    ) -> tuple[VerificationReport, frozenset[str]]:
        """Return evidence and its structural automatic-obligation IDs together."""
        if max_steps < 1:
            raise ModelInputError("max_steps must be positive")
        supplied = (*self.assumptions, *sorted(global_assumptions, key=default_sort_key))
        algebra_issue = ""
        assumption_issue = ""
        try:
            bound = tuple(self._bind(condition) for condition in supplied)
            assumptions = tuple(conjunction(*_predicate_parts(condition)) for condition in bound)
            facts = tuple(
                clause for condition in assumptions for clause in And.make_args(condition) if not clause.has(*_BINDERS)
            )
            operands = _unique_expressions(
                operand
                for original, condition in zip(supplied, bound)
                for operand in (condition, *self._context.expressions[_expression_key(original), frozenset()][1])
            )
            for operand in operands:
                for premise in derivative_premises(operand, None, S.Reals):
                    if premise.condition is S.false:
                        assumptions = (*assumptions, S.false)
                        assumption_issue = "An assumption uses an undefined native operation. " + " ".join(
                            premise.steps
                        )
                    elif premise.condition is None or truth(premise.condition, facts) is not S.true:
                        algebra_issue = "An assumption has unestablished native premises. " + " ".join(premise.steps)
        except (NotImplementedError, ModelInputError, RecursionError) as error:
            assumptions = supplied
            algebra_issue = f"Joint model algebra remains unresolved: {error}"
        try:
            feasible = consistency(assumptions)
        except RecursionError as error:
            feasible = None
            algebra_issue = f"Assumption consistency remains unresolved: {type(error).__name__}"
        if algebra_issue and feasible is True:
            feasible = None
        observation_issue = ""
        try:
            empty = self.family.is_empty if self._context.needs_solutions else False
        except (NotImplementedError, ModelInputError, RecursionError) as error:
            empty = None
            algebra_issue = f"Joint model algebra remains unresolved: {error}"
        if empty is True:
            feasible = False
            observation_issue = "The model equations and declared domains have no admissible state."
        elif empty is None:
            if feasible is True:
                feasible = None
            observation_issue = "Joint feasibility of the model equations and declared domains remains unresolved."
        records_by_id: dict[str, tuple[str, Obligation, str]] = {}
        automatic, conditional_operations = self._automatic(assumptions)
        if algebra_issue:
            automatic.append(
                ("model/auto:algebra", Obligation("model_algebra", S.One, "rule", reason=algebra_issue), "")
            )
        for record in [*self._declared(), *automatic, *((item.name, item, "") for item in obligations)]:
            key = record[0]
            if key in records_by_id and (
                records_by_id[key] != record or records_by_id[key][1]._proposition() != record[1]._proposition()
            ):
                raise ModelDefinitionError(f"Obligation IDs must be unique within a verification context: {key}")
            records_by_id[key] = record
        records = tuple(records_by_id.values())
        issues = [] if feasible is not False else ["The assumptions are inconsistent; proofs were skipped."]
        if assumption_issue:
            issues.append(assumption_issue)
        if observation_issue:
            issues.append(observation_issue)
        issues.extend(self.changes)
        active, steps = set(), [0]
        bound_hints = dict(self.bound_hints)
        used_hints: set[str] = set()

        def prove_supported(key, obligation, prefix):
            claim = obligation
            declaration_bindings = self._annotation_bindings(prefix)

            def native(value):
                return _rebind(sympify(value), declaration_bindings)

            def reference(path):
                return self._reference(f"{prefix}.{path}" if prefix else path)

            variable = (
                reference(claim.wrt)
                if isinstance(claim.wrt, str)
                else (native(claim.wrt) if claim.wrt is not None else None)
            )
            if variable is not None:
                variable = self._coordinate(variable)
            subject = reference(claim.subject) if isinstance(claim.subject, str) else native(claim.subject)
            self._require_symbols(subject)
            protected = frozenset() if variable is None else frozenset({variable})
            sources: list[Basic] = []
            hint_sources: list[Basic] = []

            def bind(value, *, optional=False):
                result = self._bind(value, protected)
                _, operands = self._context.expressions[_expression_key(value), protected]
                (hint_sources if optional else sources).extend(operands)
                return result

            expression = bind(subject)
            if callable(claim.expected):
                provided: set[Basic] = set()

                def expected_reference(path):
                    value = reference(path)
                    provided.update(_scoped_symbols(value))
                    return value

                expected = native(claim.expected(expected_reference))
                if (_scoped_symbols(expected) & self._changed_coordinate_scopes) - provided:
                    raise ModelDefinitionError(
                        "Rebuild an expected formula after its coordinate interface changes; "
                        "resolve current coordinates through its reference callback"
                    )
            else:
                expected = native(claim.expected) if claim.expected is not None else None
            expected_template = expected
            if expected is not None:
                self._require_symbols(expected)
                expected = bind(expected)

            def payload(value):
                if value is None:
                    return None
                value = native(value)
                self._require_symbols(value)
                return bind(value)

            domain, bound, point = payload(claim.domain), payload(claim.bound), payload(claim.at)
            if variable is not None and variable in domain.free_symbols:
                raise ModelDefinitionError("A quantified domain cannot depend on its own coordinate")
            if (
                variable is not None
                and claim.property in {"supremum_below", "derivative_ratio_bound"}
                and variable in bound.free_symbols
            ):
                raise ModelDefinitionError("A uniform target bound cannot depend on its quantified coordinate")
            resolved = replace(
                claim, subject=expression, wrt=variable, expected=expected, domain=domain, bound=bound, at=point
            )
            dependencies = self._dependencies(subject)
            token = (claim._proposition(), prefix)
            skip_reason = None
            if feasible is False or steps[0] >= max_steps or token in active:
                skip_reason = "Inconsistent premises, exhausted proof budget, or circular inference-rule dependency."
            else:
                try:
                    # Evaluating a domain here can only withhold a certificate;
                    # it never establishes the native operations in its bounds.
                    empty = evaluate_native(domain).is_empty
                except (NotImplementedError, ValueError, TypeError):
                    empty = None
                if empty is not False:
                    skip_reason = (
                        "The obligation domain is empty; no certificate is established."
                        if empty is True
                        else "The obligation domain is not established to be nonempty."
                    )
            if skip_reason is not None:
                return Evidence(
                    key,
                    obligation,
                    expression,
                    variable,
                    None,
                    (skip_reason,),
                    dependencies,
                    resolved=resolved,
                    expected_template=expected_template,
                    source_operands=_unique_expressions(sources),
                )
            steps[0] += 1
            rule = next((rule for rule in self.rules if rule.conclusion._proposition() == claim._proposition()), None)
            if rule is not None:
                active.add(token)
                try:
                    premises = tuple(prove(f"{key}/{item.name}", item, prefix) for item in rule.premises)
                finally:
                    active.remove(token)
                condition = conjunction(*(item.condition for item in premises))
                # A failed sufficient premise makes the rule inapplicable;
                # it cannot refute the rule's conclusion.
                if condition is S.false:
                    condition = None
                return Evidence(
                    key,
                    obligation,
                    expression,
                    variable,
                    condition,
                    (f"Apply {rule.name} v{rule.version}: {rule.source}",),
                    dependencies,
                    kind="rule",
                    premises=premises,
                    resolved=resolved,
                    expected_template=expected_template,
                    source_operands=_unique_expressions(sources),
                )
            canonical = (
                (f"{prefix}.{claim.subject}" if prefix else claim.subject) if isinstance(claim.subject, str) else ""
            )
            hint_key = key if key in bound_hints else f"{canonical}/{claim.name}"
            hint = bound_hints.get(hint_key)
            hint_failure = None
            if hint is not None:
                used_hints.add(hint_key)
                hint_binding_error = None
                try:
                    hint = native(hint)
                except (NotImplementedError, ValueError, TypeError, PolynomialError, RecursionError) as error:
                    hint_binding_error = error
                if hint_binding_error is None:
                    self._require_symbols(hint)
                    try:
                        hint = cast(Expr, bind(hint, optional=True))
                    except (NotImplementedError, ValueError, TypeError, PolynomialError, RecursionError) as error:
                        hint_binding_error = error
                if hint_binding_error is not None:
                    # Retain the proposal when only its resolution failed;
                    # necessary subject premises still need to be checked.
                    hint_failure = Attempt(
                        None,
                        (
                            "Unsupported optional hint binding: "
                            f"{type(hint_binding_error).__name__}: {hint_binding_error}",
                        ),
                    )
            source_operands = _unique_expressions(sources)
            hint_operands = _unique_expressions(hint_sources)
            attempt_key = _expression_key(
                (
                    resolved.property,
                    expression,
                    variable,
                    expected,
                    resolved.domain,
                    resolved.order,
                    resolved.bound,
                    resolved.at,
                    assumptions,
                    hint,
                    source_operands,
                    hint_operands,
                    hint_failure,
                )
            )
            attempts = self._context.attempts
            result = attempts.get(attempt_key)
            if result is None:
                try:
                    result = attempt(
                        resolved,
                        expression,
                        variable,
                        expected,
                        assumptions,
                        hint,
                        source_operands,
                        hint_operands,
                        hint_failure=hint_failure,
                    )
                except (NotImplementedError, ValueError, TypeError, PolynomialError, RecursionError) as error:
                    result = Attempt(None, (f"Unsupported CAS proof attempt: {type(error).__name__}: {error}",))
                attempts[attempt_key] = result
            return Evidence(
                key,
                obligation,
                expression,
                variable,
                result.condition,
                result.steps,
                dependencies,
                result.witness,
                kind="counterexample" if result.witness else "symbolic",
                resolved=resolved,
                hint=hint,
                expected_template=expected_template,
                source_operands=_unique_expressions((*source_operands, *hint_operands)),
            )

        def prove(key, obligation, prefix):
            try:
                return prove_supported(key, obligation, prefix)
            except (NotImplementedError, ModelInputError, RecursionError) as error:
                subject = obligation.subject
                expression = (
                    self._reference(f"{prefix}.{subject}" if prefix else subject)
                    if isinstance(subject, str)
                    else sympify(subject)
                )
                return Evidence(
                    key,
                    obligation,
                    expression,
                    None,
                    None,
                    (f"Unsupported model algebra: {error}",),
                    self._dependencies(expression),
                )

        evidence = tuple(prove(*record) for record in records)
        evidence = tuple(
            replace(
                item,
                condition=None,
                steps=(
                    *item.steps,
                    "An unconditional certificate did not establish this conditionally active operation.",
                ),
                witness=(),
                kind="symbolic" if item.kind == "counterexample" else item.kind,
            )
            if item.id in conditional_operations and item.condition is not S.true
            else item
            for item in evidence
        )
        unused = set(bound_hints) - used_hints
        if unused:
            issues.append(f"Unused proof hints: {', '.join(sorted(unused))}")
        report = VerificationReport(
            self.problem.model.__name__, evidence, assumptions, feasible, tuple(issues), self._labels
        )
        return report, frozenset(key for key, _, _ in automatic)

    def explain(self, obligation_id: str) -> Evidence:
        return self.verify().explain(obligation_id)

    def sample(self, subject: str | Basic, *, wrt: str | Symbol, points) -> Sample:
        """Evaluate finite points as observations, without discharging obligations."""
        variable = self._coordinate(wrt)
        expression = self.expression(subject, wrt=variable)
        reference = self._reference(subject) if isinstance(subject, str) else subject
        _, sources = self._context.expressions[_expression_key(reference), frozenset({variable})]
        if not isinstance(expression, Expr):
            raise ModelValidationError("Sampling needs a scalar expression")
        try:
            samples = tuple(sympify(point) for point in points)
        except (SympifyError, TypeError) as error:
            raise ModelValidationError("Sampling points must be finite real scalar values") from error
        domain = coordinate_domain(variable)
        try:
            # Finite flags can hide undefined operands in a point such as log(0)**0.
            # Retain its native premises and source domains before using its value.
            if any(
                not isinstance(point, Expr)
                or point.free_symbols
                or any(premise.condition is not S.true for premise in derivative_premises(point, None, S.Reals))
                or _defined(evaluate_native(guard_source(point)), None, S.Reals).condition is not S.true
                or point.is_finite is not True
                or point.is_real is not True
                or _membership(point, domain, ()) is not S.true
                for point in samples
            ):
                raise ModelValidationError(
                    "Sampling points must be bound finite real values in the coordinate's domain"
                )
            if any(
                premise.condition is not S.true
                for point in samples
                for value in (expression, *sources)
                for premise in derivative_premises(value, variable, domain, {variable: point})
            ):
                raise ModelValidationError("Sampling needs established native derivative premises")
            if any(
                _defined(evaluate_native(source).subs(variable, point), None, S.Reals).condition is not S.true
                for point in samples
                for source in sources
            ):
                raise ModelValidationError("Sampling needs established observation source domains")
            native = evaluate_native(expression)
            values = tuple(native.subs(variable, point).evalf() for point in samples)
        except ModelValidationError:
            raise
        except (ArithmeticError, TypeError, ValueError, RuntimeError) as error:
            raise ModelValidationError(f"Sampling could not establish finite real values: {error}") from error
        if any(value.free_symbols or value.is_finite is not True or value.is_real is not True for value in values):
            raise ModelValidationError("Sampling needs finite real values and bound parameters")
        return Sample(expression, variable, samples, values)

    def run(
        self,
        method: Method[ResultT] | Callable[[VerificationContext[ModelT]], Method[ResultT]],
        **settings,
    ) -> VerificationResult[ResultT]:
        """Run a prepared method or build one for this context before checking it."""
        if not isinstance(method, Method):
            if not callable(method):
                raise ModelDefinitionError("run needs a Method or a context-to-Method factory")
            method = method(self)
            if not isinstance(method, Method):
                raise ModelDefinitionError(
                    "A method factory must return a Method; pass its computation as Method.execute"
                )
        if method.bound_context is not None and (
            method.bound_context is not self
            or _expression_key(method.bound_assumptions) != _expression_key(frozenset(global_assumptions))
        ):
            raise VerificationError(
                f"{method.name} is bound to a different verification context or SymPy assumption context; rebuild it",
                self,
                self.verify(max_steps=1000),
            )
        report, automatic = self._verify(max_steps=1000, obligations=method.preconditions)
        direct_uses = {self._reference(path) for path in method.uses}
        uses: set[str] = set()
        for symbol in direct_uses:
            uses.update(self._dependencies(symbol, include_declarations=True))
        required = {item.name for item in method.preconditions}
        model_scopes = {
            f"{path}/{item.name}" if path else item.name: path
            for path, annotation in _Obligations.for_model(self.problem.model).items()
            if annotation.model_level
            for item in annotation.obligations
        }

        def consumed(item):
            if item.id in required:
                return True
            if item.id in automatic:
                return not item.obligation.consumer or self._reference(item.obligation.consumer) in direct_uses
            return (
                item.obligation.consumer in uses
                or item.id.partition("/")[0] in uses
                or item.obligation.consumer == method.name
                or (
                    item.id in model_scopes
                    and any(
                        not model_scopes[item.id]
                        or path == model_scopes[item.id]
                        or path.startswith(model_scopes[item.id] + ".")
                        for path in uses
                    )
                )
            )

        blocked = [item for item in report.evidence if consumed(item) and item.condition is not S.true]
        if report.consistent is False or blocked:
            details = ", ".join(f"{item.id}: {item.verdict}" for item in blocked)
            raise VerificationError(f"{method.name} has unestablished premises: {details}", self, report)
        try:
            data = method.execute(self, **settings)
        except VerificationError:
            raise
        except (ArithmeticError, ValueError, RuntimeError) as error:
            raise VerificationError(f"{method.name} failed: {error}", self, report) from error
        return VerificationResult(method.name, data, report, MappingProxyType(dict(settings)))


class VerificationModel(Modello):
    """Opt into automatic declaration reports, including every subclass.

    Ordinary model construction is unchanged. Mathematical failures stay in
    the report; methods decide which established premises they need.
    """

    _verification_definition_report: ClassVar[VerificationReport]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._verification_definition_report = VerificationContext.for_model(cls).verify()

    @classmethod
    def definition_report(cls) -> VerificationReport:
        return cls._verification_definition_report


def specifies(value: ValueT, *obligations: Obligation) -> ValueT:
    """Return a field declaration or model subclass carrying obligations.

    Property functions create the same Obligation values accepted by contexts,
    method preconditions, and inference-rule premises. Use the return value: the input expression,
    model class, and earlier verification contexts are left unchanged.
    """
    if isinstance(value, type) and issubclass(value, Modello):
        inherited = _Obligations.for_model(value).get("")
        previous = () if inherited is None else inherited.obligations
        constrained = new_class(f"{value.__name__}WithSpecification", (value,))
        annotate_model(constrained, _Obligations(*previous, *obligations, model_level=True))
        if issubclass(constrained, VerificationModel):
            constrained._verification_definition_report = VerificationContext.for_model(constrained).verify()
        return cast(ValueT, constrained)
    return annotate(value, _Obligations(*obligations))

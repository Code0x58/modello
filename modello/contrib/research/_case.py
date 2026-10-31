"""Bind declarations, assumptions, contracts, and computations to fresh cases."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field, replace
from functools import cached_property
from types import MappingProxyType, new_class
from typing import ClassVar, Generic, TypeVar, cast

from sympy import (
    And,
    Basic,
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

from modello import ConstraintKind, Coordinate, Modello, annotate, annotate_model, component, inspect_model
from modello._evaluation import _BINDERS, _expression_key, _predicate_parts, _rebind, _unique_expressions
from modello._expressions import _definition_resolver, _DefinitionResolver
from modello._operators import _bound_coordinates, _scalar_atoms, _ScopedExpression, _SymbolicTemplate
from modello._problem import ModelConstraint
from modello.diagnostics import ModelProblem
from modello.errors import ModelDefinitionError, ModelInputError, ModelValidationError
from modello.series import Recurrence, _project_series, _SeriesTerm
from modello.solutions import _field_domain, _membership

from ._proof import (
    Attempt,
    _defined,
    attempt,
    conjunction,
    consistency,
    coordinate_domain,
    derivative_premises,
    evaluate_native,
    guard_source,
    native_domains,
    truth,
)
from ._records import (
    Evidence,
    Experiment,
    Method,
    Requirement,
    ResearchError,
    ResearchReport,
    ResearchResult,
    ResultT,
    Theorem,
    _Requirements,
    differentiable,
    nonzero,
    predicate,
)

ModelT = TypeVar("ModelT", bound=Modello)
ValueT = TypeVar("ValueT")


@dataclass
class _Context:
    """Resolved work for one case under one set of global assumptions."""

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
                    continue
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
    def family(self):
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
    coordinate_map = {
        value: new_coordinates[name]
        for name, value in old_coordinates.items()
        if name in new_coordinates and value != new_coordinates[name]
    }
    own_coordinates = {name: value for name, value in _coordinates(model).items() if "." not in name}
    namespace = model._modello_namespace
    sources = (*namespace.attrs.values(), *namespace.vector_attrs.values(), *namespace.templates.values())
    used = set().union(*(_scoped_symbols(value) for value in sources))
    used.update(own_coordinates.values())
    needed = {old: new for old, new in coordinate_map.items() if old in used}
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
class ResearchCase(Generic[ModelT]):
    """A persistent research branch; construction does not solve the model."""

    problem: ModelProblem[ModelT]
    assumptions: tuple[Basic, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    hints: tuple[tuple[str, Expr], ...] = ()
    theorems: tuple[Theorem, ...] = ()
    changes: tuple[str, ...] = ()

    @classmethod
    def for_model(cls, model: type[ModelT], name: str = "research", **observations) -> ResearchCase[ModelT]:
        return cls(inspect_model(model, name, **observations))

    @property
    def symbols(self) -> ModelT:
        return self.problem.symbols

    @property
    def family(self):
        """The existing exact solution family, evaluated only on explicit demand."""
        return self._context.family

    @property
    def _context(self) -> _Context:
        key = frozenset(global_assumptions)
        if key not in self._contexts:
            self._contexts[key] = _Context(self.problem, self.assumptions)
        return self._contexts[key]

    @cached_property
    def _contexts(self) -> dict[frozenset[Basic], _Context]:
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
    def _labels(self) -> tuple[tuple[Symbol, str], ...]:
        labels: dict[Symbol, str] = {}
        for path, symbol in self._fields.items():
            previous = labels.get(symbol)
            if previous is None or (path.count("."), len(path)) < (previous.count("."), len(previous)):
                labels[symbol] = path
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
        if value.free_symbols - self._known_symbols:
            raise ModelValidationError("Use this research case's symbols or coordinates from its declaration")

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

    def expression(self, subject: str | Basic, *, wrt: str | Symbol | None = None) -> Expr:
        """Resolve a law, preserving an independent ``wrt`` symbol from observations."""
        variable = self._coordinate(wrt) if wrt is not None else None
        value = self._reference(subject) if isinstance(subject, str) else subject
        self._require_symbols(value)
        return cast(Expr, self._bind(value, frozenset() if variable is None else frozenset({variable})))

    def assume(self, *conditions) -> ResearchCase[ModelT]:
        checked = []
        for value in conditions:
            condition = sympify(value)
            if not isinstance(condition, Boolean) or isinstance(condition, Expr):
                raise ModelInputError("Assumptions must be symbolic predicates; use Eq for equalities")
            self._require_symbols(condition)
            checked.append(condition)
        return replace(self, assumptions=(*self.assumptions, *checked))

    def observe(self, **observations) -> ResearchCase[ModelT]:
        return replace(self, problem=self.problem._capture_inputs(**observations))

    def without_assumptions(self, *conditions: Basic, reason: str) -> ResearchCase[ModelT]:
        """Remove explicitly recorded assumptions, retaining why they changed.

        Declaration domains and observations are unaffected. Each new report
        checks every dependent conclusion again under the remaining premises.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ModelInputError("Record a reason when removing assumptions")
        removed = tuple(dict.fromkeys(sympify(condition) for condition in conditions))
        missing = tuple(condition for condition in removed if condition not in self.assumptions)
        if missing:
            raise ModelInputError(f"These assumptions were not recorded in this case: {missing}")
        return replace(
            self,
            assumptions=tuple(condition for condition in self.assumptions if condition not in removed),
            changes=(*self.changes, *(f"Removed assumption {condition}: {reason.strip()}" for condition in removed)),
        )

    def require(self, *requirements: Requirement) -> ResearchCase[ModelT]:
        if any(not isinstance(item, Requirement) for item in requirements):
            raise ModelDefinitionError("require accepts Requirement records")
        return replace(self, requirements=(*self.requirements, *requirements))

    def without_requirements(self, *requirement_ids: str, reason: str) -> ResearchCase[ModelT]:
        """Remove case-added obligations; method and declaration contracts remain.

        Change a declaration or component to change its annotations. Automatic
        obligations and a method's own requirements cannot be waived here.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ModelInputError("Record a reason when removing requirements")
        removed = tuple(dict.fromkeys(requirement_ids))
        available = {item.name for item in self.requirements}
        missing = tuple(key for key in removed if key not in available)
        if missing:
            raise ModelInputError(f"Only case-added requirements can be removed; unknown case IDs: {missing}")
        return replace(
            self,
            requirements=tuple(item for item in self.requirements if item.name not in removed),
            changes=(*self.changes, *(f"Removed requirement {key}: {reason.strip()}" for key in removed)),
        )

    def with_hint(self, requirement_id: str, *, upper) -> ResearchCase[ModelT]:
        """Propose a uniform upper bound; every check must establish it."""
        try:
            value = sympify(upper)
        except (SympifyError, TypeError) as error:
            raise ModelInputError("An upper-bound hint must be a scalar expression") from error
        if not isinstance(value, Expr):
            raise ModelInputError("An upper-bound hint must be a scalar expression")
        return replace(self, hints=tuple({**dict(self.hints), requirement_id: value}.items()))

    def with_theorem(self, theorem: Theorem) -> ResearchCase[ModelT]:
        if not theorem.source or any(item.name == theorem.name for item in self.theorems):
            raise ModelDefinitionError("Theorems need a source and a unique rule name")
        return replace(self, theorems=(*self.theorems, theorem))

    def replace(self, component_path: str, model: type[Modello]) -> ResearchCase[ModelT]:
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
                        "Replacement splits aliased fields or declarations; declare a new model and case"
                    )
                mapping[original] = target
        old_coordinates = _coordinates(self.problem.model)
        new_coordinates = _coordinates(declaration)
        changed_coordinates = {
            name: (old, new_coordinates[name])
            for name, old in old_coordinates.items()
            if name in new_coordinates and old != new_coordinates[name]
        }

        def coordinate_bindings(*subjects):
            referenced = set().union(*(_scoped_symbols(value) for value in subjects if isinstance(value, Basic)))
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
                    unchanged = [
                        path
                        for path, value in old_coordinates.items()
                        if value == old and new_coordinates.get(path) == old
                    ]
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

        global_coordinates = coordinate_bindings()
        for value in self.assumptions:
            for old in _scoped_symbols(value) & set(global_coordinates):
                if any(
                    coordinate == old and new_coordinates.get(path) == old
                    for path, coordinate in old_coordinates.items()
                ):
                    raise ModelDefinitionError(
                        "An assumption uses a shared renamed coordinate; refine its scope explicitly"
                    )
        observations: list[ModelConstraint] = []
        invalidated = []
        for constraint in self.problem.constraints:
            if constraint.kind is not ConstraintKind.DEFINITION:
                if constraint.kind is ConstraintKind.INFERRED and any(
                    path == component_path or path.startswith(component_path + ".")
                    for path in self._dependencies(constraint.left, include_declarations=True)
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
                            {
                                **mapping,
                                **(
                                    coordinate_bindings(constraint.left, constraint.right)
                                    if _scoped_symbols(constraint.right) & set(global_coordinates)
                                    else {}
                                ),
                            },
                        ),
                    )
                )
        problem = replace(problem, constraints=(*problem.constraints, *observations))

        def rebound(claim):
            payloads = (claim.subject, claim.wrt, claim.domain, claim.bound, claim.expected, claim.at)
            native_symbols = set().union(*(_scoped_symbols(value) for value in payloads if isinstance(value, Basic)))
            bindings = {
                **mapping,
                **(coordinate_bindings(*payloads) if native_symbols & set(global_coordinates) else {}),
            }

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

        return ResearchCase(
            problem,
            tuple(_rebind(value, {**mapping, **global_coordinates}) for value in self.assumptions),
            tuple(rebound(item) for item in self.requirements),
            tuple((key, _rebind(hint, {**mapping, **coordinate_bindings(hint)})) for key, hint in self.hints),
            tuple(
                replace(
                    rule,
                    conclusion=rebound(rule.conclusion),
                    premises=tuple(rebound(item) for item in rule.premises),
                )
                for rule in self.theorems
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
        for path, annotation in _Requirements.for_model(self.problem.model).items():
            prefix = path if annotation.model_level else path.rpartition(".")[0]
            for requirement in annotation.requirements:
                key = f"{path}/{requirement.name}" if path else requirement.name
                records.append((key, requirement, prefix))
        records.extend((item.name, item, "") for item in self.requirements)
        return records

    def _automatic(self, assumptions: tuple[Basic, ...]) -> tuple[list[tuple[str, Requirement, str]], frozenset[str]]:
        records: list[tuple[str, Requirement, str]] = []
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

            def visit(value, points=None, bound=(), active=(), uncertain=False, scopes=None):
                points = points or {}
                scopes = {} if scopes is None else scopes
                if isinstance(value, (Integral, Sum, Product, Limit)):
                    variables = tuple(_bound_coordinates(value))
                    protected = frozenset((*quantified, *bound))
                    local = {
                        **scopes,
                        **native_domains(
                            value,
                            lambda endpoint: self._bind(endpoint, protected).subs(points, simultaneous=True),
                            protected,
                        ),
                    }
                    if isinstance(value, Limit):
                        visit(value.args[2], points, bound, active, uncertain, scopes)
                        visit(value.args[0], points, (*bound, *variables), active, True, local)
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
                            )
                    if any(local[variable].is_empty is True for variable in variables):
                        return
                    visit(
                        value.function,
                        {symbol: point for symbol, point in points.items() if symbol not in variables},
                        (*bound, *variables),
                        active,
                        uncertain or isinstance(value, Integral),
                        local,
                    )
                    return
                if isinstance(value, Piecewise):
                    for branch, condition in value.args:
                        visit(condition, points, bound, active, uncertain, scopes)
                        protected = frozenset((*quantified, *bound))
                        _, source_domain = _predicate_parts(guard_source(self._law(condition, protected)))
                        source_domain = source_domain.subs(points, simultaneous=True)
                        if source_domain is not S.true:
                            coordinates = set(source_domain.atoms(Coordinate)) | (
                                set(bound) & source_domain.free_symbols
                            )
                            claim = predicate(source_domain, name="selector_defined")
                            if coordinates:
                                variable = next(iter(coordinates)) if len(coordinates) == 1 else None
                                claim = Requirement(
                                    "selector_defined",
                                    Piecewise((S.One, source_domain), (S.NaN, S.true), evaluate=False),
                                    "exact_form" if variable is not None else "theorem",
                                    wrt=variable,
                                    domain=scopes.get(variable, coordinate_domain(variable)),
                                    expected=S.One,
                                )
                            record(
                                claim,
                                "A branch selector requires defined scalar operands before choosing its value.",
                                uncertain,
                            )
                        selected = self._bind(condition, protected).subs(points, simultaneous=True)
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
                            visit(branch, points, bound, active, uncertain, scopes)
                            return
                        if selected is not S.false:
                            visit(branch, points, bound, active, True, scopes)
                            uncertain = True
                    return
                if isinstance(value, Subs):
                    local = {
                        symbol: point.subs(points, simultaneous=True)
                        for symbol, point in zip(value.variables, value.point)
                    }
                    visit(
                        value.expr,
                        {**points, **local},
                        bound,
                        active,
                        uncertain,
                        {symbol: domain for symbol, domain in scopes.items() if symbol not in value.variables},
                    )
                    for point in value.point:
                        visit(point, points, bound, active, uncertain, scopes)
                    return
                if value in sources:
                    if value not in active:
                        for source in sources[value]:
                            visit(source, points, bound, (*active, value), uncertain, scopes)
                    return
                if isinstance(value, Derivative):
                    variables = tuple(dict.fromkeys(value.variables))
                    if len(variables) == 1:
                        variable = variables[0]
                        curve = self._law(value.expr, frozenset((*bound, *variables))).subs(
                            {symbol: point for symbol, point in points.items() if symbol not in variables},
                            simultaneous=True,
                        )
                        claim = differentiable(
                            curve,
                            wrt=variable,
                            domain=scopes.get(variable, coordinate_domain(variable)),
                            order=len(value.variables),
                            at=points.get(variable),
                            name="derivative",
                        )
                        if (curve.free_symbols | (points.get(variable, S.Zero).free_symbols)) & (
                            set(bound) - {variable}
                        ):
                            claim = replace(claim, property="theorem")
                            uncertain = True
                    else:
                        claim = Requirement("mixed_derivative", value, "theorem")
                    record(
                        claim,
                        "The native derivative requires differentiability, not necessarily C2.",
                        uncertain,
                    )
                    bound = (*bound, *variables)
                if isinstance(value, Expr) and value.is_Pow and value.exp.is_negative is True:
                    denominator = self._law(value.base, frozenset(bound)).subs(points, simultaneous=True)
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
                        claim = Requirement("multivariate_denominator", denominator, "theorem")
                    record(claim, "The original denominator must be nonzero before cancellation.", uncertain)
                for arg in value.args:
                    visit(arg, points, bound, active, uncertain, scopes)

            # Follow source equations under operator scopes before algebra
            # can cancel an original division or erase a local derivative.
            try:
                visit(constraint.right)
            except (NotImplementedError, ModelInputError) as error:
                record(Requirement("unsupported_algebra", constraint.left, "theorem"), str(error), False)
        return records, frozenset(conditional)

    def check(self, *, max_steps: int = 100, requirements: tuple[Requirement, ...] = ()) -> ResearchReport:
        """Try bounded scalar rules; unknown does not mean a false proposition.

        The budget counts proof attempts, not wall-clock time inside the CAS.
        """
        return self._check(max_steps=max_steps, requirements=requirements)[0]

    def _check(self, *, max_steps: int, requirements: tuple[Requirement, ...]) -> tuple[ResearchReport, frozenset[str]]:
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
        except (NotImplementedError, ModelInputError) as error:
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
        if self._context.needs_solutions:
            try:
                empty = self.family.is_empty
            except (NotImplementedError, ModelInputError) as error:
                empty = None
                algebra_issue = f"Joint model algebra remains unresolved: {error}"
            if empty is True:
                feasible = False
                observation_issue = "The model equations and declared domains have no admissible state."
            elif empty is None:
                if feasible is True:
                    feasible = None
                observation_issue = "Joint feasibility of the model equations and declared domains remains unresolved."
        records_by_id: dict[str, tuple[str, Requirement, str]] = {}
        automatic, conditional_operations = self._automatic(assumptions)
        if algebra_issue:
            automatic.append(
                ("model/auto:algebra", Requirement("model_algebra", S.One, "theorem", reason=algebra_issue), "")
            )
        for record in [*self._declared(), *automatic, *((item.name, item, "") for item in requirements)]:
            key = record[0]
            if key in records_by_id and (
                records_by_id[key] != record or records_by_id[key][1]._proposition() != record[1]._proposition()
            ):
                raise ModelDefinitionError(f"Requirement IDs must be unique within a research case: {key}")
            records_by_id[key] = record
        records = tuple(records_by_id.values())
        issues = [] if feasible is not False else ["The assumptions are inconsistent; proofs were skipped."]
        if assumption_issue:
            issues.append(assumption_issue)
        if observation_issue:
            issues.append(observation_issue)
        issues.extend(self.changes)
        active, steps = set(), [0]
        hints = dict(self.hints)
        used_hints: set[str] = set()

        def prove_supported(key, requirement, prefix):
            claim = requirement
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
            expected = (
                native(claim.expected(reference))
                if callable(claim.expected)
                else (native(claim.expected) if claim.expected is not None else None)
            )
            expected_template = expected
            if expected is not None:
                self._require_symbols(expected)
                expected = bind(expected)

            def payload(value, *, optional=False):
                if value is None:
                    return None
                value = native(value)
                self._require_symbols(value)
                return bind(value, optional=optional)

            domain, bound, point = payload(claim.domain), payload(claim.bound), payload(claim.at)
            if variable is not None and variable in domain.free_symbols:
                raise ModelDefinitionError("A quantified domain cannot depend on its own coordinate")
            if (
                variable is not None
                and claim.property in {"uniform_bound", "curvature_bound"}
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
                skip_reason = "Inconsistent premises, exhausted proof budget, or circular theorem dependency."
            else:
                try:
                    # Evaluating a domain here can only withhold a certificate;
                    # it never establishes the native operations in its bounds.
                    empty = evaluate_native(domain).is_empty
                except (NotImplementedError, ValueError, TypeError):
                    empty = None
                if empty is not False:
                    skip_reason = (
                        "The requirement domain is empty; no certificate is established."
                        if empty is True
                        else "The requirement domain is not established to be nonempty."
                    )
            if skip_reason is not None:
                return Evidence(
                    key,
                    requirement,
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
            theorem = next(
                (rule for rule in self.theorems if rule.conclusion._proposition() == claim._proposition()), None
            )
            if theorem is not None:
                active.add(token)
                try:
                    premises = tuple(prove(f"{key}/{item.name}", item, prefix) for item in theorem.premises)
                finally:
                    active.remove(token)
                condition = conjunction(*(item.condition for item in premises))
                # A failed sufficient premise makes the theorem inapplicable;
                # it cannot refute the theorem's conclusion.
                if condition is S.false:
                    condition = None
                return Evidence(
                    key,
                    requirement,
                    expression,
                    variable,
                    condition,
                    (f"Apply {theorem.name} v{theorem.version}: {theorem.source}",),
                    dependencies,
                    kind="theorem",
                    premises=premises,
                    resolved=resolved,
                    expected_template=expected_template,
                    source_operands=_unique_expressions(sources),
                )
            canonical = (
                (f"{prefix}.{claim.subject}" if prefix else claim.subject) if isinstance(claim.subject, str) else ""
            )
            hint_key = key if key in hints else f"{canonical}/{claim.name}"
            hint = hints.get(hint_key)
            if hint is not None:
                used_hints.add(hint_key)
                hint = cast(Expr, payload(hint, optional=True))
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
                )
            )
            attempts = self._context.attempts
            result = attempts.get(attempt_key)
            if result is None:
                try:
                    result = attempt(
                        resolved, expression, variable, expected, assumptions, hint, source_operands, hint_operands
                    )
                except (NotImplementedError, ValueError, TypeError, PolynomialError, RecursionError) as error:
                    result = Attempt(None, (f"Unsupported CAS proof attempt: {type(error).__name__}: {error}",))
                attempts[attempt_key] = result
            return Evidence(
                key,
                requirement,
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

        def prove(key, requirement, prefix):
            try:
                return prove_supported(key, requirement, prefix)
            except (NotImplementedError, ModelInputError) as error:
                subject = requirement.subject
                expression = (
                    self._reference(f"{prefix}.{subject}" if prefix else subject)
                    if isinstance(subject, str)
                    else sympify(subject)
                )
                return Evidence(
                    key,
                    requirement,
                    expression,
                    None,
                    None,
                    (f"Unsupported model algebra: {error}",),
                    self._dependencies(expression),
                )

        checks = tuple(prove(*record) for record in records)
        checks = tuple(
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
            for item in checks
        )
        unused = set(hints) - used_hints
        if unused:
            issues.append(f"Unused proof hints: {', '.join(sorted(unused))}")
        report = ResearchReport(self.problem.model.__name__, checks, assumptions, feasible, tuple(issues), self._labels)
        return report, frozenset(key for key, _, _ in automatic)

    def explain(self, requirement_id: str) -> Evidence:
        return self.check().explain(requirement_id)

    def sample(self, subject: str | Basic, *, wrt: str | Symbol, points) -> Experiment:
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
        if any(
            not isinstance(point, Expr)
            or point.free_symbols
            or point.is_finite is not True
            or point.is_real is not True
            or _membership(point, domain, ()) is not S.true
            for point in samples
        ):
            raise ModelValidationError("Sampling points must be bound finite real values in the coordinate's domain")
        if any(
            premise.condition is not S.true
            for point in samples
            for value in (expression, *sources)
            for premise in derivative_premises(value, variable, domain, {variable: point})
        ):
            raise ModelValidationError("Sampling needs established native derivative premises")
        try:
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
        return Experiment(expression, variable, samples, values)

    def run(
        self,
        method: Method[ResultT] | Callable[[ResearchCase[ModelT]], Method[ResultT]],
        **settings,
    ) -> ResearchResult[ResultT]:
        """Run a prepared method or build one for this case before checking it."""
        if not isinstance(method, Method):
            if not callable(method):
                raise ModelDefinitionError("run needs a Method or a case-to-Method factory")
            method = method(self)
            if not isinstance(method, Method):
                raise ModelDefinitionError(
                    "A method factory must return a Method; pass its computation as Method.execute"
                )
        if method.bound_case is not None and (
            method.bound_case is not self or method.bound_assumptions != frozenset(global_assumptions)
        ):
            raise ResearchError(
                f"{method.name} is bound to a different research case or SymPy assumption context; rebuild it",
                self,
                self.check(max_steps=1000),
            )
        report, automatic = self._check(max_steps=1000, requirements=method.requirements)
        direct_uses = {self._reference(path) for path in method.uses}
        uses: set[str] = set()
        for symbol in direct_uses:
            uses.update(self._dependencies(symbol, include_declarations=True))
        required = {item.name for item in method.requirements}
        model_scopes = {
            f"{path}/{item.name}" if path else item.name: path
            for path, annotation in _Requirements.for_model(self.problem.model).items()
            if annotation.model_level
            for item in annotation.requirements
        }

        def consumed(item):
            if item.id in required:
                return True
            if item.id in automatic:
                return not item.requirement.consumer or self._reference(item.requirement.consumer) in direct_uses
            return (
                item.requirement.consumer in uses
                or item.id.partition("/")[0] in uses
                or item.requirement.consumer == method.name
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

        blocked = [item for item in report.checks if consumed(item) and item.condition is not S.true]
        if report.consistent is False or blocked:
            details = ", ".join(f"{item.id}: {item.verdict}" for item in blocked)
            raise ResearchError(f"{method.name} has unestablished premises: {details}", self, report)
        try:
            data = method.execute(self, **settings)
        except ResearchError:
            raise
        except (ArithmeticError, ValueError, RuntimeError) as error:
            raise ResearchError(f"{method.name} failed: {error}", self, report) from error
        return ResearchResult(method.name, data, report, MappingProxyType(dict(settings)))


class ResearchModel(Modello):
    """Opt into automatic declaration reports, including every subclass.

    Ordinary model construction is unchanged. Mathematical failures stay in
    the report; methods decide which established premises they need.
    """

    _research_definition_report: ClassVar[ResearchReport]

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._research_definition_report = ResearchCase.for_model(cls).check()

    @classmethod
    def definition_report(cls) -> ResearchReport:
        return cls._research_definition_report


def requires(value: ValueT, *requirements: Requirement) -> ValueT:
    """Return a field declaration or model subclass carrying requirements.

    Property functions create the same Requirement values accepted by cases,
    methods, and theorem premises. Use the return value: the input expression,
    model class, and earlier research cases are left unchanged.
    """
    if isinstance(value, type) and issubclass(value, Modello):
        inherited = _Requirements.for_model(value).get("")
        previous = () if inherited is None else inherited.requirements
        constrained = new_class(f"{value.__name__}WithRequirements", (value,))
        annotate_model(constrained, _Requirements(*previous, *requirements, model_level=True))
        if issubclass(constrained, ResearchModel):
            constrained._research_definition_report = ResearchCase.for_model(constrained).check()
        return cast(ValueT, constrained)
    return annotate(value, _Requirements(*requirements))

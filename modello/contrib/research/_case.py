"""Bind declarations, assumptions, contracts, and computations to fresh cases."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field as dataclass_field, replace
from functools import cached_property
from types import MappingProxyType, new_class
from typing import ClassVar, Generic, TypeVar, cast

from sympy import Basic, Derivative, Expr, S, Subs, Symbol, default_sort_key, sympify
from sympy.assumptions import global_assumptions
from sympy.logic.boolalg import Boolean
from sympy.polys.polyerrors import PolynomialError

from modello import ConstraintKind, Coordinate, Modello, component, inspect_model
from modello._evaluation import _rebind, _substitute
from modello._problem import ModelConstraint, _algebra
from modello.diagnostics import ModelProblem
from modello.errors import ModelDefinitionError, ModelInputError, ModelValidationError

from ._proof import Attempt, attempt, conjunction, consistency, coordinate_domain
from ._records import (
    Claim,
    Evidence,
    Experiment,
    Method,
    ProofHint,
    Requirement,
    Requires,
    ResearchError,
    ResearchReport,
    ResearchResult,
    ResultT,
    Theorem,
    differentiable,
    nonzero,
)

ModelT = TypeVar("ModelT", bound=Modello)


@dataclass
class _Context:
    """Resolved work for one case under one set of global assumptions."""

    problem: ModelProblem
    assumptions: tuple[Basic, ...]
    parameters: dict[Symbol, Expr] = dataclass_field(default_factory=dict)
    expressions: dict[tuple[Basic, frozenset[Basic]], Basic] = dataclass_field(default_factory=dict)
    attempts: dict[tuple, Attempt] = dataclass_field(default_factory=dict)

    @cached_property
    def law(self):
        definitions = tuple(item for item in self.problem.constraints if item.kind is ConstraintKind.DEFINITION)
        return _algebra(definitions, self.problem.templates.values()).resolve.symbolic()

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
    result = {}
    names = {
        name
        for base in model.__mro__
        for name, value in getattr(base, "_modello_namespace", {}).items()
        if isinstance(value, Coordinate)
    }
    for name in sorted(names):
        for base in model.__mro__:
            value = getattr(base, "_modello_namespace", {}).get(name)
            if isinstance(value, Coordinate):
                result[prefix + name] = value
                break
    for name, child in model._modello_namespace.nested_models.items():
        result.update(_coordinates(child._modello_model, prefix + name + "."))
    return result


def _replace_component(model: type[Modello], path: tuple[str, ...], replacement: type[Modello]) -> type[Modello]:
    if not path or not isinstance(replacement, type) or not issubclass(replacement, Modello):
        raise ModelDefinitionError("Replacement needs a component path and a Modello class")
    old_child = _schema(model, path[:1])
    if len(path) > 1:
        replacement = _replace_component(old_child, path[1:], replacement)
    old_coordinates, new_coordinates = _coordinates(old_child), _coordinates(replacement)
    coordinate_map = {
        value: new_coordinates[name]
        for name, value in old_coordinates.items()
        if name in new_coordinates and value != new_coordinates[name]
    }
    own_coordinates = {name: value for name, value in _coordinates(model).items() if "." not in name}
    used = set().union(*(value.free_symbols for value in model._modello_namespace.attrs.values()))
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
    hints: tuple[tuple[str, ProofHint], ...] = ()
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

    def _reference(self, path: str) -> Expr:
        if path in self._fields:
            return self._fields[path]
        parts = tuple(path.split("."))
        model = _schema(self.problem.model, parts[:-1])
        for base in model.__mro__:
            namespace = getattr(base, "_modello_namespace", {})
            value = namespace.get(parts[-1])
            if isinstance(value, Coordinate):
                return value
        raise ModelDefinitionError(f"Unknown scalar or coordinate path: {path}")

    def _require_symbols(self, value: Basic) -> None:
        allowed = set(self._fields.values()) | set().union(*(item.symbols for item in self.problem.constraints))
        seen = set()

        def coordinates(model):
            if model in seen:
                return
            seen.add(model)
            for base in model.__mro__:
                namespace = getattr(base, "_modello_namespace", {})
                allowed.update(item for item in namespace.values() if isinstance(item, Coordinate))
            for child in model._modello_namespace.nested_models.values():
                coordinates(child._modello_model)

        coordinates(self.problem.model)
        if value.free_symbols - allowed:
            raise ModelValidationError("Use this research case's symbols or coordinates from its declaration")

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
            for name, coordinate in getattr(base, "_modello_namespace", {}).items():
                if isinstance(coordinate, Coordinate):
                    mapping[coordinate] = self._reference(path_prefix + name)
        return mapping

    def _bind(self, value: Basic, protected: frozenset[Basic] = frozenset()) -> Basic:
        context = self._context
        key = (value, protected)
        if key in context.expressions:
            return context.expressions[key]
        result = context.law(value, protected)
        if any(item.kind is not ConstraintKind.DEFINITION for item in self.problem.constraints):
            # Bind free parameters individually. Resolving the entire curve
            # through a specialized family would also substitute its coordinate.
            replacements = {}
            for symbol in result.free_symbols - protected:
                if symbol in self._fields.values():
                    if symbol not in context.parameters:
                        context.parameters[symbol] = context.family.expression(symbol)
                    resolved = context.parameters[symbol]
                    if resolved != symbol:
                        replacements[symbol] = resolved
            result = _substitute(result, replacements)
        context.expressions[key] = result
        return result

    def expression(self, subject: str | Basic, *, wrt: str | Symbol | None = None) -> Expr:
        variable = self._reference(wrt) if isinstance(wrt, str) else wrt
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

    def without_requirements(self, *claim_ids: str, reason: str) -> ResearchCase[ModelT]:
        """Remove case-added obligations; method and declaration contracts remain.

        Change a declaration or component to change its annotations. Automatic
        obligations and a method's own requirements cannot be waived here.
        """
        if not isinstance(reason, str) or not reason.strip():
            raise ModelInputError("Record a reason when removing requirements")
        removed = tuple(dict.fromkeys(claim_ids))
        available = {item.claim.name for item in self.requirements}
        missing = tuple(key for key in removed if key not in available)
        if missing:
            raise ModelInputError(f"Only case-added requirements can be removed; unknown case IDs: {missing}")
        return replace(
            self,
            requirements=tuple(item for item in self.requirements if item.claim.name not in removed),
            changes=(*self.changes, *(f"Removed requirement {key}: {reason.strip()}" for key in removed)),
        )

    def with_hint(self, claim_id: str, hint: ProofHint) -> ResearchCase[ModelT]:
        if not isinstance(hint, ProofHint):
            raise ModelInputError("A proof hint must be created with upper_bound")
        return replace(self, hints=tuple({**dict(self.hints), claim_id: hint}.items()))

    def with_theorem(self, theorem: Theorem) -> ResearchCase[ModelT]:
        if not theorem.source or any(item.name == theorem.name for item in self.theorems):
            raise ModelDefinitionError("Theorems need a source and a unique rule name")
        return replace(self, theorems=(*self.theorems, theorem))

    def replace(self, component_path: str, model: type[Modello]) -> ResearchCase[ModelT]:
        declaration = _replace_component(self.problem.model, tuple(component_path.split(".")), model)
        problem = inspect_model(cast(type[ModelT], declaration), self.problem.name)
        paths = {field.path: field.symbol for field in problem.fields}
        mapping = {}
        for field in self.problem.fields:
            if field.path not in paths:
                raise ModelDefinitionError(f"Replacement removed a recorded field: {'.'.join(field.path)}")
            mapping[field.symbol] = paths[field.path]
        old_coordinates = _coordinates(self.problem.model)
        new_coordinates = _coordinates(declaration)
        changed_coordinates = {
            name: (old, new_coordinates[name])
            for name, old in old_coordinates.items()
            if name in new_coordinates and old != new_coordinates[name]
        }

        def coordinate_bindings(subject=None):
            dependencies = (
                self._dependencies(subject)
                if isinstance(subject, Basic)
                else ((subject,) if isinstance(subject, str) else ())
            )
            selected = {}
            for name, (old, new) in changed_coordinates.items():
                coordinate_dependencies = [
                    path
                    for path in dependencies
                    if old in self._law(self._reference(path), frozenset({old})).free_symbols
                ]
                scope = name.rpartition(".")[0]
                relevant = not dependencies or any(
                    path == scope or path.startswith(scope + ".") for path in coordinate_dependencies
                )
                if relevant:
                    unchanged = [
                        path
                        for path, value in old_coordinates.items()
                        if value == old and new_coordinates.get(path) == old
                    ]
                    if any(
                        any(
                            dependency.startswith(path.rpartition(".")[0] + ".")
                            for dependency in coordinate_dependencies
                        )
                        for path in unchanged
                        if path.rpartition(".")[0]
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
            for old in value.free_symbols & set(global_coordinates):
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
                    for path in self._dependencies(constraint.left)
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
                                    coordinate_bindings(constraint.left)
                                    if constraint.right.free_symbols & set(global_coordinates)
                                    else {}
                                ),
                            },
                        ),
                    )
                )
        problem = replace(problem, constraints=(*problem.constraints, *observations))

        def rebound(claim):
            native_symbols = set().union(
                *(
                    value.free_symbols
                    for value in (claim.subject, claim.wrt, claim.domain, claim.bound, claim.expected, claim.at)
                    if isinstance(value, Basic)
                )
            )
            bindings = {
                **mapping,
                **(coordinate_bindings(claim.subject) if native_symbols & set(global_coordinates) else {}),
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
            tuple(replace(item, claim=rebound(item.claim)) for item in self.requirements),
            tuple((key, ProofHint(_rebind(hint.upper, {**mapping, **global_coordinates}))) for key, hint in self.hints),
            tuple(
                replace(
                    rule,
                    conclusion=rebound(rule.conclusion),
                    premises=tuple(replace(item, claim=rebound(item.claim)) for item in rule.premises),
                )
                for rule in self.theorems
            ),
            (*self.changes, *invalidated),
        )

    def _dependencies(self, value: Basic) -> tuple[str, ...]:
        symbols, pending = set(), list(value.free_symbols)
        definitions: dict[Symbol, set[Symbol]] = {}
        for item in self.problem.constraints:
            if item.kind is ConstraintKind.DEFINITION:
                definitions.setdefault(item.left, set()).update(item.right.free_symbols)
        while pending:
            symbol = pending.pop()
            if symbol not in symbols:
                symbols.add(symbol)
                pending.extend(definitions.get(symbol, ()))
        return tuple(name for name, symbol in self._fields.items() if symbol in symbols)

    def _declared(self):
        records = []
        for path, annotation in Requires.for_model(self.problem.model).items():
            prefix = path.rpartition(".")[0]
            for requirement in annotation.requirements:
                records.append((f"{path}/{requirement.claim.name}", requirement, prefix))
        records.extend((item.claim.name, item, "") for item in self.requirements)
        return records

    def _automatic(self):
        records = []
        definitions = {}
        for constraint in self.problem.constraints:
            if constraint.kind is ConstraintKind.DEFINITION:
                definitions.setdefault(constraint.left, []).append(constraint.right)
        for constraint in self.problem.constraints:
            if constraint.kind is not ConstraintKind.DEFINITION:
                continue
            path = ".".join(constraint.path)
            seen = set()

            def visit(value, points=None, bound=(), active=()):
                points = points or {}
                if isinstance(value, Subs):
                    local = {
                        symbol: point.subs(points, simultaneous=True)
                        for symbol, point in zip(value.variables, value.point)
                    }
                    visit(value.expr, {**points, **local}, bound, active)
                    for point in value.point:
                        visit(point, points, bound, active)
                    return
                if value in definitions:
                    if value not in active:
                        for source in definitions[value]:
                            visit(source, points, bound, (*active, value))
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
                            domain=coordinate_domain(variable),
                            order=len(value.variables),
                            at=points.get(variable),
                            name="derivative",
                        )
                    else:
                        claim = Claim("mixed_derivative", value, "theorem")
                    if claim not in seen:
                        seen.add(claim)
                        records.append(
                            (
                                f"{path}/auto:{len(seen)}",
                                Requirement(
                                    claim, path, "The native derivative requires differentiability, not necessarily C2."
                                ),
                                "",
                            )
                        )
                    bound = (*bound, *variables)
                if isinstance(value, Expr) and value.is_Pow and value.exp.is_negative is True:
                    denominator = self._law(value.base, frozenset(bound)).subs(points, simultaneous=True)
                    coordinates = set(denominator.atoms(Coordinate)) | (set(bound) & denominator.free_symbols)
                    if len(coordinates) <= 1:
                        variable = next(iter(coordinates), None)
                        claim = nonzero(
                            denominator, wrt=variable, domain=coordinate_domain(variable), name="denominator"
                        )
                    else:
                        claim = Claim("multivariate_denominator", denominator, "theorem")
                    if claim not in seen:
                        seen.add(claim)
                        records.append(
                            (
                                f"{path}/auto:{len(seen)}",
                                Requirement(
                                    claim, path, "The original denominator must be nonzero before cancellation."
                                ),
                                "",
                            )
                        )
                for arg in value.args:
                    visit(arg, points, bound, active)

            # Follow source definitions under operator scopes before algebra
            # can cancel an original division or erase a local derivative.
            visit(constraint.right)
        return records

    def check(self, *, max_steps: int = 100, requirements: tuple[Requirement, ...] = ()) -> ResearchReport:
        """Try bounded scalar rules; unknown does not mean a false proposition.

        The budget counts proof attempts, not wall-clock time inside the CAS.
        """
        if max_steps < 1:
            raise ModelInputError("max_steps must be positive")
        assumptions = tuple(
            self._bind(condition)
            for condition in (*self.assumptions, *sorted(global_assumptions, key=default_sort_key))
        )
        feasible = consistency(assumptions)
        observation_issue = ""
        if any(item.kind is not ConstraintKind.DEFINITION for item in self.problem.constraints):
            empty = self.family.is_empty
            if empty is True:
                feasible = False
                observation_issue = "The recorded observations and model domains have no admissible state."
            elif empty is None:
                if feasible is True:
                    feasible = None
                observation_issue = "Joint feasibility of the recorded observations remains unresolved."
        records_by_id: dict[str, tuple[str, Requirement, str]] = {}
        for record in [*self._declared(), *self._automatic(), *((item.claim.name, item, "") for item in requirements)]:
            key = record[0]
            if key in records_by_id and records_by_id[key] != record:
                raise ModelDefinitionError(f"Requirement IDs must be unique within a research case: {key}")
            records_by_id[key] = record
        records = tuple(records_by_id.values())
        ids = tuple(records_by_id)
        issues = [] if feasible is not False else ["The assumptions are inconsistent; proofs were skipped."]
        if observation_issue:
            issues.append(observation_issue)
        issues.extend(self.changes)
        active, steps = set(), [0]

        def prove(key, requirement, prefix):
            claim = requirement.claim
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
            subject = reference(claim.subject) if isinstance(claim.subject, str) else native(claim.subject)
            self._require_symbols(subject)
            protected = frozenset() if variable is None else frozenset({variable})
            expression = self._bind(subject, protected)
            expected = (
                native(claim.expected(reference))
                if callable(claim.expected)
                else (native(claim.expected) if claim.expected is not None else None)
            )
            expected_template = expected
            if expected is not None:
                self._require_symbols(expected)
                expected = self._bind(expected, protected)

            def payload(value):
                if value is None:
                    return None
                value = native(value)
                self._require_symbols(value)
                return self._bind(value, protected)

            domain, bound, point = payload(claim.domain), payload(claim.bound), payload(claim.at)
            if variable is not None and variable in domain.free_symbols:
                raise ModelDefinitionError("A quantified domain cannot depend on its own coordinate")
            resolved = replace(
                claim, subject=expression, wrt=variable, expected=expected, domain=domain, bound=bound, at=point
            )
            dependencies = self._dependencies(subject)
            token = (claim, prefix)
            if feasible is False or steps[0] >= max_steps or token in active:
                return Evidence(
                    key,
                    requirement,
                    expression,
                    variable,
                    None,
                    ("Inconsistent premises, exhausted proof budget, or circular theorem dependency.",),
                    dependencies,
                    resolved_claim=resolved,
                    expected_template=expected_template,
                )
            steps[0] += 1
            theorem = next((rule for rule in self.theorems if rule.conclusion == claim), None)
            if theorem is not None:
                active.add(token)
                try:
                    premises = tuple(prove(f"{key}/{item.claim.name}", item, prefix) for item in theorem.premises)
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
                    resolved_claim=resolved,
                    expected_template=expected_template,
                )
            canonical = (
                (f"{prefix}.{claim.subject}" if prefix else claim.subject) if isinstance(claim.subject, str) else ""
            )
            hint = dict(self.hints).get(key, dict(self.hints).get(f"{canonical}/{claim.name}"))
            if hint is not None:
                hint = ProofHint(cast(Expr, payload(hint.upper)))
            attempt_key = (
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
            )
            attempts = self._context.attempts
            result = attempts.get(attempt_key)
            if result is None:
                try:
                    result = attempt(resolved, expression, variable, expected, assumptions, hint)
                except (NotImplementedError, ValueError, TypeError, PolynomialError) as error:
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
                resolved_claim=resolved,
                hint=None if hint is None else hint.upper,
                expected_template=expected_template,
            )

        checks = tuple(prove(*record) for record in records)
        unused = set(dict(self.hints)) - set(ids)
        if unused:
            issues.append(f"Unused proof hints: {', '.join(sorted(unused))}")
        labels: dict[Symbol, str] = {}
        for path, symbol in self._fields.items():
            previous = labels.get(symbol)
            if previous is None or (path.count("."), len(path)) < (previous.count("."), len(previous)):
                labels[symbol] = path
        for constraint in self.problem.constraints:
            for coordinate in constraint.right.atoms(Coordinate):
                labels.setdefault(coordinate, coordinate.name)
        return ResearchReport(
            self.problem.model.__name__, checks, assumptions, feasible, tuple(issues), tuple(labels.items())
        )

    def explain(self, claim_id: str) -> Evidence:
        return self.check().explain(claim_id)

    def sample(self, subject: str | Basic, *, wrt: str | Symbol, points) -> Experiment:
        variable = self._reference(wrt) if isinstance(wrt, str) else wrt
        expression = self.expression(subject, wrt=variable)
        samples = tuple(sympify(point) for point in points)
        values = tuple(expression.subs(variable, point).evalf() for point in samples)
        if any(value.is_finite is not True or value.is_real is not True for value in values):
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
        report = self.check(max_steps=1000, requirements=method.requirements)
        uses: set[str] = set()
        for path in method.uses:
            uses.update(self._dependencies(self._reference(path)))
        required = {item.claim.name for item in method.requirements}

        def consumed(item):
            if "/auto:" in item.id:
                return item.requirement.consumer in method.uses
            return (
                item.id in required
                or item.requirement.consumer in uses
                or item.id.partition("/")[0] in uses
                or item.requirement.consumer == method.name
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

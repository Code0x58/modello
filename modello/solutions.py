"""Joint scalar solution families with explicit domains and conditional fallbacks.

Reductions are equivalences: directed substitution and rational affine
elimination. Univariate solving is guarded against external coefficients.
No solver candidate list is promoted to a complete nonlinear solution family.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, cast

from sympy import (
    And,
    Basic,
    ConditionSet,
    Contains,
    Derivative,
    Dummy,
    Eq,
    Expr,
    FiniteSet,
    Float,
    ImageSet,
    Interval,
    Lambda,
    Ne,
    Poly,
    ProductSet,
    Q,
    S,
    Set,
    Symbol,
    Tuple,
    imageset,
    linear_eq_to_matrix,
    linsolve,
    preorder_traversal,
    solveset,
    sympify,
)
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.function import AppliedUndef
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.solvers.solveset import NonlinearError

from modello.diagnostics import (
    Candidate,
    ConstraintKind,
    FieldValue,
    ModelField,
    ModelProblem,
    ModelT,
    _canonical_fields,
    _domain,
    _FieldPrinter,
)
from modello.errors import ModelInputError, ModelValidationError, SolutionSelectionError, UnresolvedSolutionError

if TYPE_CHECKING:
    from modello import Modello, ScalarValue


@dataclass(frozen=True)
class SolutionIssue:
    """An unsuccessful simplification; the original condition is still retained."""

    operation: str
    error: Exception


@dataclass(frozen=True, slots=True)
class _FamilyField:
    """Keep a quantity, its chart symbol, and its reduced expression together."""

    field: ModelField
    symbol: Symbol
    expression: Expr


def _has_binders(expression: Basic) -> bool:
    return any(
        isinstance(node, Derivative) or bool(getattr(node, "bound_symbols", ()))
        for node in preorder_traversal(expression)
    )


def _field_domain(symbol: Symbol) -> Set:
    """An explicit finite-scalar domain, including less common assumptions."""
    if symbol.is_complex is False:
        return S.EmptySet
    assumptions: dict[str, bool] = {}
    for predicate in ("integer", "rational", "real", "positive", "negative", "nonnegative", "nonpositive"):
        if getattr(symbol, f"is_{predicate}") is True:
            assumptions[predicate] = True
    if symbol.is_zero is not None:
        assumptions["zero"] = symbol.is_zero
    implied = Dummy("domain", complex=True, **assumptions).assumptions0
    variable = Dummy("value")
    extra = []
    for predicate, expected in symbol.assumptions0.items():
        if implied.get(predicate) == expected:
            continue
        if not hasattr(Q, predicate):
            raise ModelInputError(f"Solution families do not yet support the {predicate!r} assumption")
        condition = getattr(Q, predicate)(variable)
        extra.append(condition if expected else ~condition)
    base = FiniteSet(0) if symbol.is_zero is True else _domain(symbol)
    return ConditionSet(variable, And(*extra), base)


def _evaluate_predicates(condition: Boolean) -> Boolean:
    """Use the same SymPy scalar predicates as the model's declared domains."""
    substitutions = {}
    for predicate in condition.atoms(AppliedPredicate):
        if len(predicate.arguments) == 1:
            known = getattr(predicate.arguments[0], f"is_{predicate.function.name}", None)
            if known is not None:
                substitutions[predicate] = S.true if known else S.false
    return condition.xreplace(substitutions)


def _membership(expression: Expr, domain: Set, parameters: tuple[Symbol, ...]) -> Boolean:
    """Keep membership explicit; Contains.as_set() is not an inverse image."""
    if isinstance(domain, Interval) and expression.is_extended_real is None:
        # Interval.contains eagerly introduces ordering. Keep real membership
        # explicit when a complex input may or may not produce a real value.
        return Contains(expression, domain, evaluate=False)
    result = _evaluate_predicates(domain.contains(expression))
    if result in (S.true, S.false):
        return result
    # Rational expressions of finite complex coordinates are finite precisely
    # away from denominator zeros. This also preserves reciprocal-cycle holes.
    if parameters and domain is S.Complexes and expression.is_rational_function(*parameters):
        numerator, denominator = expression.as_numer_denom()
        if numerator.is_polynomial(*parameters) and denominator.is_polynomial(*parameters):
            coefficients = [*Poly(numerator, *parameters).coeffs(), *Poly(denominator, *parameters).coeffs()]
            if all(c.is_number is True and c.is_finite is True and c.is_complex is True for c in coefficients):
                return Ne(denominator, 0)
    if isinstance(domain, Interval) and expression.is_extended_real is True:
        return domain.as_relational(expression)
    return result


def _parameter_set(
    parameters: tuple[Symbol, ...], domains: tuple[Set, ...], condition: Boolean, issues: list[SolutionIssue]
) -> Set:
    if not parameters:
        return ConditionSet(Tuple(), condition, FiniteSet(Tuple()))
    if len(parameters) != 1:
        return ConditionSet(Tuple(*parameters), condition, ProductSet(*domains))
    (parameter,) = parameters
    (domain,) = domains
    clauses = condition.args if isinstance(condition, And) else (condition,)
    for clause in clauses:
        if clause is S.true:
            continue
        if clause is S.false:
            return S.EmptySet
        # Generic symbolic coefficients and unevaluated functions can conceal
        # singular cases. Retaining the condition is exact in all those cases.
        constants = (node for node in preorder_traversal(clause) if isinstance(node, Expr) and not node.free_symbols)
        exact_constants = all(
            value.is_number is True and value.is_finite is True and value.is_zero is not None for value in constants
        )
        if clause.free_symbols <= {parameter} and not clause.has(AppliedUndef, Contains) and exact_constants:
            try:
                if isinstance(clause, Eq):
                    domain &= solveset(clause.lhs - clause.rhs, parameter, domain)
                    continue
                if isinstance(clause, Ne):
                    domain -= solveset(clause.lhs - clause.rhs, parameter, domain)
                    continue
                if isinstance(clause, Relational) and domains[0].is_subset(S.Reals) is True:
                    domain &= clause.as_set()
                    continue
            except Exception as error:
                issues.append(SolutionIssue("univariate reduction", error))
        domain = ConditionSet(parameter, clause, domain)
    return domain


@dataclass(frozen=True, repr=False)
class SolutionFamily(Generic[ModelT]):
    """All finite commutative scalar states of one recorded model problem.

    ``parameters`` are chart coordinates, not a claim about dimension. Their
    ``parameter_domain`` may retain coupled or unresolved conditions.
    """

    problem: ModelProblem[ModelT]
    parameters: tuple[ModelField, ...]
    parameter_domain: Set
    issues: tuple[SolutionIssue, ...]
    _bindings: tuple[_FamilyField, ...]
    conditions: tuple[Boolean, ...] = ()
    _preferred: tuple[ModelField, ...] = ()

    @classmethod
    def _prepare(
        cls,
        problem: ModelProblem[ModelT],
        preferred: tuple[ModelField, ...] = (),
        conditions: tuple[Boolean, ...] = (),
    ) -> SolutionFamily[ModelT]:
        for field in preferred:
            problem._require_field(field)
        fields = _canonical_fields(problem)
        if any(field.symbol.is_commutative is not True for field in fields):
            raise ModelInputError("Solution families currently require commutative scalar fields")
        original = [c.right for c in problem.constraints] + list(conditions)
        if any(not isinstance(c.right, Expr) for c in problem.constraints) or any(expr.has(Float) for expr in original):
            raise ModelInputError("Solution families require exact scalar expressions; use Rational for decimal values")
        if any(_has_binders(expression) for expression in original):
            raise ModelInputError("Evaluate bound-variable expressions, such as integrals and derivatives, first")
        domains = tuple(_field_domain(field.symbol) for field in fields)
        # Work symbols explicitly carry the finite-scalar universe, in addition
        # to the model's assumptions. Those assumptions remain in explicit sets.
        symbols = tuple(
            Dummy(
                ".".join(field.path),
                **({**field.symbol.assumptions0, "complex": True} if domain is not S.EmptySet else {}),
            )
            for field, domain in zip(fields, domains, strict=True)
        )
        bindings = {field.symbol: symbol for field, symbol in zip(fields, symbols, strict=True)}
        protected = {bindings[field.symbol] for field in preferred}
        equations = [(bindings[c.left], c.right.xreplace(bindings)) for c in problem.constraints]
        equations.extend(
            (condition.lhs.xreplace(bindings), condition.rhs.xreplace(bindings))
            for condition in conditions
            if isinstance(condition, Eq)
        )
        expressions: dict[Symbol, Expr] = {symbol: symbol for symbol in symbols}
        remaining = list(symbols)
        # Eliminate only an explicitly isolated quantity, preserving all other
        # equations and every eliminated quantity's domain below.
        while True:
            substitution = None
            for left, right in equations:
                a, b = left.xreplace(expressions), right.xreplace(expressions)
                if a in remaining and a not in protected and a not in b.free_symbols:
                    substitution = (a, b)
                    break
                if b in remaining and b not in protected and b not in a.free_symbols:
                    substitution = (b, a)
                    break
            if substitution is None:
                break
            symbol, value = substitution
            expressions = {key: expr.xreplace({symbol: value}) for key, expr in expressions.items()}
            remaining.remove(symbol)
        residuals = [left.xreplace(expressions) - right.xreplace(expressions) for left, right in equations]
        issues: list[SolutionIssue] = []
        # Exact rational affine elimination also handles joining equations, with
        # preferred coordinates ordered last so they remain free where possible.
        ordered = tuple(s for s in remaining if s not in protected) + tuple(s for s in remaining if s in protected)
        affine = []
        for residual in residuals:
            if not ordered:
                break
            try:
                matrix, constant = linear_eq_to_matrix([residual], ordered)
            except NonlinearError:
                continue
            if all(value.is_Rational for value in (*matrix, *constant)):
                affine.append(residual)
        if affine:
            try:
                solved = linsolve(affine, ordered)
                if solved is S.EmptySet:
                    domains = tuple(S.EmptySet for _ in fields)
                else:
                    (row,) = solved
                    replacements = dict(zip(ordered, row, strict=True))
                    expressions = {key: expr.xreplace(replacements) for key, expr in expressions.items()}
                    remaining = [symbol for symbol in remaining if replacements[symbol] == symbol]
            except Exception as error:
                issues.append(SolutionIssue("rational affine reduction", error))
        parameters = tuple(remaining)
        values = tuple(expressions[symbol] for symbol in symbols)
        predicates = [Eq(left.xreplace(expressions), right.xreplace(expressions)) for left, right in equations]
        predicates.extend(condition.xreplace(bindings).xreplace(expressions) for condition in conditions)
        predicates.extend(_membership(value, domain, parameters) for value, domain in zip(values, domains, strict=True))
        parameter_fields = tuple(fields[symbols.index(symbol)] for symbol in parameters)
        parameter_domains = tuple(domains[symbols.index(symbol)] for symbol in parameters)
        domain = _parameter_set(parameters, parameter_domains, And(*predicates), issues)
        return cls(
            problem,
            parameter_fields,
            domain,
            tuple(issues),
            tuple(_FamilyField(*binding) for binding in zip(fields, symbols, values, strict=True)),
            conditions,
            preferred,
        )

    def __repr__(self) -> str:
        return (
            f"SolutionFamily({self.problem.model.__name__}, {self.problem.name!r}, parameters={len(self.parameters)})"
        )

    @property
    def is_empty(self) -> bool | None:
        """Return SymPy's emptiness verdict, preserving an undecided result."""
        return self.parameter_domain.is_empty

    @property
    def symbols(self) -> ModelT:
        """Access the source problem's typed, read-only symbol namespace."""
        return self.problem.symbols

    @property
    def _coordinates(self) -> tuple[Symbol, ...]:
        return tuple(binding.symbol for binding in self._bindings if binding.field in self.parameters)

    def _printer(self, expression: Basic, scope: tuple[str, ...] = ()) -> _FieldPrinter:
        referenced = frozenset(expression.atoms(Symbol))
        symbols = frozenset().union(
            expression.free_symbols,
            *(binding.expression.free_symbols for binding in self._bindings),
            *(constraint.symbols for constraint in self.problem.constraints),
        )
        bindings = {binding.field.symbol: binding.symbol for binding in self._bindings}
        fields = tuple(
            alias for field in self.problem.fields for alias in (field, ModelField(field.path, bindings[field.symbol]))
        )
        return _FieldPrinter(fields, symbols, scope, referenced)

    def format(self, expression: Basic) -> str:
        """Format chart symbols as the model's named paths without substitution."""
        return str(self._printer(expression).doprint(expression))

    def __str__(self) -> str:
        lines = [f"{self.problem.model.__name__}({self.problem.name!r}): solution family"]
        if self.parameters:
            lines.append("  Coordinates: " + ", ".join(".".join(field.path) for field in self.parameters))
        else:
            lines.append("  No remaining coordinates.")
        lines.append(f"  Coordinate domain: {self.format(self.parameter_domain)}")
        entries = []
        for binding in self._bindings:
            field, expr = binding.field, binding.expression
            printer = self._printer(expr, field.path[:-1])
            path = ".".join(field.path[len(printer.scope) :])
            text = f"{path} [coordinate]" if field in self.parameters else f"{path} = {printer.doprint(expr)}"
            entries.append((field.path[:-1], text))
        lines.extend(self.problem._tree_lines(tuple(entries), 2))
        lines.extend(
            f"  Unresolved {issue.operation}: {type(issue.error).__name__}: {issue.error}" for issue in self.issues
        )
        return "\n".join(lines)

    def expression(self, expression: ScalarValue) -> Expr:
        """Express a scalar in the remaining coordinates, using public symbols."""
        reduced = self._chart_expression(expression)
        reverse = {binding.symbol: binding.field.symbol for binding in self._bindings}
        return cast(Expr, reduced.xreplace(reverse))

    def _scalar(self, value: object) -> Expr:
        from modello import InstanceDummy

        try:
            expression = sympify(value)
        except (TypeError, ValueError, AttributeError) as error:
            raise ModelInputError("Expected a scalar expression") from error
        if not isinstance(expression, Expr) or expression.has(Float):
            raise ModelInputError("Expected an exact scalar expression")
        if _has_binders(expression):
            raise ModelInputError("Evaluate bound-variable expressions, such as integrals and derivatives, first")
        allowed = {field.symbol for field in self.problem.fields} | set().union(
            *(c.symbols for c in self.problem.constraints)
        )
        if any(isinstance(symbol, InstanceDummy) and symbol not in allowed for symbol in expression.free_symbols):
            raise ModelValidationError("The expression refers to another model problem; use family.symbols")
        return expression

    def _chart_expression(self, expression: ScalarValue) -> Expr:
        bindings = {binding.field.symbol: binding.expression for binding in self._bindings}
        return cast(Expr, self._scalar(expression).xreplace(bindings))

    def values(self, *expressions: ScalarValue) -> Set:
        """Return possible scalar values, or coupled tuples for multiple outputs."""
        if not expressions:
            raise ModelInputError("Select at least one output expression")
        outputs = tuple(self._chart_expression(expr) for expr in expressions)
        output = outputs[0] if len(outputs) == 1 else Tuple(*outputs)
        parameters = self._coordinates
        if not parameters:
            unit = Dummy("unit")
            base = ConditionSet(unit, self.parameter_domain.contains(Tuple()), FiniteSet(0))
            return imageset(Lambda(unit, output), base)
        signature = parameters[0] if len(parameters) == 1 else (Tuple(*parameters),)
        try:
            return imageset(Lambda(signature, output), self.parameter_domain)
        except (NotImplementedError, ValueError):
            return ImageSet(Lambda(signature, output), self.parameter_domain)

    def where(self, *conditions: Boolean | bool) -> SolutionFamily[ModelT]:
        """Retain further symbolic conditions; use Eq for equality observations."""
        checked = []
        for condition in conditions:
            try:
                value = sympify(condition)
            except (TypeError, ValueError, AttributeError) as error:
                raise ModelInputError("Conditions must be relations or booleans; use Eq for equality") from error
            if not isinstance(value, Boolean) or isinstance(value, Expr):
                raise ModelInputError("Conditions must be relations or booleans; use Eq for equality")
            if value.has(Float):
                raise ModelInputError("Conditions require exact values; use Rational for decimals")
            # Validate identities using each scalar atom in a relation.
            for symbol in value.free_symbols:
                self._scalar(symbol)
            checked.append(value)
        return self._prepare(self.problem, self._preferred, (*self.conditions, *checked))

    def observe(self, **inputs: ScalarValue | Modello) -> SolutionFamily[ModelT]:
        """Add constructor-style scalar or component observations immutably.

        Observations accumulate as constraints. Use ``where(Eq(...))`` for a
        relation involving nested quantities without constructing a component.
        """
        from modello import _with_inputs

        problem = _with_inputs(self.problem, inputs)
        for constraint in problem.constraints[len(self.problem.constraints) :]:
            self._scalar(constraint.right)
        return self._prepare(problem, self._preferred, self.conditions)

    def cases(self) -> tuple[ModelT, ...]:
        """Materialize a finite family as independently bound, checked models.

        Like solved components, cases use Modello's population mechanism;
        custom constructor hooks are not invoked.
        """
        from modello import _populate, inspect_model

        if self.parameter_domain is S.EmptySet:
            return ()
        if not isinstance(self.parameter_domain, FiniteSet):
            raise UnresolvedSolutionError(self, "The coordinate domain is not an explicit finite set.")
        parameters = self._coordinates
        cases = []
        for point in self.parameter_domain:
            coordinates = (point,) if len(parameters) == 1 else tuple(point)
            replacements = dict(zip(parameters, coordinates, strict=True))
            values = tuple(
                FieldValue(binding.field, binding.expression.xreplace(replacements)) for binding in self._bindings
            )
            candidate = Candidate(values)
            check = self.problem.check(candidate)
            assignment = {value.field.symbol: value.value for value in values}
            if check.verdict != "satisfied" or any(
                _evaluate_predicates(condition.xreplace(assignment)) is not S.true for condition in self.conditions
            ):
                raise UnresolvedSolutionError(self, "A state could not be verified against the original model.", check)
            fresh = inspect_model(self.problem.model, self.problem.name)
            solution = {fresh.field(*value.field.path).symbol: value.value for value in values}
            input_symbols = {c.left for c in self.problem.constraints if c.kind is ConstraintKind.INPUT}
            inputs = frozenset(
                fresh.field(*binding.field.path).symbol
                for binding in self._bindings
                if binding.field.symbol in input_symbols
            )
            instance = object.__new__(self.problem.model)
            _populate(instance, fresh, solution, inputs)
            cases.append(instance)
        return tuple(cases)

    def one(self) -> ModelT:
        """Select exactly one checked state, retaining this family on failure."""
        cases = self.cases()
        if len(cases) != 1:
            raise SolutionSelectionError(self, len(cases))
        return cases[0]

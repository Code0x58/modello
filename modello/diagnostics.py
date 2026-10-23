"""Problem preparation and opt-in conservative diagnosis.

Consistency requires a checked, closed witness; inconsistency requires an exact
contradiction. Solver failure alone establishes neither.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, Mapping, cast

from sympy import (
    Basic,
    ConditionSet,
    Derivative,
    Eq,
    Expr,
    FiniteSet,
    Float,
    I,
    Integral,
    Limit,
    Matrix,
    S,
    Set,
    Symbol,
    linear_eq_to_matrix,
    simplify,
    solve,
    solveset,
    sqrt,
)
from sympy.core.function import AppliedUndef
from sympy.solvers.solveset import NonlinearError

from modello._expressions import _evaluate, _substitute
from modello._inputs import _InputCapture
from modello._model import ModelloMeta, _ModelReference
from modello._paths import _path_label
from modello._problem import (
    AssumptionCheck,
    Candidate,
    CandidateCheck,
    ConstraintKind,
    FieldPath,
    FieldValue,
    ModelConstraint,
    ModelField,
    ModelT,
    ResidualCheck,
    Status,
    Verdict,
    _algebra,
    _canonical_fields,
    _domain,
    _FieldPrinter,
)
from modello.errors import ModelInputError, ModelSolveError as ModelSolveError, ModelValidationError
from modello.solutions import SolutionFamily
from modello.vectors import Vector

if TYPE_CHECKING:
    from modello import Modello, ScalarValue, SymbolicScalar, VectorValue


@dataclass(frozen=True)
class SolutionSet:
    """Necessary values for one field, after substituting forced ground values.

    ``values`` solves only the recorded ``constraints`` in ``domain``. Other
    equations and predicates still require checking against the whole problem.
    Infinite families and unresolved ConditionSets are retained as SymPy sets.
    """

    field: ModelField
    domain: Set
    values: Set
    constraints: tuple[ModelConstraint, ...]
    substitutions: tuple[FieldValue, ...]
    solver_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentView:
    """A structural view, including equations crossing this component's boundary."""

    path: FieldPath
    model: type[Modello]
    fields: tuple[ModelField, ...]
    owned_constraints: tuple[ModelConstraint, ...]
    boundary_constraints: tuple[ModelConstraint, ...]


@dataclass(frozen=True)
class LinearCertificate:
    """An exact weighted sum of original residuals equal to a nonzero constant."""

    terms: tuple[tuple[ModelConstraint, Expr], ...]
    constant: Expr

    def verify(self) -> bool:
        """Independently check the certificate against its original equations."""
        residual = sum(weight * (constraint.left - constraint.right) for constraint, weight in self.terms)
        return simplify(_evaluate(residual - self.constant)).is_zero is True and self.constant.is_zero is False


@dataclass(frozen=True)
class LinearAnalysis:
    """Rank of the exact rational affine part, ignoring variable domains."""

    fields: tuple[ModelField, ...]
    constraints: tuple[ModelConstraint, ...]
    rank: int
    augmented_rank: int
    complete: bool
    certificate: LinearCertificate | None = None

    @property
    def nullity(self) -> int:
        """Return the affine equality nullity, not a nonlinear/domain dimension."""
        return len(self.fields) - self.rank


@dataclass(frozen=True)
class RemovalCheck:
    """The evidenced outcome of removing specified constraints."""

    removed: tuple[ModelConstraint, ...]
    status: Status
    reason: str
    witness: CandidateCheck | None


@dataclass(frozen=True)
class InputCheck(RemovalCheck):
    """A removal probe for one entire constructor argument, including copied values."""

    input: str


@dataclass(frozen=True)
class Conflict:
    """A proven conflict relative to all declared domains; minimality may be unknown."""

    constraints: tuple[ModelConstraint, ...]
    irreducible: bool | None
    reason: str
    linear_certificate: LinearCertificate | None = None
    removal_checks: tuple[RemovalCheck, ...] = ()


@dataclass(frozen=True, repr=False)
class ModelProblem(Generic[ModelT]):
    """A frozen snapshot shared by preparation, construction, and failure handling."""

    model: type[ModelT]
    name: str
    fields: tuple[ModelField, ...]
    constraints: tuple[ModelConstraint, ...]
    candidates: tuple[Candidate, ...] = ()

    def _capture_inputs(self, **value_map: ScalarValue | VectorValue | Modello) -> ModelProblem[ModelT]:
        """Append constructor inputs without rebinding this snapshot."""
        if not value_map:
            return self
        constraints = _InputCapture(self).capture(value_map)
        return replace(self, constraints=constraints, candidates=())

    def _fresh(self) -> ModelProblem[ModelT]:
        """Prepare the same declaration with fresh identities and no observations."""
        return _prepare_problem(self.model, self.name)

    @cached_property
    def symbols(self) -> ModelT:
        """Access this problem's independent quantities through typed components.

        Like references in a model declaration, this read-only namespace exposes
        scalar symbols and nested components, rather than solved model methods.
        Derived fields also have their own symbols, so they can be observed.
        """
        return cast(ModelT, _ModelReference(self.model, {field.path: field.symbol for field in self.fields}))

    @cached_property
    def vectors(self) -> Mapping[FieldPath, Vector]:
        """Read-only vector fields, retaining paths, order, aliases, and empty shapes."""
        fields = {field.path: field.symbol for field in self.fields}
        bindings = {dummy: fields[path] for path, dummy in self.model._modello_definition.fields.items()}
        return MappingProxyType(
            {path: value.xreplace(bindings) for path, value in self.model._modello_definition.vectors.items()}
        )

    def vector(self, *path: str) -> Vector:
        """Look up a whole vector by its component path, without printed-name matching."""
        return self.vectors[path]

    @cached_property
    def vector_definitions(self) -> Mapping[FieldPath, Vector]:
        """Read-only, unevaluated whole-vector formulas, before scalar expansion."""
        fields = {field.path: field.symbol for field in self.fields}
        bindings = {dummy: fields[path] for path, dummy in self.model._modello_definition.fields.items()}
        return MappingProxyType(
            {
                path: value.xreplace(bindings)
                for path, value in self.model._modello_definition.vector_definitions.items()
            }
        )

    def __repr__(self) -> str:
        return (
            f"ModelProblem({self.model.__name__}, {self.name!r}, "
            f"fields={len(self.fields)}, constraints={len(self.constraints)})"
        )

    def format(self, expression: Basic) -> str:
        """Render paths from the model root without changing the expression."""
        return str(self._printer(expression).doprint(expression))

    def latex(
        self,
        expression: Basic,
        *,
        symbol_names: Mapping[Symbol, str] | None = None,
        vector_names: Mapping[Vector, str] | None = None,
    ) -> str:
        """Export scoped paths and vector notation, with optional LaTeX name overrides."""
        from modello._printing import _latex

        printer = self._printer(expression)
        return _latex(expression, printer, printer.vector_labels, symbol_names, vector_names)

    def _printer(self, *expressions: Basic, scope: FieldPath = ()) -> _FieldPrinter:
        # Bound variables in a ConditionSet need the same labels as free symbols.
        referenced: frozenset[Symbol] = frozenset().union(*(expression.atoms(Symbol) for expression in expressions))
        symbols = frozenset().union(*(c.symbols for c in self.constraints), referenced)
        return _FieldPrinter(self.fields, symbols, scope, referenced, self.vectors.items())

    def __str__(self) -> str:
        return "\n".join([f"{self.model.__name__}({self.name!r})", *self._constraint_lines(self.constraints)])

    def _tree_lines(self, entries: tuple[tuple[FieldPath, str], ...], indent: int) -> list[str]:
        def visit(model: type[Modello], path: FieldPath, depth: int) -> list[str]:
            lines = [" " * depth + text for owner, text in entries if owner == path]
            for name, vector in model._modello_namespace.vectors.items():
                vector_path = (*path, name)
                children = [text for owner, text in entries if owner == vector_path]
                if children:
                    lines.append(f"{' ' * depth}{name}: Vector[{len(vector)}]")
                    lines.extend(" " * (depth + 2) + text for text in children)
            for name, component in model._modello_namespace.nested_models.items():
                child_path = (*path, name)
                if any(owner[: len(child_path)] == child_path for owner, _ in entries):
                    child = component._modello_model
                    lines.append(f"{' ' * depth}{name}: {child.__name__}")
                    lines.extend(visit(child, child_path, depth + 2))
            return lines

        return visit(self.model, (), indent)

    def _constraint_lines(self, constraints: tuple[ModelConstraint, ...], indent: int = 2) -> list[str]:
        entries = []
        for c in constraints:
            printer = self._printer(c.right, scope=c.path[:-1])
            if c.kind is ConstraintKind.DEFINITION:
                source = "definition"
            elif c.kind is ConstraintKind.INFERRED:
                source = f"inferred in {c.input!r}"
            else:
                source = f"{c.kind} {c.input!r}"
            path = _path_label(c.path[len(printer.scope) :])
            entries.append((c.path[:-1], f"c{c.index}: {path} = {printer.doprint(c.right)} [{source}]"))
        return self._tree_lines(tuple(entries), indent)

    def field(self, *path: str | int) -> ModelField:
        """Look up a declared path, without matching symbols by printed name."""
        normalized = tuple(str(part) for part in path)
        for field in self.fields:
            if field.path == normalized:
                return field
        raise KeyError(_path_label(normalized))

    def aliases(self, field: ModelField) -> tuple[ModelField, ...]:
        """Find all paths to the same quantity in this problem."""
        self._require_field(field)
        return tuple(other for other in self.fields if other.symbol == field.symbol)

    def uses(self, field: ModelField) -> tuple[ModelConstraint, ...]:
        """Find constraints mentioning a quantity, in either equation direction."""
        self._require_field(field)
        return tuple(constraint for constraint in self.constraints if field.symbol in constraint.symbols)

    def _require_field(self, field: ModelField) -> None:
        if field not in self.fields:
            raise ModelValidationError("The field belongs to another problem")

    def at(self, *path: str) -> ComponentView:
        """Inspect a component subtree and its boundary; this does not isolate a solve."""
        model: type[Modello] = self.model
        for part in path:
            child = model._modello_namespace.nested_models.get(part)
            if child is None:
                raise KeyError(".".join(path))
            model = child._modello_model
        fields = tuple(field for field in self.fields if field.path[: len(path)] == path)
        symbols = {field.symbol for field in fields}
        owned = tuple(c for c in self.constraints if c.path[:-1][: len(path)] == path)
        boundary = tuple(c for c in self.constraints if c not in owned and c.symbols & symbols)
        return ComponentView(path, model, fields, owned, boundary)

    def without(self, *inputs: str) -> ModelProblem[ModelT]:
        """Remove whole recorded constructor inputs, keeping definitions and domains."""
        existing = {c.input for c in self.constraints if c.input is not None}
        if set(inputs) - existing:
            raise KeyError("Unknown constructor input")
        return replace(self, constraints=tuple(c for c in self.constraints if c.input not in inputs), candidates=())

    def check(self, candidate: Candidate) -> CandidateCheck:
        """Audit a candidate against original equations and all declared domains."""
        return _audit(self, self.constraints, candidate)

    def analyse(self, *, conflicts: bool = False, relaxations: bool = False, max_checks: int = 100) -> ProblemAnalysis:
        """Run explicit exact checks and bounded searches; unknown never means inconsistent.

        ``max_checks`` bounds distinct subset assessments, not elapsed time.
        Domain assumptions remain fixed throughout. Relaxations remove one
        complete constructor argument and require a validated closed witness.
        """
        return _Analyser(self, max_checks).run(conflicts, relaxations)

    def solutions(self, *parameters: SymbolicScalar | Vector) -> SolutionFamily[ModelT]:
        """Explore the joint family, optionally retaining chosen parameter fields.

        This explicit operation can invoke exact symbolic solvers. It preserves
        unresolved conditions rather than choosing a candidate or branch.
        """
        fields = []
        coordinates = tuple(
            entry
            for parameter in parameters
            for entry in (parameter if isinstance(parameter, Vector) else (parameter,))
        )
        for parameter in coordinates:
            if not isinstance(parameter, Symbol):
                raise ModelInputError("A preferred coordinate must be one of this problem's field symbols")
            field = next((field for field in self.fields if field.symbol == parameter), None)
            if field is None:
                raise ModelValidationError("The coordinate belongs to another model problem; use problem.symbols")
            if field not in fields:
                fields.append(field)
        return SolutionFamily._prepare(self, tuple(fields))


def _prepare_problem(
    model: type[ModelT], name: str, /, **value_map: ScalarValue | VectorValue | Modello
) -> ModelProblem[ModelT]:
    """Capture a declaration and its constructor inputs as one bound snapshot."""
    if not isinstance(model, ModelloMeta):
        raise ModelInputError("inspect_model requires a Modello class and constructor inputs")
    definition = model._modello_definition
    bindings = {d: d.bound(name) for d in dict.fromkeys(definition.fields.values())}
    fields = tuple(ModelField(path, bindings[d]) for path, d in definition.fields.items())
    constraints = [
        ModelConstraint(
            i,
            constraint.path,
            ConstraintKind.DEFINITION,
            bindings[constraint.left],
            constraint.right.xreplace(bindings),
        )
        for i, constraint in enumerate(definition.constraints)
    ]
    problem = ModelProblem(model, name, fields, tuple(constraints))
    return problem._capture_inputs(**value_map)


@dataclass(frozen=True)
class ProblemAnalysis:
    """Evidence-backed conclusions, with the scope and unfinished work retained."""

    problem: ModelProblem
    status: Status
    reason: str
    linear: LinearAnalysis | None
    deductions: CandidateCheck
    candidates: tuple[CandidateCheck, ...]
    witness: CandidateCheck | None
    conflict: Conflict | None
    input_checks: tuple[InputCheck, ...]
    checks: int
    budget_exhausted: bool
    solution_sets: tuple[SolutionSet, ...] = ()

    @property
    def relaxations(self) -> tuple[InputCheck, ...]:
        """Return just the input removals with a checked closed witness."""
        return tuple(check for check in self.input_checks if check.witness is not None)

    def __str__(self) -> str:
        lines = [f"{self.problem.model.__name__}({self.problem.name!r}): {self.status}", f"  {self.reason}"]
        if self.linear is not None:
            scope = "all equations" if self.linear.complete else "affine subset only"
            lines.append(
                f"  Exact equality rank {self.linear.rank}; augmented rank {self.linear.augmented_rank}; "
                f"nullity {self.linear.nullity} ({scope}, domains excluded)."
            )
        if self.solution_sets:
            lines.append("  Univariate solution sets (necessary conditions):")
            entries = []
            for result in self.solution_sets:
                printer = self.problem._printer(result.values, scope=result.field.path[:-1])
                path = _path_label(result.field.path[len(printer.scope) :])
                entries.append(
                    (
                        result.field.path[:-1],
                        f"{path} in {printer.doprint(result.values)} "
                        f"[domain: {result.domain}; " + ", ".join(f"c{c.index}" for c in result.constraints) + "]",
                    )
                )
            lines.extend(self.problem._tree_lines(tuple(entries), 4))
            for result in self.solution_sets:
                lines.extend(f"    Solver: {error}" for error in result.solver_errors)
        if self.conflict is not None:
            minimal = "irreducible" if self.conflict.irreducible is True else "irreducibility not established"
            lines.append(f"  Proven conflict ({minimal}; declared domains retained):")
            lines.extend(self.problem._constraint_lines(self.conflict.constraints, 4))
            certificate = self.conflict.linear_certificate
            if certificate is not None:
                terms = " + ".join(f"({weight})*c{c.index}" for c, weight in certificate.terms)
                lines.append(f"    Certificate: {terms} gives {certificate.constant} = 0.")
        if self.input_checks:
            lines.append("  Input removals:")
            for check in self.input_checks:
                lines.append(f"    {check.input}: {check.status}")
                if check.status == "unknown":
                    lines.append(f"      {check.reason}")
        if self.candidates:
            lines.append("  Solver candidate audits: " + ", ".join(check.verdict for check in self.candidates) + ".")
        suffix = "; check budget exhausted" if self.budget_exhausted else ""
        lines.append(f"  Subset assessments: {self.checks}{suffix}.")
        return "\n".join(lines)


def _candidate(problem: ModelProblem, values: dict[Symbol, Expr]) -> Candidate:
    return Candidate(
        tuple(FieldValue(field, values.get(field.symbol, field.symbol)) for field in _canonical_fields(problem))
    )


def _audit(problem: ModelProblem, constraints: tuple[ModelConstraint, ...], candidate: Candidate) -> CandidateCheck:
    bindings: dict[Symbol, Expr] = {}
    for item in candidate.values:
        problem._require_field(item.field)
        if item.field.symbol in bindings and bindings[item.field.symbol] != item.value:
            raise ModelValidationError("Candidate contains conflicting values for aliases")
        bindings[item.field.symbol] = item.value
    residuals = []
    for c, (lhs, rhs) in zip(constraints, _algebra(constraints).equations, strict=True):
        left, right = lhs.xreplace(bindings), _substitute(rhs, bindings)
        verdict: Verdict
        if not isinstance(left, Expr) or not isinstance(right, Expr):
            residual, verdict = right, "unknown"
        elif left.has(S.NaN, S.ComplexInfinity) or right.has(S.NaN, S.ComplexInfinity):
            residual, verdict = S.NaN, "violated"
        else:
            residual = simplify(_evaluate(left - right))
            if (
                c.right.has(Float)
                or left.has(Float)
                or right.has(Float)
                or any(symbol.is_commutative is not True for symbol in c.symbols)
            ):
                verdict = "unknown"
            else:
                verdict = (
                    "satisfied" if residual.is_zero is True else "violated" if residual.is_zero is False else "unknown"
                )
        residuals.append(ResidualCheck(c, residual, verdict))
    assumptions = tuple(
        AssumptionCheck(item, predicate, expected, getattr(item.value, f"is_{predicate}", None))
        for item in candidate.values
        for predicate, expected in item.field.symbol.assumptions0.items()
    ) + tuple(
        AssumptionCheck(
            item,
            "defined",
            True,
            False
            if item.value.has(S.NaN, S.ComplexInfinity)
            else True
            if item.value.is_number is True and item.value.is_finite is True
            else None,
        )
        for item in candidate.values
    )
    closed = all(
        field.symbol in bindings
        and not bindings[field.symbol].free_symbols
        and bindings[field.symbol].is_number is True
        and bindings[field.symbol].is_finite is True
        for field in _canonical_fields(problem)
    )
    violated = any(check.verdict == "violated" for check in residuals) or any(
        check.actual is not None and check.actual != check.expected for check in assumptions
    )
    satisfied = (
        closed
        and all(check.verdict == "satisfied" for check in residuals)
        and all(check.actual == check.expected for check in assumptions)
    )
    verdict = "violated" if violated else "satisfied" if satisfied else "unknown"
    return CandidateCheck(candidate, tuple(residuals), assumptions, verdict)


def _violation(problem: ModelProblem, check: CandidateCheck) -> str:
    for assumption in check.assumptions:
        if assumption.actual is not None and assumption.actual != assumption.expected:
            return (
                f"{problem.format(assumption.value.field.symbol)} is forced to "
                f"{problem.format(assumption.value.value)}, violating {assumption.predicate}={assumption.expected}."
            )
    for residual in check.residuals:
        if residual.verdict == "violated":
            return f"c{residual.constraint.index} has impossible residual {problem.format(residual.residual)}."
    return "No closed witness or exact contradiction was established."


def _fixed_values(constraints: tuple[ModelConstraint, ...]) -> dict[Symbol, Expr]:
    fixed: dict[Symbol, Expr] = {}
    equations = _algebra(constraints).equations
    while True:
        additions = {}
        for left, right in equations:
            if left in fixed or not isinstance(right, Expr):
                continue
            value = _substitute(right, fixed)
            if not value.free_symbols and (isinstance(right, Float) or not value.has(Float)):
                additions[left] = _evaluate(value)
        if not additions:
            return fixed
        fixed.update(additions)


def _linear_part(
    problem: ModelProblem, constraints: tuple[ModelConstraint, ...]
) -> tuple[LinearAnalysis | None, dict[Symbol, Expr]]:
    fields = tuple(field for field in _canonical_fields(problem) if field.symbol.is_commutative is True)
    symbols = tuple(field.symbol for field in fields)
    defined = {c.left for c in constraints if c.kind is ConstraintKind.DEFINITION}
    rows, constants, included = [], [], []
    for c in constraints:
        if not isinstance(c.right, Expr) or not c.symbols <= set(symbols):
            continue
        # A certificate must combine original residuals with constant weights.
        # Expanding another definition under calculus needs a different proof.
        if c.right.has(Derivative, Integral, Limit) and c.right.free_symbols & defined:
            continue
        try:
            matrix, constant = linear_eq_to_matrix((_evaluate(c.left - c.right),), symbols)
        except NonlinearError:
            continue
        if not all(value.is_Rational for value in (*matrix, *constant)):
            continue
        rows.append(list(matrix))
        constants.append(constant[0])
        included.append(c)
    if not included:
        return None, {}
    matrix, constant = Matrix(rows), Matrix(constants)
    reduced, pivots = matrix.row_join(constant).rref()
    rank = sum(pivot < len(symbols) for pivot in pivots)
    augmented_rank = len(pivots)
    certificate = None
    expressions: dict[Symbol, Expr] = {}
    if augmented_rank > rank:
        for weights in matrix.T.nullspace():
            value = -(weights.T * constant)[0]
            if value != 0:
                certificate = LinearCertificate(
                    tuple((c, weight) for c, weight in zip(included, weights, strict=True) if weight != 0), value
                )
                break
    else:
        free = tuple(i for i in range(len(symbols)) if i not in pivots)
        for row, pivot in enumerate(pivots):
            expressions[symbols[pivot]] = reduced[row, -1] - sum(reduced[row, i] * symbols[i] for i in free)
    return (
        LinearAnalysis(fields, tuple(included), rank, augmented_rank, len(included) == len(constraints), certificate),
        expressions,
    )


@dataclass(frozen=True)
class _Assessment:
    status: Status
    reason: str
    deductions: CandidateCheck
    linear: LinearAnalysis | None = None
    witness: CandidateCheck | None = None
    candidates: tuple[CandidateCheck, ...] = ()
    solution_sets: tuple[SolutionSet, ...] = ()


def _solution_sets(
    problem: ModelProblem, constraints: tuple[ModelConstraint, ...], expressions: dict[Symbol, Expr]
) -> tuple[SolutionSet, ...]:
    # Use only forced, finite, exact numerical substitutions. External parameters
    # and simultaneous multivariate systems are left to the witness generator.
    substitutions = tuple(
        FieldValue(field, expressions[field.symbol])
        for field in _canonical_fields(problem)
        if field.symbol in expressions
        and expressions[field.symbol].is_number is True
        and expressions[field.symbol].is_finite is True
        and not expressions[field.symbol].has(Float)
    )
    bindings = {item.field.symbol: item.value for item in substitutions}
    fields = {field.symbol: field for field in _canonical_fields(problem)}
    groups: dict[Symbol, list[tuple[ModelConstraint, Expr]]] = {}
    for c, (left, right) in zip(constraints, _algebra(constraints).equations, strict=True):
        if (
            not isinstance(c.right, Expr)
            or c.right.has(Float)
            or any(symbol.is_commutative is not True for symbol in c.symbols)
        ):
            continue
        residual = _evaluate(left.xreplace(bindings) - _substitute(right, bindings))
        symbols = residual.free_symbols
        if len(symbols) == 1:
            symbol = next(iter(symbols))
            if symbol in fields:
                groups.setdefault(symbol, []).append((c, residual))
    results = []
    for symbol, equations in groups.items():
        domain = _domain(symbol)
        values = domain
        errors = []
        for c, residual in equations:
            try:
                # An unevaluated function can hide an external coefficient even
                # when free_symbols contains only the model field. A generic
                # solve may silently discard the coefficient's zero case.
                values &= (
                    ConditionSet(symbol, Eq(residual, 0), domain)
                    if residual.has(AppliedUndef)
                    else solveset(residual, symbol, domain)
                )
            except Exception as error:
                values &= ConditionSet(symbol, Eq(residual, 0), domain)
                errors.append(f"c{c.index}: {type(error).__name__}: {error}")
        results.append(
            SolutionSet(fields[symbol], domain, values, tuple(c for c, _ in equations), substitutions, tuple(errors))
        )
    return tuple(results)


_SAMPLES = (S.Zero, S.One, S.NegativeOne, S(2), S(-2), I, sqrt(2))


def _sample(symbol: Symbol, preferred: Expr) -> Expr:
    for value in (preferred, *_SAMPLES):
        if all(
            getattr(value, f"is_{predicate}", None) == expected for predicate, expected in symbol.assumptions0.items()
        ):
            return value
    return preferred


class _Analyser:
    """A cached, three-valued checker and a conservative deletion filter."""

    def __init__(self, problem: ModelProblem, max_checks: int) -> None:
        if max_checks < 1:
            raise ModelValidationError("max_checks must be at least 1")
        self.problem = problem
        self.max_checks = max_checks
        self.cache: dict[tuple[ModelConstraint, ...], _Assessment] = {}
        self.exhausted = False

    def assess(self, constraints: tuple[ModelConstraint, ...]) -> _Assessment:
        key = constraints
        if key in self.cache:
            return self.cache[key]
        if len(self.cache) >= self.max_checks:
            self.exhausted = True
            empty = CandidateCheck(_candidate(self.problem, {}), (), (), "unknown")
            return _Assessment("unknown", "Subset check budget exhausted.", empty)
        try:
            result = self._assess(constraints)
        except Exception as error:
            empty = CandidateCheck(_candidate(self.problem, {}), (), (), "unknown")
            result = _Assessment("unknown", f"Diagnostic check raised {type(error).__name__}: {error}", empty)
        self.cache[key] = result
        return result

    def _assess(self, constraints: tuple[ModelConstraint, ...]) -> _Assessment:
        fixed = _fixed_values(constraints)
        deductions = _audit(self.problem, constraints, _candidate(self.problem, fixed))
        linear, expressions = _linear_part(self.problem, constraints)
        if linear is not None and linear.certificate is not None:
            return _Assessment(
                "inconsistent", "An exact linear combination proves a contradiction.", deductions, linear
            )
        if deductions.verdict == "violated":
            return _Assessment("inconsistent", _violation(self.problem, deductions), deductions, linear)
        expressions.update(fixed)
        deductions = _audit(self.problem, constraints, _candidate(self.problem, expressions))
        if deductions.verdict == "violated":
            return _Assessment("inconsistent", _violation(self.problem, deductions), deductions, linear)
        solution_sets = _solution_sets(self.problem, constraints, expressions)
        if any(result.values is S.EmptySet for result in solution_sets):
            return _Assessment(
                "inconsistent",
                "A univariate subsystem has an empty solution set in a domain containing all permitted values.",
                deductions,
                linear,
                solution_sets=solution_sets,
            )
        # A failed sample never proves inconsistency. Only a fully checked
        # closed assignment may establish consistency, including all domains.
        witness = self._witness(constraints, expressions)
        if witness is not None:
            return _Assessment(
                "consistent",
                "A closed witness satisfies every equation and declared domain.",
                deductions,
                linear,
                witness,
                solution_sets=solution_sets,
            )
        generated = []
        witness = None
        for result in solution_sets:
            if isinstance(result.values, FiniteSet):
                for value in result.values:
                    proposal = {**expressions, result.field.symbol: value}
                    check = _audit(self.problem, constraints, _candidate(self.problem, proposal))
                    generated.append(check)
                    witness = witness or self._witness(constraints, proposal)
        if witness is not None:
            return _Assessment(
                "consistent",
                "A solution-set candidate supplies a checked closed witness.",
                deductions,
                linear,
                witness,
                tuple(generated),
                solution_sets,
            )
        # Solver answers are witness proposals. An empty answer is not a proof.
        detail = "No closed witness or exact contradiction was established."
        try:
            solutions = solve(
                [left - right for left, right in _algebra(constraints).equations],
                [field.symbol for field in _canonical_fields(self.problem)],
                dict=True,
                numerical=False,
            )
        except Exception as error:
            solutions = []
            detail += f" Diagnostic solver raised {type(error).__name__}: {error}"
        for solution in solutions:
            check = _audit(self.problem, constraints, _candidate(self.problem, solution))
            generated.append(check)
            witness = check if check.verdict == "satisfied" else self._witness(constraints, solution)
            if witness is not None:
                return _Assessment(
                    "consistent",
                    "A diagnostic solver candidate supplies a checked closed witness.",
                    deductions,
                    linear,
                    witness,
                    tuple(generated),
                    solution_sets,
                )
        return _Assessment(
            "unknown",
            detail + " Nonlinear, parameter, or Float checks may be undecided.",
            deductions,
            linear,
            candidates=tuple(generated),
            solution_sets=solution_sets,
        )

    def _witness(
        self, constraints: tuple[ModelConstraint, ...], expressions: dict[Symbol, Expr]
    ) -> CandidateCheck | None:
        for sample in _SAMPLES:
            choices = {field.symbol: _sample(field.symbol, sample) for field in _canonical_fields(self.problem)}
            choices.update({symbol: value for symbol, value in expressions.items() if not value.free_symbols})
            values = {
                field.symbol: _evaluate(_substitute(expressions.get(field.symbol, field.symbol), choices))
                for field in self.problem.fields
            }
            check = _audit(self.problem, constraints, _candidate(self.problem, values))
            if check.verdict == "satisfied":
                return check
        return None

    def run(self, conflicts: bool, relaxations: bool) -> ProblemAnalysis:
        problem = self.problem
        original = self.assess(problem.constraints)
        candidates = tuple(problem.check(candidate) for candidate in problem.candidates) + original.candidates
        witness = original.witness or next((check for check in candidates if check.verdict == "satisfied"), None)
        status: Status = "consistent" if witness is not None else original.status
        reason = (
            "A solver candidate is a checked closed witness."
            if original.witness is None and witness
            else original.reason
        )
        conflict = None
        if status == "inconsistent":
            certificate = original.linear.certificate if original.linear else None
            current = tuple(c for c, _ in certificate.terms) if certificate is not None else problem.constraints
            assessment = original
            if conflicts:
                # Removing only proven-unnecessary members preserves a proof.
                # Unknown checks retain the member and cannot certify minimality.
                for c in current:
                    trial = tuple(other for other in current if other != c)
                    checked = self.assess(trial)
                    if checked.status == "inconsistent":
                        current, assessment = trial, checked
            irreducible = None
            deletion_checks = []
            if conflicts:
                for c in current:
                    checked = self.assess(tuple(other for other in current if other != c))
                    deletion_checks.append(RemovalCheck((c,), checked.status, checked.reason, checked.witness))
                if all(check.status == "consistent" for check in deletion_checks):
                    irreducible = True
            certificate = assessment.linear.certificate if assessment.linear else None
            conflict = Conflict(current, irreducible, assessment.reason, certificate, tuple(deletion_checks))
        removals = []
        if relaxations and status == "inconsistent":
            for name in dict.fromkeys(c.input for c in problem.constraints if c.input is not None):
                removed = tuple(c for c in problem.constraints if c.input == name)
                checked = self.assess(tuple(c for c in problem.constraints if c.input != name))
                removals.append(InputCheck(removed, checked.status, checked.reason, checked.witness, name))
        return ProblemAnalysis(
            problem,
            status,
            reason,
            original.linear,
            original.deductions,
            candidates,
            witness,
            conflict,
            tuple(removals),
            len(self.cache),
            self.exhausted,
            original.solution_sets,
        )

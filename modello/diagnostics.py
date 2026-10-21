"""Opt-in, conservative diagnosis of a recorded symbolic problem.

Constraints and domains are kept separate. A failed solver is not an
inconsistency oracle: consistency requires a checked, closed witness and
inconsistency requires an exact contradiction. Everything else is unknown.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from sympy import (
    Basic,
    Expr,
    Float,
    I,
    Matrix,
    S,
    Symbol,
    SympifyError,
    linear_eq_to_matrix,
    simplify,
    solve,
    sqrt,
    sympify,
)
from sympy.printing.str import StrPrinter
from sympy.solvers.solveset import NonlinearError

if TYPE_CHECKING:
    from modello import Modello, ScalarValue

FieldPath = tuple[str, ...]
Status = Literal["consistent", "inconsistent", "unknown"]
Verdict = Literal["satisfied", "violated", "unknown"]


@dataclass(frozen=True)
class ModelField:
    """One public path to a quantity; aliases retain the same symbol identity."""

    path: FieldPath
    symbol: Symbol


@dataclass(frozen=True)
class ModelConstraint:
    """An equation's origin, original sides, and optional constructor-input group."""

    index: int
    path: FieldPath
    kind: Literal["definition", "observation"]
    left: Symbol
    right: Basic
    input: str | None = None

    @property
    def symbols(self) -> frozenset[Symbol]:
        """Return identities mentioned by either side, without simplifying them."""
        return frozenset(self.left.free_symbols | self.right.free_symbols)


@dataclass(frozen=True, init=False)
class FieldValue:
    """A value associated with an actual field of this problem."""

    field: ModelField
    value: Expr

    def __init__(self, field: ModelField, value: ScalarValue) -> None:
        try:
            expression = value if isinstance(value, Expr) else sympify(value)
        except (SympifyError, TypeError) as error:
            raise TypeError("A field value must be a SymPy-compatible scalar") from error
        if not isinstance(expression, Expr):
            raise TypeError("A field value must be a SymPy-compatible scalar")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "value", expression)


@dataclass(frozen=True)
class Candidate:
    """A proposed assignment, rather than a claim of validity or completeness."""

    values: tuple[FieldValue, ...]

    def value(self, field: ModelField) -> Expr:
        """Read a quantity through any of its recorded aliases."""
        matches = tuple(item.value for item in self.values if item.field.symbol == field.symbol)
        if not matches:
            raise KeyError(".".join(field.path))
        if any(value != matches[0] for value in matches):
            raise ValueError("Candidate contains conflicting values for aliases")
        return matches[0]


@dataclass(frozen=True)
class ResidualCheck:
    """An equation checked against a candidate, retaining an undecided outcome."""

    constraint: ModelConstraint
    residual: Basic
    verdict: Verdict


@dataclass(frozen=True)
class AssumptionCheck:
    """A field's required predicate compared with its proposed value."""

    value: FieldValue
    predicate: str
    expected: bool
    actual: bool | None


@dataclass(frozen=True)
class CandidateCheck:
    """An audit; only a satisfied, closed assignment is a consistency witness."""

    candidate: Candidate
    residuals: tuple[ResidualCheck, ...]
    assumptions: tuple[AssumptionCheck, ...]
    verdict: Verdict


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
        return simplify(residual - self.constant).is_zero is True and self.constant.is_zero is False


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


class _FieldPrinter(StrPrinter):
    """Label quantities without substitution, which could change the algebra."""

    def __init__(self, fields: tuple[ModelField, ...]) -> None:
        super().__init__()
        self.labels: dict[Symbol, str] = {}
        for field in fields:
            self.labels.setdefault(field.symbol, "self." + ".".join(field.path))

    def _print_Dummy(self, expression: Symbol) -> str:
        return self.labels.get(expression, super()._print_Dummy(expression))


@dataclass(frozen=True, repr=False)
class ModelProblem:
    """A frozen snapshot shared by preparation, construction, and failure handling."""

    model: type[Modello]
    name: str
    fields: tuple[ModelField, ...]
    constraints: tuple[ModelConstraint, ...]
    candidates: tuple[Candidate, ...] = ()

    def __repr__(self) -> str:
        return (
            f"ModelProblem({self.model.__name__}, {self.name!r}, "
            f"fields={len(self.fields)}, constraints={len(self.constraints)})"
        )

    def format(self, expression: Basic) -> str:
        """Render identities as field paths, retaining every original expression."""
        return str(_FieldPrinter(self.fields).doprint(expression))

    def __str__(self) -> str:
        equations = "\n".join(
            f"  c{c.index} [{c.kind} {'.'.join(c.path)}]: {self.format(c.left)} = {self.format(c.right)}"
            for c in self.constraints
        )
        return f"{self.model.__name__}({self.name!r})\n{equations}"

    def field(self, *path: str) -> ModelField:
        """Look up a declared path, without matching symbols by printed name."""
        for field in self.fields:
            if field.path == path:
                return field
        raise KeyError(".".join(path))

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
            raise ValueError("The field belongs to another problem")

    def at(self, *path: str) -> ComponentView:
        """Inspect a component subtree and its boundary; this does not isolate a solve."""
        model = self.model
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

    def without(self, *inputs: str) -> ModelProblem:
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


class ModelSolveError(ValueError):
    """A construction failure carrying the original problem and solver candidates."""

    def __init__(self, problem: ModelProblem, count: int | None, candidates: tuple[Candidate, ...] = ()) -> None:
        self.problem = replace(problem, candidates=candidates)
        self.solution_count = count
        prefix = "Solver failed" if count is None else f"{count} solutions reported"
        super().__init__(
            f"{prefix} for {problem.model.__name__}({problem.name!r})\n"
            "Inspect .problem or call .problem.analyse() for an opt-in diagnostic report."
        )


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

    @property
    def relaxations(self) -> tuple[InputCheck, ...]:
        """Return just the input removals with a checked closed witness."""
        return tuple(check for check in self.input_checks if check.witness is not None)

    def __str__(self) -> str:
        lines = [f"{self.problem.model.__name__}({self.problem.name!r}): {self.status}", self.reason]
        if self.linear is not None:
            scope = "all equations" if self.linear.complete else "affine subset only"
            lines.append(
                f"Exact equality rank {self.linear.rank}; augmented rank {self.linear.augmented_rank}; "
                f"nullity {self.linear.nullity} ({scope}, domains excluded)."
            )
        if self.conflict is not None:
            minimal = "irreducible" if self.conflict.irreducible is True else "irreducibility not established"
            lines.append(f"Proven conflict ({minimal}; declared domains retained):")
            for c in self.conflict.constraints:
                lines.append(
                    f"  c{c.index} [{c.kind} {'.'.join(c.path)}]: "
                    f"{self.problem.format(c.left)} = {self.problem.format(c.right)}"
                )
            certificate = self.conflict.linear_certificate
            if certificate is not None:
                terms = " + ".join(f"({weight})*c{c.index}" for c, weight in certificate.terms)
                lines.append(f"  Certificate: {terms} gives {certificate.constant} = 0.")
        for check in self.input_checks:
            lines.append(f"Removing input {check.input!r}: {check.status}.")
        if self.candidates:
            lines.append("Solver candidate audits: " + ", ".join(check.verdict for check in self.candidates) + ".")
        suffix = "; check budget exhausted" if self.budget_exhausted else ""
        lines.append(f"Subset assessments: {self.checks}{suffix}.")
        return "\n".join(lines)


def _canonical_fields(problem: ModelProblem) -> tuple[ModelField, ...]:
    fields: dict[Symbol, ModelField] = {}
    for field in problem.fields:
        fields.setdefault(field.symbol, field)
    return tuple(fields.values())


def _candidate(problem: ModelProblem, values: dict[Symbol, Expr]) -> Candidate:
    return Candidate(
        tuple(FieldValue(field, values.get(field.symbol, field.symbol)) for field in _canonical_fields(problem))
    )


def _audit(problem: ModelProblem, constraints: tuple[ModelConstraint, ...], candidate: Candidate) -> CandidateCheck:
    bindings: dict[Symbol, Expr] = {}
    for item in candidate.values:
        problem._require_field(item.field)
        if item.field.symbol in bindings and bindings[item.field.symbol] != item.value:
            raise ValueError("Candidate contains conflicting values for aliases")
        bindings[item.field.symbol] = item.value
    residuals = []
    for c in constraints:
        left, right = c.left.xreplace(bindings), c.right.xreplace(bindings)
        verdict: Verdict
        if not isinstance(left, Expr) or not isinstance(right, Expr):
            residual, verdict = right, "unknown"
        elif left.has(S.NaN, S.ComplexInfinity) or right.has(S.NaN, S.ComplexInfinity):
            residual, verdict = S.NaN, "violated"
        else:
            residual = simplify(left - right)
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
    while True:
        additions = {
            c.left: c.right.xreplace(fixed)
            for c in constraints
            if c.left not in fixed
            and isinstance(c.right, Expr)
            and not c.right.xreplace(fixed).free_symbols
            and (isinstance(c.right, Float) or not c.right.xreplace(fixed).has(Float))
        }
        if not additions:
            return fixed
        fixed.update(additions)


def _linear_part(
    problem: ModelProblem, constraints: tuple[ModelConstraint, ...]
) -> tuple[LinearAnalysis | None, dict[Symbol, Expr]]:
    fields = tuple(field for field in _canonical_fields(problem) if field.symbol.is_commutative is True)
    symbols = tuple(field.symbol for field in fields)
    rows, constants, included = [], [], []
    for c in constraints:
        if not isinstance(c.right, Expr) or not c.symbols <= set(symbols):
            continue
        try:
            matrix, constant = linear_eq_to_matrix((c.left - c.right,), symbols)
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
            raise ValueError("max_checks must be at least 1")
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
            )
        # Solver answers are witness proposals. An empty answer is not a proof.
        generated = []
        detail = "No closed witness or exact contradiction was established."
        try:
            solutions = solve(
                [c.left - c.right for c in constraints],
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
                )
        return _Assessment(
            "unknown",
            detail + " Nonlinear, parameter, or Float checks may be undecided.",
            deductions,
            linear,
            candidates=tuple(generated),
        )

    def _witness(
        self, constraints: tuple[ModelConstraint, ...], expressions: dict[Symbol, Expr]
    ) -> CandidateCheck | None:
        for sample in _SAMPLES:
            choices = {field.symbol: _sample(field.symbol, sample) for field in _canonical_fields(self.problem)}
            values = {
                field.symbol: expressions.get(field.symbol, field.symbol).xreplace(choices)
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
        )

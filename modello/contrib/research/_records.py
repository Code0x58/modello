"""Immutable claims, contracts, and evidence for mathematical investigations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from sympy import Basic, Expr, S, Set, Symbol, sympify

from modello import Annotation
from modello.errors import ModelDefinitionError, ModelValidationError

if TYPE_CHECKING:
    from ._case import ResearchCase

Verdict: TypeAlias = Literal["strictly true", "conditionally true", "strictly false", "unresolved"]
Subject: TypeAlias = str | Basic
Formula: TypeAlias = Expr | Callable[[Callable[[str], Expr]], Expr]
Property: TypeAlias = Literal[
    "positive",
    "nonnegative",
    "nonzero",
    "increasing",
    "decreasing",
    "convex",
    "concave",
    "smooth",
    "differentiable",
    "limit",
    "uniform_bound",
    "curvature_bound",
    "exact_form",
    "predicate",
    "theorem",
]


@dataclass(frozen=True)
class Claim:
    """A property of a subject, quantified over an explicit coordinate domain.

    Strings select relative model paths; native expressions retain their symbol
    identities. Other free symbols are parameters, not quantified coordinates.
    ``at`` requests differentiability in a neighborhood of a particular point.
    """

    name: str
    subject: Subject
    property: Property
    wrt: str | Symbol | None = None
    domain: Set = S.Reals
    order: int = 1
    bound: Expr | None = None
    expected: Formula | None = None
    at: Expr | None = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.domain, Set):
            raise ModelDefinitionError("Claims need a name and an explicit SymPy set domain")
        if type(self.order) is not int or self.order < 1:
            raise ModelDefinitionError("Derivative orders must be positive")
        if not isinstance(self.subject, (str, Basic)):
            object.__setattr__(self, "subject", sympify(self.subject))
        if self.expected is not None and not callable(self.expected) and not isinstance(self.expected, Basic):
            object.__setattr__(self, "expected", sympify(self.expected))
        if self.property in {"uniform_bound", "curvature_bound"} and self.bound is None:
            raise ModelDefinitionError("Uniform bounds need a target bound")
        if (
            self.property
            in {"smooth", "differentiable", "increasing", "decreasing", "convex", "concave", "limit", "curvature_bound"}
            and self.wrt is None
        ):
            raise ModelDefinitionError(f"{self.property} needs a coordinate")


def positive(subject: Subject, *, wrt=None, domain=S.Reals, name="positive") -> Claim:
    return Claim(name, subject, "positive", wrt, domain)


def nonnegative(subject: Subject, *, wrt=None, domain=S.Reals, name="nonnegative") -> Claim:
    return Claim(name, subject, "nonnegative", wrt, domain)


def nonzero(subject: Subject, *, wrt=None, domain=S.Reals, name="nonzero") -> Claim:
    return Claim(name, subject, "nonzero", wrt, domain)


def increasing(subject: Subject, *, wrt, domain=S.Reals, name="increasing") -> Claim:
    return Claim(name, subject, "increasing", wrt, domain)


def decreasing(subject: Subject, *, wrt, domain=S.Reals, name="decreasing") -> Claim:
    return Claim(name, subject, "decreasing", wrt, domain)


def convex(subject: Subject, *, wrt, domain=S.Reals, name="convex") -> Claim:
    return Claim(name, subject, "convex", wrt, domain)


def concave(subject: Subject, *, wrt, domain=S.Reals, name="concave") -> Claim:
    return Claim(name, subject, "concave", wrt, domain)


def smooth(subject: Subject, *, wrt, order=2, domain=S.Reals, name="smooth") -> Claim:
    return Claim(name, subject, "smooth", wrt, domain, order=order)


def differentiable(subject: Subject, *, wrt, order=1, domain=S.Reals, at=None, name="differentiable") -> Claim:
    return Claim(name, subject, "differentiable", wrt, domain, order=order, at=None if at is None else sympify(at))


def vanishes(subject: Subject, *, wrt, name="right_tail") -> Claim:
    return Claim(name, subject, "limit", wrt, expected=S.Zero)


def uniform_bound(subject: Subject, bound, *, wrt, domain=S.Reals, name="uniform_bound") -> Claim:
    return Claim(name, subject, "uniform_bound", wrt, domain, bound=sympify(bound))


def curvature_bound(subject: Subject, *, wrt, bound=2, domain=S.Reals, name="curvature_bound") -> Claim:
    return Claim(name, subject, "curvature_bound", wrt, domain, bound=sympify(bound))


def exact_form(subject: Subject, expected: Formula, *, wrt=None, domain=S.Reals, name="exact_form") -> Claim:
    return Claim(name, subject, "exact_form", wrt, domain, expected=expected)


def predicate(condition: Subject, *, name="predicate") -> Claim:
    return Claim(name, condition, "predicate")


@dataclass(frozen=True)
class Requirement:
    """A claim together with the consumer that needs it and its motivation."""

    claim: Claim
    consumer: str
    reason: str
    source: str = ""


@dataclass(frozen=True, init=False)
class Requires(Annotation):
    """Attach relative requirements with ``annotate(expression, Requires(...))``."""

    requirements: tuple[Requirement, ...]

    def __init__(self, *requirements: Requirement) -> None:
        if any(not isinstance(item, Requirement) for item in requirements):
            raise ModelDefinitionError("Requires accepts Requirement records")
        object.__setattr__(self, "requirements", requirements)


@dataclass(frozen=True)
class ProofHint:
    """A proposed uniform upper bound; the checker must establish it."""

    upper: Expr


def upper_bound(value) -> ProofHint:
    return ProofHint(sympify(value))


@dataclass(frozen=True)
class Evidence:
    """One result, including guards, derivation, and validated counterexamples."""

    id: str
    requirement: Requirement
    expression: Basic
    coordinate: Symbol | None
    condition: Basic | None
    steps: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    witness: tuple[tuple[Symbol, Expr], ...] = ()
    kind: str = "symbolic"
    condition_strength: str = "sufficient"
    premises: tuple[Evidence, ...] = ()
    resolved_claim: Claim | None = None
    hint: Expr | None = None

    @property
    def verdict(self) -> Verdict:
        if self.condition is S.true:
            return "strictly true"
        if self.condition is S.false:
            return "strictly false"
        return "unresolved" if self.condition is None else "conditionally true"


@dataclass(frozen=True)
class ResearchReport:
    """Evidence for one immutable case; consistency is checked separately."""

    model: str
    checks: tuple[Evidence, ...]
    assumptions: tuple[Basic, ...] = ()
    consistent: bool | None = True
    issues: tuple[str, ...] = ()
    labels: tuple[tuple[Symbol, str], ...] = ()

    def explain(self, claim_id: str) -> Evidence:
        def search(items):
            for item in items:
                yield item
                yield from search(item.premises)

        matches = [item for item in search(self.checks) if item.id == claim_id]
        if len(matches) != 1:
            raise KeyError(claim_id)
        return matches[0]

    def __str__(self) -> str:
        lines = [f"Research: {self.model}; assumptions consistent: {self.consistent}"]
        for item in self.checks:
            condition = f" if {item.condition}" if item.verdict == "conditionally true" else ""
            lines.append(f"  {item.id}: {item.verdict}{condition} — {item.requirement.reason}")
        return "\n".join([*lines, *(f"  Issue: {issue}" for issue in self.issues)])

    def to_dict(self) -> dict[str, Any]:
        """Export descriptive evidence; this is not an executable case format."""
        from ._report import report_dict

        return report_dict(self)

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def to_html(self, *, title: str = "Mathematical research report") -> str:
        from ._report import report_html

        return report_html(self, title)


@dataclass(frozen=True)
class Theorem:
    """A trusted, versioned rule whose premises are recursively checked.

    Registered theorem code is part of the trust boundary, like the CAS. A
    bibliographic note by itself never registers or proves a theorem.
    """

    name: str
    conclusion: Claim
    premises: tuple[Requirement, ...]
    source: str
    version: str = "1"


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class Method(Generic[ResultT]):
    """A computation with explicit requirements and the model fields it reads."""

    name: str
    execute: Callable[..., ResultT]
    requirements: tuple[Requirement, ...] = ()
    uses: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchResult(Generic[ResultT]):
    method: str
    data: ResultT
    report: ResearchReport
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class Experiment:
    """Finite observations, never a proof of a universal claim."""

    expression: Expr
    coordinate: Symbol
    points: tuple[Expr, ...]
    values: tuple[Expr, ...]
    kind: str = "numerical observation"


class ResearchError(ModelValidationError):
    """A blocked computation retaining the case and its current evidence."""

    def __init__(self, message: str, case: ResearchCase, report: ResearchReport) -> None:
        self.case, self.report = case, report
        super().__init__(message)

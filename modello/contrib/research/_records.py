"""Immutable claims, contracts, and evidence for mathematical investigations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from sympy import Basic, Expr, S, Set, Symbol, sympify
from sympy.assumptions import global_assumptions

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
class Requirement:
    """A mathematical condition and optional motivation, on an explicit domain.

    Strings select relative model paths; native expressions retain their symbol
    identities. ``wrt`` selects a known independent symbol, rather than a
    defined quantity. Other free symbols are parameters, not quantified coordinates.
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
    consumer: str = field(default="", kw_only=True)
    reason: str = field(default="", kw_only=True)
    source: str = field(default="", kw_only=True)

    def _proposition(self) -> tuple:
        """Match mathematical statements independently of their motivation."""
        return (
            self.name,
            self.subject,
            self.property,
            self.wrt,
            self.domain,
            self.order,
            self.bound,
            self.expected,
            self.at,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not isinstance(self.domain, Set):
            raise ModelDefinitionError("Requirements need a name and an explicit SymPy set domain")
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


def positive(
    subject: Subject,
    *,
    wrt=None,
    domain=S.Reals,
    name="positive",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(name, subject, "positive", wrt, domain, consumer=consumer, reason=reason, source=source)


def nonnegative(
    subject: Subject,
    *,
    wrt=None,
    domain=S.Reals,
    name="nonnegative",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(name, subject, "nonnegative", wrt, domain, consumer=consumer, reason=reason, source=source)


def nonzero(
    subject: Subject,
    *,
    wrt=None,
    domain=S.Reals,
    name="nonzero",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(name, subject, "nonzero", wrt, domain, consumer=consumer, reason=reason, source=source)


def increasing(
    subject: Subject, *, wrt, domain=S.Reals, name="increasing", consumer: str = "", reason: str = "", source: str = ""
) -> Requirement:
    return Requirement(name, subject, "increasing", wrt, domain, consumer=consumer, reason=reason, source=source)


def decreasing(
    subject: Subject, *, wrt, domain=S.Reals, name="decreasing", consumer: str = "", reason: str = "", source: str = ""
) -> Requirement:
    return Requirement(name, subject, "decreasing", wrt, domain, consumer=consumer, reason=reason, source=source)


def convex(
    subject: Subject, *, wrt, domain=S.Reals, name="convex", consumer: str = "", reason: str = "", source: str = ""
) -> Requirement:
    return Requirement(name, subject, "convex", wrt, domain, consumer=consumer, reason=reason, source=source)


def concave(
    subject: Subject, *, wrt, domain=S.Reals, name="concave", consumer: str = "", reason: str = "", source: str = ""
) -> Requirement:
    return Requirement(name, subject, "concave", wrt, domain, consumer=consumer, reason=reason, source=source)


def smooth(
    subject: Subject,
    *,
    wrt,
    order=2,
    domain=S.Reals,
    name="smooth",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(
        name, subject, "smooth", wrt, domain, order=order, consumer=consumer, reason=reason, source=source
    )


def differentiable(
    subject: Subject,
    *,
    wrt,
    order=1,
    domain=S.Reals,
    at=None,
    name="differentiable",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(
        name,
        subject,
        "differentiable",
        wrt,
        domain,
        order=order,
        at=None if at is None else sympify(at),
        consumer=consumer,
        reason=reason,
        source=source,
    )


def vanishes(
    subject: Subject, *, wrt, name="right_tail", consumer: str = "", reason: str = "", source: str = ""
) -> Requirement:
    return Requirement(name, subject, "limit", wrt, expected=S.Zero, consumer=consumer, reason=reason, source=source)


def uniform_bound(
    subject: Subject,
    bound,
    *,
    wrt,
    domain=S.Reals,
    name="uniform_bound",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(
        name,
        subject,
        "uniform_bound",
        wrt,
        domain,
        bound=sympify(bound),
        consumer=consumer,
        reason=reason,
        source=source,
    )


def curvature_bound(
    subject: Subject,
    *,
    wrt,
    bound=2,
    domain=S.Reals,
    name="curvature_bound",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(
        name,
        subject,
        "curvature_bound",
        wrt,
        domain,
        bound=sympify(bound),
        consumer=consumer,
        reason=reason,
        source=source,
    )


def exact_form(
    subject: Subject,
    expected: Formula,
    *,
    wrt=None,
    domain=S.Reals,
    name="exact_form",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Requirement:
    return Requirement(
        name, subject, "exact_form", wrt, domain, expected=expected, consumer=consumer, reason=reason, source=source
    )


def predicate(
    condition: Subject, *, name="predicate", consumer: str = "", reason: str = "", source: str = ""
) -> Requirement:
    return Requirement(name, condition, "predicate", consumer=consumer, reason=reason, source=source)


@dataclass(frozen=True, init=False)
class _Requirements(Annotation):
    """One annotation registry entry for a declaration's requirements."""

    requirements: tuple[Requirement, ...]
    model_level: bool

    def __init__(self, *requirements: Requirement, model_level: bool = False) -> None:
        if any(not isinstance(item, Requirement) for item in requirements):
            raise ModelDefinitionError("requires accepts property requirements")
        object.__setattr__(self, "requirements", requirements)
        object.__setattr__(self, "model_level", model_level)


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
    resolved: Requirement | None = None
    hint: Expr | None = None
    expected_template: Basic | None = None
    source_operands: tuple[Basic, ...] = ()

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

    def __post_init__(self) -> None:
        pending = list(self.checks)
        ids: set[str] = set()
        while pending:
            item = pending.pop()
            if item.id in ids:
                raise ModelDefinitionError(f"Evidence IDs must be unique within a research report: {item.id}")
            ids.add(item.id)
            pending.extend(item.premises)

    def explain(self, requirement_id: str) -> Evidence:
        def search(items):
            for item in items:
                yield item
                yield from search(item.premises)

        matches = [item for item in search(self.checks) if item.id == requirement_id]
        if len(matches) != 1:
            raise KeyError(requirement_id)
        return matches[0]

    def __str__(self) -> str:
        lines = [f"Research: {self.model}; assumptions consistent: {self.consistent}"]
        for item in self.checks:
            condition = f" if {item.condition}" if item.verdict == "conditionally true" else ""
            reason = f" — {item.requirement.reason}" if item.requirement.reason else ""
            lines.append(f"  {item.id}: {item.verdict}{condition}{reason}")
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
    conclusion: Requirement
    premises: tuple[Requirement, ...]
    source: str
    version: str = "1"

    def __post_init__(self) -> None:
        premises: dict[str, Requirement] = {}
        for requirement in self.premises:
            if not isinstance(requirement, Requirement):
                raise ModelDefinitionError("Theorem premises must be requirements")
            if requirement.name in premises and premises[requirement.name] != requirement:
                raise ModelDefinitionError(f"Theorem premise names must be unique: {requirement.name}")
            premises[requirement.name] = requirement
        object.__setattr__(self, "premises", tuple(premises.values()))


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class Method(Generic[ResultT]):
    """A computation with explicit requirements and the model fields it reads."""

    name: str
    execute: Callable[..., ResultT]
    requirements: tuple[Requirement, ...] = ()
    uses: tuple[str, ...] = ()
    bound_case: ResearchCase | None = field(default=None, repr=False, compare=False)
    bound_assumptions: frozenset[Basic] | None = field(default=None, repr=False, compare=False)

    def bind(self, case: ResearchCase) -> Method[ResultT]:
        """Bind case-specific expressions to their full immutable context.

        Rebuild a bound method after branching. Rebinding stale requirements
        would not recompute the expressions that captured the earlier case.
        """
        context = frozenset(global_assumptions)
        if self.bound_case is not None and (self.bound_case is not case or self.bound_assumptions != context):
            raise ModelDefinitionError("Rebuild a case-bound method after changing its case or SymPy assumptions")
        return self if self.bound_case is case else replace(self, bound_case=case, bound_assumptions=context)


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

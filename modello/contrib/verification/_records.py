"""Immutable claims, contracts, and evidence for mathematical investigations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeAlias, TypeVar

from sympy import Basic, Expr, S, Set, Symbol, sympify
from sympy.assumptions import global_assumptions

from modello import Annotation
from modello._reconstruction import _expression_key
from modello.errors import ModelDefinitionError, ModelValidationError

if TYPE_CHECKING:
    from ._context import VerificationContext

Verdict: TypeAlias = Literal["strictly true", "conditionally true", "strictly false", "unresolved"]
Subject: TypeAlias = str | Basic
Formula: TypeAlias = Expr | Callable[[Callable[[str], Expr]], Expr]
Property: TypeAlias = Literal[
    "positive",
    "nonnegative",
    "nonzero",
    "strictly_increasing",
    "strictly_decreasing",
    "convex",
    "concave",
    "continuously_differentiable",
    "differentiable",
    "limit",
    "supremum_below",
    "derivative_ratio_bound",
    "equal_to",
    "predicate",
    "rule",
]


@dataclass(frozen=True)
class Obligation:
    """A named proposition to establish, with an explicit domain and motivation.

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

    def _expression_fields(self) -> dict[str, Subject | Formula | None]:
        """The claim's symbolic payload, including paths and formula callbacks."""
        return {
            "subject": self.subject,
            "wrt": self.wrt,
            "domain": self.domain,
            "bound": self.bound,
            "expected": self.expected,
            "at": self.at,
        }

    def _proposition(self) -> tuple:
        """Match mathematical statements independently of their motivation."""
        return _expression_key((self.name, self.property, self.order, tuple(self._expression_fields().values())))

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or not isinstance(self.domain, Set):
            raise ModelDefinitionError("Obligations need a name and an explicit SymPy set domain")
        if type(self.order) is not int or self.order < 1:
            raise ModelDefinitionError("Derivative orders must be positive")
        if not isinstance(self.subject, (str, Basic)):
            object.__setattr__(self, "subject", sympify(self.subject))
        if self.expected is not None and not callable(self.expected) and not isinstance(self.expected, Basic):
            object.__setattr__(self, "expected", sympify(self.expected))
        if self.property in {"supremum_below", "derivative_ratio_bound"} and self.bound is None:
            raise ModelDefinitionError("Uniform bounds need a target bound")
        if (
            self.property
            in {
                "continuously_differentiable",
                "differentiable",
                "strictly_increasing",
                "strictly_decreasing",
                "convex",
                "concave",
                "limit",
                "derivative_ratio_bound",
            }
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
) -> Obligation:
    return Obligation(name, subject, "positive", wrt, domain, consumer=consumer, reason=reason, source=source)


def nonnegative(
    subject: Subject,
    *,
    wrt=None,
    domain=S.Reals,
    name="nonnegative",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    return Obligation(name, subject, "nonnegative", wrt, domain, consumer=consumer, reason=reason, source=source)


def nonzero(
    subject: Subject,
    *,
    wrt=None,
    domain=S.Reals,
    name="nonzero",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    return Obligation(name, subject, "nonzero", wrt, domain, consumer=consumer, reason=reason, source=source)


def strictly_increasing(
    subject: Subject,
    *,
    wrt,
    domain=S.Reals,
    name="strictly_increasing",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    """State f(x) < f(y) for every ordered pair x < y in the domain."""
    return Obligation(
        name, subject, "strictly_increasing", wrt, domain, consumer=consumer, reason=reason, source=source
    )


def strictly_decreasing(
    subject: Subject,
    *,
    wrt,
    domain=S.Reals,
    name="strictly_decreasing",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    """State f(x) > f(y) for every ordered pair x < y in the domain."""
    return Obligation(
        name, subject, "strictly_decreasing", wrt, domain, consumer=consumer, reason=reason, source=source
    )


def convex(
    subject: Subject, *, wrt, domain=S.Reals, name="convex", consumer: str = "", reason: str = "", source: str = ""
) -> Obligation:
    return Obligation(name, subject, "convex", wrt, domain, consumer=consumer, reason=reason, source=source)


def concave(
    subject: Subject, *, wrt, domain=S.Reals, name="concave", consumer: str = "", reason: str = "", source: str = ""
) -> Obligation:
    return Obligation(name, subject, "concave", wrt, domain, consumer=consumer, reason=reason, source=source)


def continuously_differentiable(
    subject: Subject,
    *,
    wrt,
    order=2,
    domain=S.Reals,
    name="continuously_differentiable",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    """Require C^n regularity through the finite derivative order n."""
    return Obligation(
        name,
        subject,
        "continuously_differentiable",
        wrt,
        domain,
        order=order,
        consumer=consumer,
        reason=reason,
        source=source,
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
) -> Obligation:
    return Obligation(
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


def tends_to_zero(
    subject: Subject, *, wrt, name="right_tail", consumer: str = "", reason: str = "", source: str = ""
) -> Obligation:
    """State a zero limit as the coordinate tends to positive infinity."""
    return Obligation(name, subject, "limit", wrt, expected=S.Zero, consumer=consumer, reason=reason, source=source)


def supremum_below(
    subject: Subject,
    bound,
    *,
    wrt,
    domain=S.Reals,
    name="supremum_below",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    """State sup(f) < bound on the domain, with a strict uniform margin."""
    return Obligation(
        name,
        subject,
        "supremum_below",
        wrt,
        domain,
        bound=sympify(bound),
        consumer=consumer,
        reason=reason,
        source=source,
    )


def derivative_ratio_bound(
    subject: Subject,
    *,
    wrt,
    bound=2,
    domain=S.Reals,
    name="derivative_ratio_bound",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    """Bound sup(f*f''/(f')**2) strictly, requiring f'' to exist and f' != 0."""
    return Obligation(
        name,
        subject,
        "derivative_ratio_bound",
        wrt,
        domain,
        bound=sympify(bound),
        consumer=consumer,
        reason=reason,
        source=source,
    )


def equal_to(
    subject: Subject,
    expected: Formula,
    *,
    wrt=None,
    domain=S.Reals,
    name="equal_to",
    consumer: str = "",
    reason: str = "",
    source: str = "",
) -> Obligation:
    """Require equality on the domain, allowing algebraically equivalent forms."""
    return Obligation(
        name, subject, "equal_to", wrt, domain, expected=expected, consumer=consumer, reason=reason, source=source
    )


def predicate(
    condition: Subject, *, name="predicate", consumer: str = "", reason: str = "", source: str = ""
) -> Obligation:
    return Obligation(name, condition, "predicate", consumer=consumer, reason=reason, source=source)


@dataclass(frozen=True, init=False)
class _Obligations(Annotation):
    """One annotation registry entry for a declaration's obligations."""

    obligations: tuple[Obligation, ...]
    model_level: bool

    def __init__(self, *obligations: Obligation, model_level: bool = False) -> None:
        if any(not isinstance(item, Obligation) for item in obligations):
            raise ModelDefinitionError("specifies accepts Obligation records")
        object.__setattr__(self, "obligations", obligations)
        object.__setattr__(self, "model_level", model_level)


@dataclass(frozen=True)
class Evidence:
    """One result, including guards, derivation, and validated counterexamples."""

    id: str
    obligation: Obligation
    expression: Basic
    coordinate: Symbol | None
    condition: Basic | None
    steps: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    witness: tuple[tuple[Symbol, Expr], ...] = ()
    kind: str = "symbolic"
    condition_strength: str = "sufficient"
    premises: tuple[Evidence, ...] = ()
    resolved: Obligation | None = None
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
class VerificationReport:
    """Evidence for one immutable context; consistency is checked separately."""

    model: str
    evidence: tuple[Evidence, ...]
    assumptions: tuple[Basic, ...] = ()
    consistent: bool | None = True
    issues: tuple[str, ...] = ()
    labels: tuple[tuple[Symbol, str], ...] = ()

    def __post_init__(self) -> None:
        ids: set[str] = set()
        for item in self._walk():
            if item.id in ids:
                raise ModelDefinitionError(f"Evidence IDs must be unique within a verification report: {item.id}")
            ids.add(item.id)

    def _walk(self):
        """Visit nested evidence without depending on Python's recursion limit."""
        pending = list(self.evidence)
        while pending:
            item = pending.pop()
            yield item
            pending.extend(item.premises)

    def explain(self, obligation_id: str | Obligation | InferenceRule) -> Evidence:
        """Find evidence by ID, obligation, or an inference rule's conclusion."""

        subject = obligation_id.conclusion if isinstance(obligation_id, InferenceRule) else obligation_id
        if isinstance(subject, str):
            matches = [item for item in self._walk() if item.id == subject]
        else:
            proposition = subject._proposition()
            matches = [item for item in self.evidence if item.obligation._proposition() == proposition]
            if not matches:
                matches = [item for item in self._walk() if item.obligation._proposition() == proposition]
            named = [item for item in matches if item.id == subject.name]
            if named:
                matches = named
        if len(matches) != 1:
            raise KeyError(obligation_id)
        return matches[0]

    def __str__(self) -> str:
        lines = [f"Verification: {self.model}; assumptions consistent: {self.consistent}"]
        for item in self.evidence:
            condition = f" if {item.condition}" if item.verdict == "conditionally true" else ""
            reason = f" — {item.obligation.reason}" if item.obligation.reason else ""
            lines.append(f"  {item.id}: {item.verdict}{condition}{reason}")
        return "\n".join([*lines, *(f"  Issue: {issue}" for issue in self.issues)])

    def to_dict(self) -> dict[str, Any]:
        """Export descriptive evidence; this is not an executable context format."""
        from ._report import report_dict

        return report_dict(self)

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n"

    def to_html(self, *, title: str = "Mathematical verification report") -> str:
        from ._report import report_html

        return report_html(self, title)


@dataclass(frozen=True)
class InferenceRule:
    """A trusted, versioned rule whose premises are recursively checked.

    Registered inference rules are part of the trust boundary, like the CAS.
    A bibliographic note alone does not register a rule or establish its conclusion.
    """

    name: str
    conclusion: Obligation
    premises: tuple[Obligation, ...]
    source: str
    version: str = "1"

    def __post_init__(self) -> None:
        premises: dict[tuple, Obligation] = {}
        for obligation in self.premises:
            if not isinstance(obligation, Obligation):
                raise ModelDefinitionError("Inference-rule premises must be Obligation records")
            # Keep distinct premises until a catalog can supply their names.
            # Complete scoped records can already share one evidence item.
            key = (obligation._proposition(), obligation.consumer, obligation.reason, obligation.source)
            premises.setdefault(key, obligation)
        object.__setattr__(self, "premises", tuple(premises.values()))


ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class Method(Generic[ResultT]):
    """A guarded computation with preconditions and the model fields it reads."""

    name: str
    execute: Callable[..., ResultT]
    preconditions: tuple[Obligation, ...] = ()
    uses: tuple[str, ...] = ()
    bound_context: VerificationContext[Any, object] | None = field(default=None, repr=False, compare=False)
    bound_assumptions: frozenset[Basic] | None = field(default=None, repr=False, compare=False)

    def bind(self, context: VerificationContext[Any, object]) -> Method[ResultT]:
        """Bind context-specific expressions to their full immutable context.

        Rebuild a bound method after branching. Rebinding stale preconditions
        would not recompute the expressions that captured the earlier context.
        """
        assumptions = frozenset(global_assumptions)
        if self.bound_context is not None and (
            self.bound_context is not context or _expression_key(self.bound_assumptions) != _expression_key(assumptions)
        ):
            raise ModelDefinitionError("Rebuild a context-bound method after changing its context or SymPy assumptions")
        return (
            self
            if self.bound_context is context
            else replace(self, bound_context=context, bound_assumptions=assumptions)
        )


@dataclass(frozen=True)
class VerificationResult(Generic[ResultT]):
    method: str
    data: ResultT
    report: VerificationReport
    settings: Mapping[str, Any]


@dataclass(frozen=True)
class Sample:
    """Finite observations, never a proof of a universal claim."""

    expression: Expr
    coordinate: Symbol
    points: tuple[Expr, ...]
    values: tuple[Expr, ...]
    kind: str = "numerical observation"


class VerificationError(ModelValidationError):
    """A blocked computation retaining its verification context and report."""

    def __init__(self, message: str, context: VerificationContext[Any, object], report: VerificationReport) -> None:
        self.context, self.report = context, report
        super().__init__(message)

"""Shared records and formatting for symbolic problems and solution families."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

from sympy import Basic, Expr, FiniteSet, Interval, S, Set, Symbol, SympifyError, sympify
from sympy.printing.str import StrPrinter

from modello._expressions import _DefinitionResolver, _expand_definitions
from modello._paths import _path_label
from modello._symbols import Index
from modello.errors import ModelInputError, ModelValidationError
from modello.vectors import Vector

if sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnum
else:
    from enum import Enum

    class _StrEnum(str, Enum):
        """String-enum behaviour for the supported Python 3.10 runtime."""

        __str__ = str.__str__

        def __format__(self, format_spec: str) -> str:
            return str.__format__(self, format_spec)


if TYPE_CHECKING:
    from modello import Modello, ScalarValue
    from modello.diagnostics import ModelProblem

ModelT = TypeVar("ModelT", bound="Modello")

FieldPath = tuple[str, ...]
Status = Literal["consistent", "inconsistent", "unknown"]
Verdict = Literal["satisfied", "violated", "unknown"]


class ConstraintKind(_StrEnum):
    """A defining equation, supplied input, or inferred component result."""

    DEFINITION = "definition"
    INPUT = "input"
    INFERRED = "inferred"


@dataclass(frozen=True)
class ModelField:
    """One public path to a quantity; aliases retain the same symbol identity."""

    path: FieldPath
    symbol: Symbol


@dataclass(frozen=True)
class ModelConstraint:
    """An equation's origin, original sides, and optional constructor-input group.

    Copied component values distinguish supplied inputs from inferred results.
    ``input`` groups both under the component argument;
    it does not mean that each copied value was supplied explicitly.
    """

    index: int
    path: FieldPath
    kind: ConstraintKind
    left: Symbol
    right: Basic
    input: str | None = None

    @property
    def symbols(self) -> frozenset[Symbol]:
        """Return identities mentioned by either side, without simplifying them."""
        return frozenset(self.left.free_symbols | self.right.free_symbols)


@dataclass(frozen=True)
class _Algebra:
    """The evaluated view of a snapshot, leaving original equations intact."""

    definitions: dict[Symbol, Basic]
    cyclic: tuple[Symbol, ...]
    equations: tuple[tuple[Symbol, Basic], ...]
    resolve: _DefinitionResolver


def _algebra(constraints: tuple[ModelConstraint, ...]) -> _Algebra:
    definitions: dict[Symbol, Basic] = {}
    ambiguous: set[Symbol] = set()
    for constraint in constraints:
        if constraint.kind is ConstraintKind.DEFINITION:
            previous = definitions.get(constraint.left)
            if previous is not None and previous != constraint.right:
                ambiguous.add(constraint.left)
            definitions[constraint.left] = constraint.right
    expanded, cyclic, resolve = _expand_definitions(definitions, frozenset(ambiguous))
    return _Algebra(expanded, cyclic, tuple((c.left, resolve(c.right)) for c in constraints), resolve)


@dataclass(frozen=True, init=False)
class FieldValue:
    """A value associated with an actual field of this problem."""

    field: ModelField
    value: Expr

    def __init__(self, field: ModelField, value: ScalarValue) -> None:
        try:
            expression = value if isinstance(value, Expr) else sympify(value)
        except (SympifyError, TypeError) as error:
            raise ModelInputError("A field value must be a SymPy-compatible scalar") from error
        if not isinstance(expression, Expr):
            raise ModelInputError("A field value must be a SymPy-compatible scalar")
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
            raise ModelValidationError("Candidate contains conflicting values for aliases")
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


def _common_prefix(left: FieldPath, right: FieldPath) -> FieldPath:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return left[:index]
    return left[: min(len(left), len(right))]


class _FieldPrinter(StrPrinter):
    """Label quantities without substitution, which could change the algebra."""

    def __init__(
        self,
        fields: tuple[ModelField, ...],
        symbols: frozenset[Symbol],
        scope: FieldPath,
        referenced: frozenset[Symbol],
    ) -> None:
        super().__init__()
        paths: dict[Symbol, FieldPath] = {}
        for field in fields:
            previous = paths.get(field.symbol)
            # Prefer aliases in this component; ties keep the recorded order.
            if previous is None or len(_common_prefix(scope, field.path[:-1])) > len(
                _common_prefix(scope, previous[:-1])
            ):
                paths[field.symbol] = field.path
        self.scope = scope
        self._bound_symbols: frozenset[Symbol] = frozenset()
        self._field_symbols = frozenset(paths)
        # Both sides share one scope, even when the right side reaches outside it.
        for symbol in referenced & paths.keys():
            self.scope = _common_prefix(self.scope, paths[symbol][:-1])
        self.labels = {symbol: _path_label(path[len(self.scope) :]) for symbol, path in paths.items()}
        groups: dict[FieldPath, list[ModelField]] = {}
        for field in fields:
            if field.path[-1].isdecimal():
                groups.setdefault(field.path[:-1], []).append(field)
        self.vector_labels = {
            Vector(*(field.symbol for field in sorted(group, key=lambda field: int(field.path[-1])))): _path_label(
                prefix[len(self.scope) :]
            )
            or prefix[-1]
            for prefix, group in reversed(tuple(groups.items()))
        }
        external: dict[str, list[Symbol]] = {}
        for symbol in sorted(
            symbols - self.labels.keys(),
            key=lambda s: (s.name, str(sorted(s.assumptions0.items())), getattr(s, "dummy_index", -1)),
        ):
            external.setdefault(symbol.name, []).append(symbol)
        for name, group in external.items():
            for index, symbol in enumerate(group, 1):
                suffix = f"#{index}" if len(group) > 1 else ""
                self.labels[symbol] = f"external[{name}{suffix}]"

    def _print_Symbol(self, expression: Symbol) -> str:
        if expression in self._bound_symbols and expression not in self._field_symbols:
            return expression.name
        return self.labels.get(expression, super()._print_Symbol(expression))

    def _print_Dummy(self, expression: Symbol) -> str:
        if expression in self._bound_symbols and expression not in self._field_symbols:
            return expression.name
        return self.labels.get(expression, super()._print_Dummy(expression))

    def _print_Index(self, expression: Index) -> str:
        return expression.name

    def _with_bound(self, expression: Basic, render: Callable[[Basic], str]) -> str:
        previous = self._bound_symbols
        self._bound_symbols |= frozenset(getattr(expression, "variables", ()))
        try:
            return render(expression)
        finally:
            self._bound_symbols = previous

    def _print_Sum(self, expression: Basic) -> str:
        return self._with_bound(expression, super()._print_Sum)

    def _print_Product(self, expression: Basic) -> str:
        return self._with_bound(expression, super()._print_Basic)

    def _print_Integral(self, expression: Basic) -> str:
        return self._with_bound(expression, super()._print_Integral)

    def _print_Derivative(self, expression: Basic) -> str:
        return self._with_bound(expression, super()._print_Derivative)

    def _print_Limit(self, expression: Basic) -> str:
        previous = self._bound_symbols
        self._bound_symbols |= {expression.args[1]}
        try:
            return super()._print_Limit(expression)
        finally:
            self._bound_symbols = previous

    def _print_VectorFormula(self, expression: Any) -> str:
        previous = self._bound_symbols
        self._bound_symbols |= frozenset(expression.bound_symbols)
        try:
            return f"vector({self.doprint(expression.function)}, {self.doprint(expression.limits)})"
        finally:
            self._bound_symbols = previous


def _canonical_fields(problem: ModelProblem) -> tuple[ModelField, ...]:
    fields: dict[Symbol, ModelField] = {}
    for field in problem.fields:
        fields.setdefault(field.symbol, field)
    return tuple(fields.values())


def _domain(symbol: Symbol) -> Set:
    """A numeric search domain containing every permitted finite scalar value."""
    domain = (
        S.Integers
        if symbol.is_integer
        else S.Rationals
        if symbol.is_rational
        else S.Reals
        if symbol.is_real
        else S.Complexes
    )
    if symbol.is_real:
        if symbol.is_positive:
            domain &= Interval.open(0, S.Infinity)
        elif symbol.is_nonnegative:
            domain &= Interval(0, S.Infinity)
        elif symbol.is_negative:
            domain &= Interval.open(S.NegativeInfinity, 0)
        elif symbol.is_nonpositive:
            domain &= Interval(S.NegativeInfinity, 0)
    if symbol.is_zero is False:
        domain -= FiniteSet(0)
    return domain

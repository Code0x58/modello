"""Validation of observed recurrence windows and row streams."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import count
from typing import Literal, TypeAlias

from sympy import Abs, Basic, Dummy, Expr, Integer, Rational, S, Symbol, check_assumptions, simplify, sympify

from modello import Modello, ScalarValue
from modello._evaluation import _evaluate, _expand_finite_indexed, _substitute
from modello._operators import _conditioned, _definitely_undefined, _denominator_condition, _scalar_atoms
from modello._paths import FieldPath, _path_label
from modello._problem import _domain
from modello.errors import ModelInputError
from modello.model_vectors import _collection_columns
from modello.vectors import _VectorElement

__all__ = ["CheckResult", "ValidationSummary", "WindowResult", "WindowValidator"]

CheckVerdict: TypeAlias = Literal["satisfied", "violated", "unknown", "assumed"]
CheckKind: TypeAlias = Literal["domain", "definition", "alias", "recurrence"]
Row: TypeAlias = Mapping[str, object] | Sequence[object]
_UNLABELLED = object()


@dataclass(frozen=True)
class CheckResult:
    """One declared check, including checks skipped as initial history.

    ``start`` is the first absolute row position where this check applies.
    ``assumed`` means it has not run. Unknown dependencies are not violations.
    Equation residuals are observed minus expected; ``index`` is only a label.
    """

    position: int
    index: Hashable
    field: str
    rule: str
    observed: object
    expected: Basic | None
    residual: Expr | None
    verdict: CheckVerdict
    reason: str = ""
    start: int = 0
    kind: CheckKind = "recurrence"


@dataclass(frozen=True)
class WindowResult:
    """All checks of the final row of one explicitly positioned window."""

    position: int
    index: Hashable
    checks: tuple[CheckResult, ...]

    @property
    def first_invalid(self) -> CheckResult | None:
        """The first proven violation, without claiming coverage for other checks."""
        return next((check for check in self.checks if check.verdict == "violated"), None)

    @property
    def complete(self) -> bool:
        """Whether every declared check ran and reached a decided outcome."""
        return bool(self.checks) and all(check.verdict in ("satisfied", "violated") for check in self.checks)

    @property
    def assumed(self) -> bool:
        """Whether all checks are initial-history assumptions at this position."""
        return bool(self.checks) and all(check.verdict == "assumed" for check in self.checks)

    @property
    def valid(self) -> bool | None:
        """False for a proved violation, True for complete success, otherwise None."""
        if self.first_invalid is not None:
            return False
        return True if self.complete else None


@dataclass(frozen=True)
class ValidationSummary:
    """Coverage and evidence from a complete scan, using constant working memory.

    ``checked_rows`` counts rows where every check ran and was decided, including
    fully checked invalid rows. ``assumed_rows`` counts wholly assumed history;
    ``unknown_rows`` counts other rows without a decided validity. A violated row
    can still have incomplete coverage, so these counts are not interchangeable.
    """

    row_count: int
    checked_rows: int
    assumed_rows: int
    unknown_rows: int
    violated_rows: int
    first_invalid: CheckResult | None
    start_position: int = 0

    @classmethod
    def from_results(cls, results: Iterable[WindowResult], *, start_position: int = 0) -> ValidationSummary:
        """Summarise a contiguous interval, starting from zero by default.

        Gaps, duplicate positions, and reordered results raise ModelInputError;
        omitting a prefix must not turn partial coverage into a complete audit.
        Periodic stream consumers may explicitly name another start position;
        the returned interval then describes only that selected horizon.
        """
        if isinstance(start_position, bool) or not isinstance(start_position, int) or start_position < 0:
            raise ModelInputError("Audit start_position must be an absolute nonnegative integer")
        rows = checked = assumed = unknown = violated = 0
        first = None
        for result in results:
            if result.position != start_position + rows:
                raise ModelInputError(
                    f"Audit results require every position from {start_position} in order; "
                    f"expected {start_position + rows}"
                )
            rows += 1
            checked += result.complete
            assumed += result.assumed
            invalid = result.first_invalid
            if invalid is not None:
                violated += 1
                if first is None:
                    first = invalid
            elif not result.assumed and result.valid is None:
                unknown += 1
        return cls(rows, checked, assumed, unknown, violated, first, start_position)

    @property
    def stop_position(self) -> int:
        """Exclusive end of the audited interval; preceding history is outside it."""
        return self.start_position + self.row_count

    @property
    def complete(self) -> bool:
        """Whether every row in the nonempty audited interval was fully checked."""
        return self.row_count > 0 and self.checked_rows == self.row_count

    @property
    def valid(self) -> bool | None:
        """A strict interval verdict; assumed or unresolved rows prevent True."""
        if self.first_invalid is not None:
            return False
        return True if self.complete else None


@dataclass(frozen=True)
class _Observation:
    raw: object
    value: Expr
    problem: Literal["missing observation", "not a scalar number"] | None = None


@dataclass(frozen=True)
class _Reference:
    symbol: Symbol
    field: int
    lag: int


@dataclass(frozen=True)
class _Rule:
    name: str
    field: int
    start: int
    expression: Expr
    references: tuple[_Reference, ...]
    index: Symbol | None = None
    kind: CheckKind = "recurrence"


class WindowValidator:
    """Compile a row schema and causal laws into a stateless window function.

    Rows are flat mappings keyed by ``fields`` or sequences in exactly that
    order. A window is oldest first and ends at the row being checked. Its size
    must be ``min(position + 1, window_size)``; only true stream prefixes may be
    short. ``position`` is required, absolute, zero based, and independent of
    labels, chunks, or the current window's length. Null adapters use ``None``.
    """

    def __init__(
        self,
        model: type[Modello],
        rows: str,
        *,
        parameters: Mapping[str, ScalarValue] | None = None,
        atol: ScalarValue = 0,
        rtol: ScalarValue = 0,
    ) -> None:
        if not isinstance(model, type) or not issubclass(model, Modello):
            raise ModelInputError("A window validator requires a Modello class")
        namespace = model._modello_namespace
        if rows not in namespace.collections:
            raise ModelInputError(f"{rows!r} must name a declared model_vector on {model.__name__}")
        self.model = model
        self.rows = rows
        self._row_model = namespace.collections[rows].model
        self._fields = tuple(self._row_model._modello_definition.fields.items())
        self._positions = {path: i for i, (path, _) in enumerate(self._fields)}
        self._atol, self._rtol = (_tolerance(value) for value in (atol, rtol))
        known = _parameters(model, rows, parameters or {})
        columns = _collection_columns(namespace.nested_models[rows])
        self._columns = {column: path for path, column in reversed(tuple(columns.items()))}
        self._symbols = {symbol: path for path, symbol in reversed(self._fields)}
        laws = [(name, law) for name, law in namespace.model_laws.items() if law.target.path == (rows,)]
        if not laws:
            raise ModelInputError(f"No model_recurrence targets {rows!r}")
        self._start = min(int(law.template.start) for _, law in laws)
        self._complete_start = max(int(law.template.start) for _, law in laws)
        rules = [
            self._compile("definition", item.path, item.right, self._start, kind="definition")
            for item in self._row_model._modello_definition.constraints
        ]
        for path, symbol in self._fields:
            if path != self._symbols[symbol]:
                rules.append(self._compile("alias", path, symbol, self._start, kind="alias"))
        for name, law in laws:
            for path, expression in law.updates:
                body = _substitute(_conditioned(expression, _denominator_condition(expression)), known)
                rules.append(
                    self._compile(name, (*law.target.suffix, *path), body, int(law.template.start), law.template.index)
                )
        self._rules = tuple(rules)
        self._window_size = 1 + max((ref.lag for rule in rules for ref in rule.references), default=0)

    @property
    def fields(self) -> tuple[str, ...]:
        """Required flat mapping keys, also the exact order for projected row tuples."""
        return tuple(_path_label(path) for path, _ in self._fields)

    @property
    def field_paths(self) -> tuple[FieldPath, ...]:
        """The same fields as nested attribute paths, for adapter column binding."""
        return tuple(path for path, _ in self._fields)

    @property
    def order(self) -> int:
        """Largest fixed lag referenced by the prepared laws."""
        return self.window_size - 1

    @property
    def window_size(self) -> int:
        """Required rolling-window width: order plus one."""
        return self._window_size

    @property
    def first_test_position(self) -> int:
        """Earliest declared start; normally p for one order-p law."""
        return self._start

    @property
    def fully_testable_position(self) -> int:
        """Earliest position where every law has started, before data checks."""
        return self._complete_start

    def check_window(
        self,
        window: Sequence[Row],
        *,
        position: int,
        index: Hashable = _UNLABELLED,
    ) -> WindowResult:
        """Check only the last row, retaining explicit assumptions and unknowns.

        No history is stored between calls. Missing or excess window rows, wrong
        row widths, missing mapping keys, and invalid positions raise
        ModelInputError. Malformed scalar cells instead produce domain violations.
        Coverage describes the current row only; preceding window entries are
        supplied observations and are not independently audited by this call.
        """
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise ModelInputError("Window position must be an absolute nonnegative integer")
        expected = min(position + 1, self.window_size)
        try:
            size = len(window)
        except TypeError as error:
            raise ModelInputError("A validation window must be a finite sequence of rows") from error
        if size != expected:
            raise ModelInputError(f"Position {position} requires exactly {expected} window rows, got {size}")
        label = position if index is _UNLABELLED else index
        if not isinstance(label, Hashable):
            raise ModelInputError("A row label must be hashable")
        observed = tuple(tuple(_observe(value) for value in self._project(row)) for row in window)
        current = observed[-1]
        checks = tuple(
            _domain_check(position, label, path, symbol, value, self._start)
            for (path, symbol), value in zip(self._fields, current, strict=True)
        )
        checks += tuple(self._check(rule, position, label, observed) for rule in self._rules)
        return WindowResult(position, label, checks)

    __call__ = check_window

    def iter_results(
        self,
        rows: Iterable[Row],
        *,
        labels: Iterable[Hashable] | None = None,
    ) -> Iterator[WindowResult]:
        """Scan one ordered stream with at most window_size retained rows.

        Each call starts at position zero. To span batches, chain their row
        iterators into one call; alternatively retain your own overlap and call
        check_window with absolute positions. Labels must have exactly one entry
        per row and never determine the mathematical coordinate.
        """
        window: deque[tuple[object, ...]] = deque(maxlen=self.window_size)
        labelled = zip(rows, count() if labels is None else labels, strict=labels is not None)
        for position, (row, label) in enumerate(labelled):
            window.append(self._project(row))
            yield self(tuple(window), position=position, index=label)

    def audit(self, rows: Iterable[Row], *, labels: Iterable[Hashable] | None = None) -> ValidationSummary:
        """Scan the entire input and return coverage plus the first proved violation."""
        return ValidationSummary.from_results(self.iter_results(rows, labels=labels))

    def _project(self, row: Row) -> tuple[object, ...]:
        if isinstance(row, Mapping):
            missing = [field for field in self.fields if field not in row]
            if missing:
                raise ModelInputError(f"Missing row fields: {', '.join(missing)}")
            return tuple(row[field] for field in self.fields)
        if isinstance(row, (str, bytes)):
            raise ModelInputError("A projected row must be a sequence of scalar cells")
        try:
            values = tuple(row)
        except TypeError as error:
            raise ModelInputError("A projected row must be a sequence of scalar cells") from error
        if len(values) != len(self._fields):
            raise ModelInputError(f"Projected rows require {len(self._fields)} fields in validator.fields order")
        return values

    def _compile(
        self,
        name: str,
        path: FieldPath,
        body: Expr,
        start: int,
        index: Symbol | None = None,
        kind: CheckKind = "recurrence",
    ) -> _Rule:
        body = _expand_finite_indexed(body)
        condition = _denominator_condition(body)
        replacements = {}
        references = []
        for term in sorted(_scalar_atoms(body, _VectorElement), key=str):
            if index is None or term.vector not in self._columns:
                raise ModelInputError("Window laws must reference only the selected row vector")
            delay = simplify(index - term.index)
            if not isinstance(delay, Integer) or not 0 <= delay <= start:
                raise ModelInputError(
                    "Window laws require fixed past lags covered by start; future and fixed rows are unsupported"
                )
            symbol = Dummy("observed")
            replacements[term] = symbol
            references.append(_Reference(symbol, self._positions[self._columns[term.vector]], int(delay)))
        for symbol in body.free_symbols & self._symbols.keys():
            placeholder = Dummy("observed")
            replacements[symbol] = placeholder
            references.append(_Reference(placeholder, self._positions[self._symbols[symbol]], 0))
        # Capture source denominator requirements before substitution can cancel them.
        expression = _conditioned(_substitute(body, replacements), _substitute(condition, replacements))
        unexpected = expression.free_symbols - {ref.symbol for ref in references} - {index}
        if unexpected or _scalar_atoms(expression, _VectorElement):
            names = ", ".join(sorted(str(symbol) for symbol in unexpected))
            raise ModelInputError(f"Unbound window-law quantities: {names}. Supply scalar parameters or row fields.")
        return _Rule(name, self._positions[path], start, expression, tuple(references), index, kind)

    def _check(
        self, rule: _Rule, position: int, label: Hashable, window: Sequence[tuple[_Observation, ...]]
    ) -> CheckResult:
        observation = window[-1][rule.field]
        if position < rule.start:
            return _assumed(
                position, label, self._fields[rule.field][0], rule.name, observation.raw, rule.start, rule.kind
            )
        values = {ref.symbol: window[-1 - ref.lag][ref.field].value for ref in rule.references}
        if rule.index is not None:
            values[rule.index] = Integer(position)
        expected = _evaluate(_substitute(rule.expression, values))
        residual = None
        verdict: CheckVerdict = "unknown"
        reason = "unresolved or missing dependency"
        if _definitely_undefined(expected) or expected.is_finite is False:
            verdict, reason = "violated", "undefined expression"
        elif observation.problem:
            reason = observation.problem
        elif expected.is_number is True and expected.is_finite is True and observation.value.is_number is True:
            residual = simplify(observation.value - expected)
            within = Abs(residual) <= self._atol + self._rtol * Abs(expected)
            if within is S.true:
                verdict, reason = "satisfied", ""
            elif within is S.false:
                verdict, reason = "violated", "observed differs from expected"
        unavailable = {
            window[-1 - ref.lag][ref.field].value for ref in rule.references if window[-1 - ref.lag][ref.field].problem
        }
        return CheckResult(
            position,
            label,
            _path_label(self._fields[rule.field][0]),
            rule.name,
            observation.raw,
            None if expected.free_symbols & unavailable else expected,
            residual,
            verdict,
            reason,
            rule.start,
            rule.kind,
        )


def _assumed(
    position: int,
    label: Hashable,
    path: FieldPath,
    rule: str,
    observed: object,
    start: int,
    kind: CheckKind,
) -> CheckResult:
    return CheckResult(
        position,
        label,
        _path_label(path),
        rule,
        observed,
        None,
        None,
        "assumed",
        f"Initial history; first test position is {start}",
        start,
        kind,
    )


def _observe(value: object) -> _Observation:
    missing = value is None or isinstance(value, Decimal) and value.is_nan()
    if missing:
        return _Observation(value, Dummy("missing"), "missing observation")
    try:
        # Decimal amounts retain their exact decimal value. Text cells are never parsed.
        expression = (
            Rational(*value.as_integer_ratio())
            if isinstance(value, Decimal) and value.is_finite()
            else sympify(value, strict=True)
        )
    except (TypeError, ValueError):
        expression = None
    if expression is S.NaN:
        return _Observation(value, Dummy("missing"), "missing observation")
    if not isinstance(expression, Expr) or expression.is_Matrix:
        return _Observation(value, Dummy("invalid"), "not a scalar number")
    return _Observation(value, expression)


def _domain_check(
    position: int, label: Hashable, path: FieldPath, symbol: Symbol, value: _Observation, start: int
) -> CheckResult:
    if position < start:
        return _assumed(position, label, path, "domain", value.raw, start, "domain")
    verdict: CheckVerdict = "unknown"
    reason: str = value.problem or "unresolved numeric value"
    if value.problem == "not a scalar number":
        verdict = "violated"
    elif not value.problem and value.value.is_number is True:
        result = check_assumptions(value.value, against=symbol)
        if value.value.is_finite is False or result is False:
            verdict, reason = "violated", "value is outside the field domain"
        elif value.value.is_finite is True and result is True:
            verdict, reason = "satisfied", ""
    return CheckResult(
        position, label, _path_label(path), "domain", value.raw, _domain(symbol), None, verdict, reason, start, "domain"
    )


def _tolerance(value: ScalarValue) -> Expr:
    observation = _observe(value)
    if (
        observation.problem
        or observation.value.is_number is not True
        or observation.value.is_finite is not True
        or observation.value.is_nonnegative is not True
    ):
        raise ModelInputError("Equation tolerances must be finite nonnegative numbers")
    return observation.value


def _parameters(model: type[Modello], rows: str, parameters: Mapping[str, ScalarValue]) -> dict[Symbol, Expr]:
    definition = model._modello_definition
    external = {_path_label(path): symbol for path, symbol in definition.fields.items() if path[0] != rows}
    pending = {
        constraint.left: constraint.right
        for constraint in definition.constraints
        if constraint.left in external.values()
    }
    known = {}
    for name, raw in parameters.items():
        if name not in external or external[name] in pending:
            raise ModelInputError(f"Parameter {name!r} must be an independent scalar model field")
        if external[name] in model._modello_namespace.collections[rows].size.free_symbols:
            raise ModelInputError("The input stream supplies the row vector's size; omit size parameters")
        observation = _observe(raw)
        if observation.problem or observation.value.is_number is not True or observation.value.is_finite is not True:
            raise ModelInputError(f"Parameter {name!r} must be a finite scalar number")
        if check_assumptions(observation.value, against=external[name]) is not True:
            raise ModelInputError(f"Parameter {name!r} violates its field domain")
        known[external[name]] = observation.value
    while pending:
        ready = {
            symbol: _evaluate(_substitute(_conditioned(expression, _denominator_condition(expression)), known))
            for symbol, expression in pending.items()
            if expression.free_symbols <= known.keys() and not _scalar_atoms(expression, _VectorElement)
        }
        ready = {
            symbol: value for symbol, value in ready.items() if value.is_number is True and value.is_finite is True
        }
        if not ready:
            break
        for symbol, value in ready.items():
            if check_assumptions(value, against=symbol) is not True:
                name = next(name for name, field in external.items() if field == symbol)
                raise ModelInputError(f"Defined parameter {name!r} violates its field domain")
        known.update(ready)
        for symbol in ready:
            del pending[symbol]
    return known

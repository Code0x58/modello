"""Validation of observed recurrence windows and row streams."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import count
from typing import Literal, TypeAlias

from sympy import Abs, Basic, Dummy, Expr, Integer, Rational, S, Symbol, check_assumptions, simplify, sympify
from sympy.core.logic import fuzzy_and

from modello import Modello, ScalarValue
from modello._evaluation import _evaluate, _expand_finite_indexed, _substitute
from modello._operators import _conditioned, _definitely_undefined, _denominator_condition, _scalar_atoms
from modello._paths import FieldPath, _path_label
from modello._problem import _domain
from modello.errors import ModelInputError
from modello.model_vectors import _collection_columns
from modello.vectors import _VectorElement

__all__ = ["CheckResult", "ValidationResult", "ValidationSummary", "WindowValidator"]

CheckKind: TypeAlias = Literal["domain", "definition", "alias", "recurrence"]
Row: TypeAlias = Mapping[str, object] | Sequence[object]
_UNLABELLED = object()


@dataclass(frozen=True)
class CheckResult:
    """One declared check, including checks skipped as initial history.

    ``valid`` is True for a passing check, False for a proven violation, and
    None for assumed history or an undecidable outcome. ``start`` is the first
    applicable row position; ``assumed`` distinguishes history from uncertainty.
    Equation residuals are observed minus expected; ``index`` is only a label.
    """

    position: int
    index: Hashable
    field: str
    rule: str
    observed: object
    expected: Basic | None
    residual: Expr | None
    valid: bool | None
    reason: str = ""
    start: int = 0
    kind: CheckKind = "recurrence"

    @property
    def invalid(self) -> bool:
        """Whether this check proves a violation; uncertainty is not a violation."""
        return self.valid is not None and not self.valid

    @property
    def assumed(self) -> bool:
        """Whether this check has not started at the current row position."""
        return self.position < self.start


@dataclass(frozen=True)
class ValidationResult:
    """Validation outcome and check details for one row."""

    position: int
    index: Hashable
    checks: tuple[CheckResult, ...]

    @property
    def invalid(self) -> bool:
        """Whether any check proves this row invalid."""
        return any(check.invalid for check in self.checks)

    @property
    def complete(self) -> bool:
        """Whether every current-row check ran and reached a decided outcome."""
        return bool(self.checks) and all(check.valid is not None for check in self.checks)

    @property
    def assumed(self) -> bool:
        """Whether all checks are initial-history assumptions at this position."""
        return bool(self.checks) and all(check.assumed for check in self.checks)

    @property
    def valid(self) -> bool | None:
        """False for a proven violation, True for complete success, otherwise None."""
        return fuzzy_and(check.valid for check in self.checks) if self.checks else None


@dataclass(frozen=True)
class ValidationSummary:
    """Row counts, coverage, and the first invalid row in a finite interval.

    The valid, invalid, assumed, and unknown row counts partition the input.
    Assumed rows contain only initial-history checks; unknown rows are other
    rows whose validity is undecidable.
    An invalid row may still contain undecidable checks; ``complete`` records
    whether every check in the nonempty interval reached a decision.
    """

    valid_rows: int
    invalid_rows: int
    assumed_rows: int
    unknown_rows: int
    complete: bool
    first_invalid: ValidationResult | None
    start_position: int = 0

    @classmethod
    def from_results(cls, results: Iterable[ValidationResult], *, start_position: int = 0) -> ValidationSummary:
        """Summarise a contiguous interval, starting from zero by default.

        Gaps, duplicates, or reordered results raise ModelInputError. Supply
        another start_position explicitly to audit a later interval; earlier
        observations are outside its coverage.
        """
        if isinstance(start_position, bool) or not isinstance(start_position, int) or start_position < 0:
            raise ModelInputError("Audit start_position must be an absolute nonnegative integer")
        valid_rows = invalid_rows = assumed_rows = unknown_rows = 0
        complete = True
        first = None
        for position, result in enumerate(results, start=start_position):
            if result.position != position:
                raise ModelInputError(
                    f"Audit results require every position from {start_position} in order; expected {position}"
                )
            complete = complete and result.complete
            valid = result.valid
            if valid:
                valid_rows += 1
            elif valid is not None:
                invalid_rows += 1
                if first is None:
                    first = result
            elif result.assumed:
                assumed_rows += 1
            else:
                unknown_rows += 1
        return cls(
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            assumed_rows=assumed_rows,
            unknown_rows=unknown_rows,
            complete=bool(valid_rows + invalid_rows + assumed_rows + unknown_rows) and complete,
            first_invalid=first,
            start_position=start_position,
        )

    @property
    def row_count(self) -> int:
        """Total rows in the audited interval."""
        return self.valid_rows + self.invalid_rows + self.assumed_rows + self.unknown_rows

    @property
    def stop_position(self) -> int:
        """Exclusive end of the audited interval; preceding history is outside it."""
        return self.start_position + self.row_count

    @property
    def invalid(self) -> bool:
        """Whether the audited interval contains a proven invalid row."""
        return bool(self.invalid_rows)

    @property
    def valid(self) -> bool | None:
        """False for a proven violation; True only for a nonempty interval of passing rows.

        Assumed or undecidable rows otherwise yield None, as does empty input.
        """
        if self.invalid:
            return False
        return True if self.row_count and self.valid_rows == self.row_count else None


@dataclass(frozen=True)
class _Observation:
    raw: object
    value: Expr
    problem: Literal["missing observation", "not a scalar number"] | None = None


@dataclass(frozen=True)
class _Rule:
    name: str
    field: int
    start: int
    expression: Basic
    kind: CheckKind = "recurrence"


class WindowValidator:
    """Compile a row schema and causal laws into a stateless window function.

    Rows are flat mappings keyed by ``fields`` or sequences in exactly that
    order. A window is oldest first and ends at the row being checked. Its size
    must be ``min(position + 1, window_size)``; only true stream prefixes may be
    short. ``position`` is required, absolute, zero based, and independent of
    labels, batches, or the current window's length. Convert framework-specific
    null sentinels to ``None``.
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
        row_definition = namespace.collections[rows].model._modello_definition
        self._fields = tuple(row_definition.fields.items())
        self._field_names = tuple(_path_label(path) for path, _ in self._fields)
        positions = {path: i for i, (path, _) in enumerate(self._fields)}
        self._atol, self._rtol = (_tolerance(value) for value in (atol, rtol))
        known = _parameters(model, rows, parameters or {})
        columns = _collection_columns(namespace.nested_models[rows])
        # Resolve aliases through their first declared field, then check each copy.
        self._sources = {symbol: positions[path] for path, symbol in reversed(self._fields)}
        self._sources.update({column: positions[path] for path, column in reversed(tuple(columns.items()))})
        self._coordinates: dict[tuple[int, int], Symbol] = {}
        self._index = Dummy("position")
        laws = [(name, law) for name, law in namespace.model_laws.items() if law.target.path == (rows,)]
        if not laws:
            raise ModelInputError(f"No model_recurrence targets {rows!r}")
        self._start = min(int(law.template.start) for _, law in laws)
        self._complete_start = max(int(law.template.start) for _, law in laws)
        rules = [
            _Rule("domain", i, self._start, _domain(symbol), kind="domain")
            for i, (_, symbol) in enumerate(self._fields)
        ]
        rules.extend(
            self._compile("definition", positions[item.path], item.right, self._start, kind="definition")
            for item in row_definition.constraints
        )
        for path, symbol in self._fields:
            if positions[path] != self._sources[symbol]:
                rules.append(self._compile("alias", positions[path], symbol, self._start, kind="alias"))
        for name, law in laws:
            for path, expression in law.updates:
                body = _substitute(_conditioned(expression, _denominator_condition(expression)), known)
                rules.append(
                    self._compile(
                        name, positions[law.target.suffix + path], body, int(law.template.start), law.template.index
                    )
                )
        self._rules = tuple(rules)
        self._window_size = 1 + max((lag for lag, _ in self._coordinates), default=0)

    @property
    def fields(self) -> tuple[str, ...]:
        """Flat mapping keys, in the required order for row tuples."""
        return self._field_names

    @property
    def field_paths(self) -> tuple[FieldPath, ...]:
        """Nested paths for selecting source columns, in fields order."""
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
        """Earliest declared law start; often equal to order."""
        return self._start

    @property
    def fully_testable_position(self) -> int:
        """Earliest position where all laws apply; values may still be undecidable."""
        return self._complete_start

    def __call__(
        self,
        window: Sequence[Row],
        *,
        position: int,
        index: Hashable = _UNLABELLED,
    ) -> ValidationResult:
        """Check only the last row, retaining explicit assumptions and unknowns.

        No history is stored between calls. Wrong window/row sizes, missing
        mapping keys, invalid positions, and unhashable labels raise
        ModelInputError. Malformed scalars produce domain violations when their
        checks apply. Preceding rows supply observations for the current checks;
        their own validation requires separate calls.
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
        return self._validate(tuple(self._observe_row(row) for row in window), position, index)

    def iter_results(
        self,
        rows: Iterable[Row],
        *,
        labels: Iterable[Hashable] | None = None,
    ) -> Iterator[ValidationResult]:
        """Scan one ordered stream with at most window_size retained rows.

        Each call starts at position zero. To span batches, chain their row
        iterators into one call; alternatively retain your own overlap and call
        this validator with absolute positions. Labels are reporting metadata;
        their count must match the rows (ValueError on mismatch).
        Each row is projected and converted once when read.
        """
        window: deque[tuple[_Observation, ...]] = deque(maxlen=self.window_size)
        labelled = zip(rows, count() if labels is None else labels, strict=labels is not None)
        for position, (row, label) in enumerate(labelled):
            window.append(self._observe_row(row))
            yield self._validate(tuple(window), position, label)

    def audit(self, rows: Iterable[Row], *, labels: Iterable[Hashable] | None = None) -> ValidationSummary:
        """Summarise the finite input, including its first invalid row and coverage."""
        return ValidationSummary.from_results(self.iter_results(rows, labels=labels))

    def _validate(self, window: Sequence[tuple[_Observation, ...]], position: int, index: Hashable) -> ValidationResult:
        label = position if index is _UNLABELLED else index
        if not isinstance(label, Hashable):
            raise ModelInputError("A row label must be hashable")
        values = {self._index: Integer(position)}
        for (lag, field), symbol in self._coordinates.items():
            if lag < len(window):
                observation = window[-1 - lag][field]
                if not observation.problem:
                    values[symbol] = observation.value
        return ValidationResult(
            position,
            label,
            tuple(self._check(rule, position, label, window[-1][rule.field], values) for rule in self._rules),
        )

    def _observe_row(self, row: Row) -> tuple[_Observation, ...]:
        if isinstance(row, Mapping):
            missing = [field for field in self.fields if field not in row]
            if missing:
                raise ModelInputError(f"Missing row fields: {', '.join(missing)}")
            values = tuple(row[field] for field in self.fields)
        else:
            if isinstance(row, (str, bytes)):
                raise ModelInputError("A projected row must be a sequence of scalar cells")
            try:
                values = tuple(row)
            except TypeError as error:
                raise ModelInputError("A projected row must be a sequence of scalar cells") from error
            if len(values) != len(self._fields):
                raise ModelInputError(f"Projected rows require {len(self._fields)} fields in validator.fields order")
        return tuple(_observe(value) for value in values)

    def _compile(
        self,
        name: str,
        field: int,
        body: Expr,
        start: int,
        index: Symbol | None = None,
        kind: CheckKind = "recurrence",
    ) -> _Rule:
        body = _expand_finite_indexed(body)
        condition = _denominator_condition(body)
        locations = {}
        for term in sorted(_scalar_atoms(body, _VectorElement), key=str):
            if index is None or term.vector not in self._sources:
                raise ModelInputError("Window laws must reference only the selected row vector")
            delay = simplify(index - term.index)
            if not isinstance(delay, Integer) or not 0 <= delay <= start:
                raise ModelInputError(
                    "Window laws require fixed past lags covered by start; future and fixed rows are unsupported"
                )
            locations[term] = (int(delay), self._sources[term.vector])
        for symbol in body.free_symbols & self._sources.keys():
            locations[symbol] = (0, self._sources[symbol])
        replacements = {
            source: self._coordinates.setdefault(location, Dummy("observed")) for source, location in locations.items()
        }
        if index is not None:
            replacements[index] = self._index
        # Capture source denominator requirements before substitution can cancel them.
        expression = _conditioned(_substitute(body, replacements), _substitute(condition, replacements))
        unexpected = expression.free_symbols.difference(replacements.values())
        if unexpected or _scalar_atoms(expression, _VectorElement):
            names = ", ".join(sorted(str(symbol) for symbol in unexpected))
            raise ModelInputError(f"Unbound window-law quantities: {names}. Supply scalar parameters or row fields.")
        return _Rule(name, field, start, expression, kind)

    def _check(
        self,
        rule: _Rule,
        position: int,
        label: Hashable,
        observation: _Observation,
        values: Mapping[Symbol, Expr],
    ) -> CheckResult:
        expected: Basic | None = None
        residual = None
        valid: bool | None = None
        if position < rule.start:
            reason = f"Initial history; first test position is {rule.start}"
        elif rule.kind == "domain":
            symbol = self._fields[rule.field][1]
            expected = rule.expression
            reason = observation.problem or "unresolved numeric value"
            if observation.problem == "not a scalar number":
                valid = False
            elif not observation.problem and observation.value.is_number is True:
                valid = fuzzy_and((observation.value.is_finite, check_assumptions(observation.value, against=symbol)))
                if valid is not None:
                    reason = "" if valid else "value is outside the field domain"
        else:
            expected = _evaluate(_substitute(rule.expression, values))
            reason = "unresolved or missing dependency"
            if _definitely_undefined(expected) or expected.is_finite is False:
                valid, reason = False, "undefined expression"
            elif observation.problem:
                reason = observation.problem
            elif _finite_number(expected) and observation.value.is_number is True:
                residual = simplify(observation.value - expected)
                within = Abs(residual) <= self._atol + self._rtol * Abs(expected)
                if within is S.true:
                    valid, reason = True, ""
                elif within is S.false:
                    valid, reason = False, "observed differs from expected"
            if not expected.free_symbols.isdisjoint(self._coordinates.values()):
                expected = None
        return CheckResult(
            position,
            label,
            self.fields[rule.field],
            rule.name,
            observation.raw,
            expected,
            residual,
            valid,
            reason,
            rule.start,
            rule.kind,
        )


def _observe(value: object) -> _Observation:
    missing = value is None or isinstance(value, Decimal) and value.is_nan()
    if missing:
        return _Observation(value, S.NaN, "missing observation")
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
        return _Observation(value, S.NaN, "missing observation")
    if not isinstance(expression, Expr) or expression.is_Matrix:
        return _Observation(value, S.NaN, "not a scalar number")
    return _Observation(value, expression)


def _finite_number(value: Basic) -> bool:
    return value.is_number is True and value.is_finite is True


def _tolerance(value: ScalarValue) -> Expr:
    number = _observe(value).value
    if not _finite_number(number) or number.is_nonnegative is not True:
        raise ModelInputError("Equation tolerances must be finite nonnegative numbers")
    return number


def _parameters(model: type[Modello], rows: str, parameters: Mapping[str, ScalarValue]) -> dict[Symbol, Expr]:
    definition = model._modello_definition
    external = {_path_label(path): symbol for path, symbol in definition.fields.items() if path[0] != rows}
    pending = {
        constraint.left: constraint.right
        for constraint in definition.constraints
        if constraint.left in external.values()
    }
    known: dict[Symbol, Expr] = {}
    for name, raw in parameters.items():
        symbol = external.get(name)
        if symbol is None or symbol in pending:
            raise ModelInputError(f"Parameter {name!r} must be an independent scalar model field")
        if symbol in model._modello_namespace.collections[rows].size.free_symbols:
            raise ModelInputError("The input stream supplies the row vector's size; omit size parameters")
        value = _observe(raw).value
        if not _finite_number(value):
            raise ModelInputError(f"Parameter {name!r} must be a finite scalar number")
        if check_assumptions(value, against=symbol) is not True:
            raise ModelInputError(f"Parameter {name!r} violates its field domain")
        if symbol in known and simplify(known[symbol] - value).is_zero is not True:
            raise ModelInputError(f"Conflicting values for parameter {name!r}")
        known.setdefault(symbol, value)
    while pending:
        remaining = len(pending)
        for symbol, expression in tuple(pending.items()):
            if not expression.free_symbols <= known.keys() or _scalar_atoms(expression, _VectorElement):
                continue
            value = _evaluate(_substitute(_conditioned(expression, _denominator_condition(expression)), known))
            if not _finite_number(value):
                continue
            if check_assumptions(value, against=symbol) is not True:
                name = next(name for name, field in external.items() if field == symbol)
                raise ModelInputError(f"Defined parameter {name!r} violates its field domain")
            known[symbol] = value
            del pending[symbol]
        if len(pending) == remaining:
            break
    return known

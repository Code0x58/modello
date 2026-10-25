"""Check observed rows against model equations and bounded recurrence laws."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator, Mapping
from dataclasses import dataclass, fields
from decimal import Decimal
from itertools import groupby
from typing import Literal

import pandas as pd
from pandas.api.types import is_scalar
from sympy import Abs, Basic, Dummy, Expr, Integer, Rational, S, Symbol, check_assumptions, simplify, sympify

from modello import Modello, ScalarValue
from modello._evaluation import _evaluate, _expand_finite_indexed, _substitute
from modello._operators import _conditioned, _definitely_undefined, _denominator_condition, _scalar_atoms
from modello._paths import FieldPath, _path_label
from modello._problem import Verdict, _domain
from modello.errors import ModelInputError
from modello.model_vectors import _collection_columns
from modello.vectors import _VectorElement

__all__ = ["FrameCheck", "FrameValidator"]


@dataclass(frozen=True)
class FrameCheck:
    """One equation or field-domain check at an observed row position.

    ``index`` is the original pandas label; ``position`` disambiguates repeated
    labels. Equation residuals are observed minus expected. Missing dependencies
    produce ``unknown``, while undefined equations and invalid domains violate.
    """

    position: int
    index: Hashable
    field: str
    rule: str
    observed: object
    expected: Basic | None
    residual: Expr | None
    verdict: Verdict
    reason: str = ""


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


class FrameValidator:
    """Adapt a model vector's row schema and causal laws to a DataFrame.

    ``FrameValidator(Ledger, 'rows')`` selects a declared ``model_vector`` and
    its ``model_recurrence`` laws. Each law must use current rows and fixed past
    lags from this vector, covered by its ``start``. The frame supplies the
    window and initial history. All row scalar fields must have columns.

    No model instances are constructed and no solver runs. Every equation uses
    observed predecessors, including after a break. Initial history is assumed
    correct. ``first_invalid`` stops at the first proven violation; ``validate``
    returns an index-aligned nullable boolean Series. ``report`` collects all
    checks. ``iter_checks`` retains at most ``window_size`` rows.
    Independent scalar model parameters can be supplied once in ``parameters``.
    Equation tolerance is ``atol + rtol * abs(expected)``, zero by default.
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
            raise ModelInputError("A frame validator requires a Modello class")
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
        rules = []
        for definition in self._row_model._modello_definition.constraints:
            rules.append(self._compile("definition", definition.path, definition.right, 0))
        for path, symbol in self._fields:
            if path != self._symbols[symbol]:
                rules.append(self._compile("alias", path, symbol, 0))
        laws = [(name, law) for name, law in namespace.model_laws.items() if law.target.path == (rows,)]
        if not laws:
            raise ModelInputError(f"No model_recurrence targets {rows!r}")
        self._start = min(int(law.template.start) for _, law in laws)
        self._complete_start = max(int(law.template.start) for _, law in laws)
        for name, law in laws:
            for path, expression in law.updates:
                body = _substitute(_conditioned(expression, _denominator_condition(expression)), known)
                rules.append(
                    self._compile(name, (*law.target.suffix, *path), body, int(law.template.start), law.template.index)
                )
        self._rules = tuple(rules)
        self._window_size = 1 + max((ref.lag for rule in rules for ref in rule.references), default=0)

    @property
    def window_size(self) -> int:
        """Largest referenced lag plus one, including the current row."""
        return self._window_size

    def first_invalid(self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None) -> FrameCheck | None:
        """Return the first proven violation, stopping without reading later rows.

        ``None`` means no violation was established. Initial history is trusted,
        and missing dependencies can prevent proof: this is not a guarantee that
        every row was validated. Use ``validate`` or ``report`` to see coverage.
        """
        return next((check for check in self.iter_checks(frame, columns=columns) if check.verdict == "violated"), None)

    def validate(self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None) -> pd.Series:
        """Return True, False, or pd.NA at every original index position.

        False means at least one check is violated. True requires all laws to
        have started and all row checks to pass. Assumed history, incomplete law
        coverage, and unresolved checks yield pd.NA unless a violation is known.
        Duplicate and gappy labels, order, and index metadata are preserved.
        """
        states: list[bool | None] = []
        for position, checks in groupby(self.iter_checks(frame, columns=columns), key=lambda check: check.position):
            states.extend([None] * (position - len(states)))
            state: bool | None = True if position >= self._complete_start else None
            for check in checks:
                if check.verdict == "violated":
                    state = False
                elif check.verdict == "unknown" and state is not False:
                    state = None
            states.append(state)
        states.extend([None] * (len(frame) - len(states)))
        return pd.Series(states, index=frame.index.copy(), dtype="boolean", name="valid")

    def report(self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None) -> pd.DataFrame:
        """Collect detailed checks without modifying, sorting, or filling the frame.

        Columns normally use dotted field names or MultiIndex field paths.
        ``columns`` maps model field paths to alternative pandas column labels.
        Extra columns are ignored. Missing columns are an input error; missing
        cells yield unknown checks. Initial history is assumed correct and emits
        no checks until the earliest recurrence starts. Checks are in positional
        order, with domains first, then row definitions, then recurrent equations.
        """
        names = tuple(field.name for field in fields(FrameCheck))
        records = (tuple(getattr(check, name) for name in names) for check in self.iter_checks(frame, columns=columns))
        return pd.DataFrame(records, columns=names, dtype=object)

    def iter_checks(
        self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None
    ) -> Iterator[FrameCheck]:
        """Yield checks in frame order using a bounded window of observations.

        Each call starts a fresh history. Split independent accounts before
        calling; the index labels do not imply grouping or a sampling interval.
        Rows before the earliest recurrence starts are trusted history. Later
        laws retain their own starts; row definitions and domains are checked
        from the earliest start, even while another law still needs history.
        """
        positions = self._bind_columns(frame, columns or {})
        window: deque[tuple[_Observation, ...]] = deque(maxlen=self.window_size)
        for position, (label, row) in enumerate(
            zip(frame.index, frame.itertuples(index=False, name=None), strict=True)
        ):
            observations = tuple(_observe(row[column]) for column in positions)
            window.append(observations)
            if position < self._start:
                continue
            for (path, symbol), observation in zip(self._fields, observations, strict=True):
                yield _domain_check(position, label, path, symbol, observation)
            for rule in self._rules:
                if position >= rule.start:
                    yield self._check(rule, position, label, window)

    def _compile(self, name: str, path: FieldPath, body: Expr, start: int, index: Symbol | None = None) -> _Rule:
        body = _expand_finite_indexed(body)
        condition = _denominator_condition(body)
        replacements = {}
        references = []
        for term in sorted(_scalar_atoms(body, _VectorElement), key=str):
            if index is None or term.vector not in self._columns:
                raise ModelInputError("Frame laws must reference only the selected row vector")
            delay = simplify(index - term.index)
            if not isinstance(delay, Integer) or not 0 <= delay <= start:
                raise ModelInputError(
                    "Frame laws require fixed past lags covered by start; future and fixed rows are unsupported"
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
            raise ModelInputError(f"Unbound frame-law quantities: {names}. Supply scalar parameters or row fields.")
        return _Rule(name, self._positions[path], start, expression, tuple(references), index)

    def _bind_columns(self, frame: pd.DataFrame, columns: Mapping[str, Hashable]) -> tuple[int, ...]:
        if not isinstance(frame, pd.DataFrame):
            raise ModelInputError("Frame validation requires a pandas DataFrame")
        if not frame.columns.is_unique:
            raise ModelInputError("Frame columns must be unique")
        field_names = {_path_label(path): path for path, _ in self._fields}
        field_names.update({".".join(path): path for path, _ in self._fields})
        overrides = {}
        for name, label in columns.items():
            if name not in field_names:
                raise ModelInputError(f"Unknown row field {name!r}")
            overrides[field_names[name]] = label
        paths = {}
        for position, label in enumerate(frame.columns):
            path = field_names.get(label, _column_path(label)) if isinstance(label, str) else _column_path(label)
            if path in self._positions and path not in overrides:
                if path in paths:
                    raise ModelInputError(f"Ambiguous columns for {_path_label(path)}")
                paths[path] = position
        for path, label in overrides.items():
            name = _path_label(path)
            try:
                location = frame.columns.get_loc(label)
            except KeyError as error:
                raise ModelInputError(f"Missing frame column {label!r} for {name}") from error
            if not isinstance(location, int):
                raise ModelInputError(f"Column mapping for {name} must select one complete column label")
            paths[path] = location
        missing = [_path_label(path) for path, _ in self._fields if path not in paths]
        if missing:
            raise ModelInputError(f"Missing frame columns: {', '.join(missing)}")
        return tuple(paths[path] for path, _ in self._fields)

    def _check(
        self, rule: _Rule, position: int, label: Hashable, window: deque[tuple[_Observation, ...]]
    ) -> FrameCheck:
        observation = window[-1][rule.field]
        values = {ref.symbol: window[-1 - ref.lag][ref.field].value for ref in rule.references}
        if rule.index is not None:
            values[rule.index] = Integer(position)
        expected = _evaluate(_substitute(rule.expression, values))
        residual = None
        verdict: Verdict = "unknown"
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
        return FrameCheck(
            position,
            label,
            _path_label(self._fields[rule.field][0]),
            rule.name,
            observation.raw,
            None if expected.free_symbols & unavailable else expected,
            residual,
            verdict,
            reason,
        )


def _observe(value: object) -> _Observation:
    missing = value.is_nan() if isinstance(value, Decimal) else is_scalar(value) and (value is S.NaN or pd.isna(value))
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
    if not isinstance(expression, Expr) or expression.is_Matrix:
        return _Observation(value, Dummy("invalid"), "not a scalar number")
    return _Observation(value, expression)


def _domain_check(position: int, label: Hashable, path: FieldPath, symbol: Symbol, value: _Observation) -> FrameCheck:
    verdict: Verdict = "unknown"
    reason: str = value.problem or "unresolved numeric value"
    if value.problem == "not a scalar number":
        verdict = "violated"
    elif not value.problem and value.value.is_number is True:
        result = check_assumptions(value.value, against=symbol)
        if value.value.is_finite is False or result is False:
            verdict, reason = "violated", "value is outside the field domain"
        elif value.value.is_finite is True and result is True:
            verdict, reason = "satisfied", ""
    return FrameCheck(position, label, _path_label(path), "domain", value.raw, _domain(symbol), None, verdict, reason)


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
            raise ModelInputError("The frame supplies the row vector's size; omit size parameters")
        observation = _observe(raw)
        if observation.problem or observation.value.is_number is not True or observation.value.is_finite is not True:
            raise ModelInputError(f"Parameter {name!r} must be a finite scalar number")
        if check_assumptions(observation.value, against=external[name]) is not True:
            raise ModelInputError(f"Parameter {name!r} violates its field domain")
        known[external[name]] = observation.value
    while pending:
        ready = {
            symbol: _evaluate(_substitute(expression, known))
            for symbol, expression in pending.items()
            if expression.free_symbols <= known.keys() and not _scalar_atoms(expression, _VectorElement)
        }
        ready = {
            symbol: value for symbol, value in ready.items() if value.is_number is True and value.is_finite is True
        }
        if not ready:
            break
        known.update(ready)
        for symbol in ready:
            del pending[symbol]
    return known


def _column_path(label: Hashable) -> FieldPath:
    if isinstance(label, str):
        return tuple(label.split("."))
    if isinstance(label, tuple):
        parts = list(label)
        while parts and (isinstance(parts[-1], str) and parts[-1] == "" or is_scalar(parts[-1]) and pd.isna(parts[-1])):
            parts.pop()
        if all(isinstance(part, (str, int)) for part in parts):
            return tuple(str(part) for part in parts)
    return ()

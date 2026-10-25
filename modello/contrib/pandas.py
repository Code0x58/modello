"""Thin pandas adaptation of the independent rolling-window validator."""

from __future__ import annotations

from collections.abc import Hashable, Iterator, Mapping
from dataclasses import fields
from decimal import Decimal

import pandas as pd
from pandas.api.types import is_scalar

from modello import Modello, ScalarValue
from modello._paths import FieldPath, _path_label
from modello.contrib.validation import (
    CheckResult,
    CheckResult as FrameCheck,
    ValidationSummary,
    WindowResult,
    WindowValidator,
)
from modello.errors import ModelInputError

__all__ = ["FrameCheck", "FrameValidator"]


class FrameValidator:
    """Bind columns, normalise pandas nulls, and check projected row windows.

    ``window`` is the public pandas-independent WindowValidator used by every
    output method. It can also be called directly from an external framework.
    Assumed checks are retained in reports, and audit exposes coverage counts.
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
        self._window = WindowValidator(model, rows, parameters=parameters, atol=atol, rtol=rtol)
        self.model = model
        self.rows = rows

    @classmethod
    def from_window(cls, window: WindowValidator) -> FrameValidator:
        """Adapt an already prepared plan, preserving its parameters and tolerances."""
        if not isinstance(window, WindowValidator):
            raise ModelInputError("from_window requires a WindowValidator")
        instance = cls.__new__(cls)
        instance._window = window
        instance.model = window.model
        instance.rows = window.rows
        return instance

    @property
    def window(self) -> WindowValidator:
        """The public pandas-independent plan used by every adapter operation."""
        return self._window

    @property
    def window_size(self) -> int:
        """The independent validator's required window width, p+1."""
        return self.window.window_size

    @property
    def first_test_position(self) -> int:
        """First absolute test position, normally p rather than p+1."""
        return self.window.first_test_position

    def iter_results(
        self,
        frame: pd.DataFrame,
        *,
        columns: Mapping[str, Hashable] | None = None,
    ) -> Iterator[WindowResult]:
        """Yield one coverage-aware result for every original frame position."""
        positions = self._bind_columns(frame, columns or {})
        rows = (
            tuple(_normalise(row[column]) for column in positions) for row in frame.itertuples(index=False, name=None)
        )
        yield from self.window.iter_results(rows, labels=frame.index)

    def iter_checks(
        self,
        frame: pd.DataFrame,
        *,
        columns: Mapping[str, Hashable] | None = None,
    ) -> Iterator[CheckResult]:
        """Yield every declared check, including assumed history and unknowns."""
        for result in self.iter_results(frame, columns=columns):
            yield from result.checks

    def first_invalid(
        self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None
    ) -> CheckResult | None:
        """Stop at the first proved violation; None alone is not a coverage claim.

        Services requiring a strict decision should use audit or iter_results:
        short histories and missing dependencies can also return None here.
        """
        return next((check for check in self.iter_checks(frame, columns=columns) if check.verdict == "violated"), None)

    def validate(self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None) -> pd.Series:
        """Return aligned nullable booleans; assumptions and unknowns remain pd.NA."""
        states = [result.valid for result in self.iter_results(frame, columns=columns)]
        return pd.Series(states, index=frame.index.copy(), dtype="boolean", name="valid")

    def audit(self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None) -> ValidationSummary:
        """Scan every row and return explicit coverage with the first proved violation.

        summary.valid is strict: assumed or unresolved rows prevent True, even
        if all tested transitions passed. Inspect counts to apply a deliberate
        policy for trusted initial history; an empty scan is never complete.
        """
        return ValidationSummary.from_results(self.iter_results(frame, columns=columns))

    def report(self, frame: pd.DataFrame, *, columns: Mapping[str, Hashable] | None = None) -> pd.DataFrame:
        """Return checks with explicit assumed verdicts, kind, and first test position.

        Missing columns are configuration errors. Missing cells are normalised
        to None before the common window function, and can produce unknowns.
        """
        names = tuple(field.name for field in fields(CheckResult))
        records = (tuple(getattr(check, name) for name in names) for check in self.iter_checks(frame, columns=columns))
        return pd.DataFrame(records, columns=names, dtype=object)

    def _bind_columns(self, frame: pd.DataFrame, columns: Mapping[str, Hashable]) -> tuple[int, ...]:
        if not isinstance(frame, pd.DataFrame):
            raise ModelInputError("Frame validation requires a pandas DataFrame")
        if not frame.columns.is_unique:
            raise ModelInputError("Frame columns must be unique")
        field_positions = {path: i for i, path in enumerate(self.window.field_paths)}
        field_names = {_path_label(path): path for path in self.window.field_paths}
        field_names.update({".".join(path): path for path in self.window.field_paths})
        overrides = {}
        for name, label in columns.items():
            if name not in field_names:
                raise ModelInputError(f"Unknown row field {name!r}")
            overrides[field_names[name]] = label
        paths = {}
        for position, label in enumerate(frame.columns):
            path = field_names.get(label, _column_path(label)) if isinstance(label, str) else _column_path(label)
            if path in field_positions and path not in overrides:
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
        missing = [_path_label(path) for path in self.window.field_paths if path not in paths]
        if missing:
            raise ModelInputError(f"Missing frame columns: {', '.join(missing)}")
        return tuple(paths[path] for path in self.window.field_paths)


def _normalise(value: object) -> object:
    if isinstance(value, Decimal):
        return None if value.is_nan() else value
    return None if is_scalar(value) and pd.isna(value) else value


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

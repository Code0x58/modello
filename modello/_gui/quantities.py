"""A model quantity together with its display units, labels, and exact values."""

from __future__ import annotations

import math
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any

from sympy import Expr, FiniteSet, Interval, Range, S, Set, Symbol, Union

from modello._gui.formatting import display_number
from modello.diagnostics import FieldValue, ModelField, ModelProblem

if TYPE_CHECKING:
    from modello import Modello
    from modello.gui import FieldInfo


def is_finite_real(value: Expr) -> bool:
    return value.is_number is True and value.is_real is True and value.is_finite is True


@dataclass(frozen=True)
class Quantity:
    """Keep exact model values and their presentation scale in one place."""

    field: ModelField
    info: FieldInfo
    problem: ModelProblem

    @property
    def path(self) -> str:
        return ".".join(self.field.path)

    @property
    def symbol(self) -> Symbol:
        return self.field.symbol

    def read(self, instance: Modello) -> Expr:
        return FieldValue(self.field, attrgetter(self.path)(instance)).value

    def scale(self, value: Expr) -> Expr:
        return value * 100 if self.info.number_format == "percent" else value

    def input(self, value: Expr) -> str:
        return self.problem.format(self.scale(value)) + ("%" if self.info.number_format == "percent" else "")

    def number(self, value: Expr) -> float | None:
        scaled = self.scale(value)
        if not is_finite_real(scaled):
            return None
        number = float(scaled.evalf(16))
        return number if math.isfinite(number) else None

    def present(self, value: Expr) -> dict[str, Any]:
        display, approximate = display_number(self.scale(value), percent=self.info.number_format == "percent")
        return {
            "value": self.problem.format(value),
            "display_value": display or self.problem.format(self.scale(value)),
            "approximate": approximate,
            "number": self.number(value),
        }

    def same_units(self, other: Quantity) -> bool:
        return bool(self.info.unit) and (self.info.unit, self.info.number_format) == (
            other.info.unit,
            other.info.number_format,
        )

    def describe(self, values: Set) -> dict[str, str]:
        """Describe the possible values without turning an implicit set into an interval."""

        def scalar(value: Expr) -> str:
            view = self.present(value)
            return ("≈" if view["approximate"] else "") + view["display_value"]

        if values.is_empty is True:
            kind, text = "empty", "No feasible value"
        elif isinstance(values, Interval):
            left, right = ("(" if values.left_open else "["), (")" if values.right_open else "]")
            kind, text = "interval", f"{left}{scalar(values.start)}, {scalar(values.end)}{right}"
        elif isinstance(values, FiniteSet):
            kind = "point" if len(values) == 1 else "finite"
            entries = [scalar(value) for value in sorted(values, key=str)[:6]]
            text = ", ".join(entries) + (", …" if len(values) > 6 else "")
            if kind != "point":
                text = "{" + text + "}"
        elif isinstance(values, Range) and values.size.is_Integer:
            if values.size <= 6:
                return self.describe(FiniteSet(*values))
            kind = "finite"
            text = f"{scalar(values[0])}, {scalar(values[1])}, …, {scalar(values[-1])} ({values.size} values)"
        elif isinstance(values, Union):
            parts = [self.describe(value) for value in values.args]
            kind = "union" if all(part["kind"] != "implicit" for part in parts) else "implicit"
            text = " ∪ ".join(part["text"] for part in parts) if kind == "union" else "Conditional set"
        elif values in (S.Reals, S.Integers, S.Complexes):
            kind, text = (
                "domain",
                {S.Reals: "Real numbers", S.Integers: "Integers", S.Complexes: "Complex numbers"}[values],
            )
        else:
            kind, text = "implicit", "Conditional set"
        return {"kind": kind, "text": text}

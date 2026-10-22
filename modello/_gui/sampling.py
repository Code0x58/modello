"""Coordinates, display windows, and independently checked samples of a family."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from sympy import ConditionSet, Eq, Expr, Interval, ProductSet, S, Symbol

from modello import Modello, ScalarValue, SolutionFamily
from modello._gui.quantities import Quantity, is_finite_real
from modello.diagnostics import Candidate, FieldValue
from modello.errors import ModelInputError, SolutionSelectionError, UnresolvedSolutionError

if TYPE_CHECKING:
    from modello._gui.presentation import Layout


@dataclass(frozen=True)
class Coordinates:
    """The remaining model quantities, including coupled and background restrictions."""

    family: SolutionFamily
    layout: Layout

    @cached_property
    def quantities(self) -> tuple[Quantity, ...]:
        return tuple(self.layout.quantity(".".join(field.path)) for field in self.family.parameters)

    @cached_property
    def symbols(self) -> frozenset[Symbol]:
        return frozenset(field.symbol for field in self.family.parameters)

    def check_point(self, point: Mapping[str, ScalarValue]) -> Modello:
        assignments: dict[Symbol, Expr] = {}
        for path, value in point.items():
            quantity = self.layout.quantity(path, context="point")
            if path in self.layout.hidden or quantity.symbol not in self.symbols:
                raise ModelInputError(f"point: {path!r} must be a visible remaining coordinate")
            parsed = FieldValue(quantity.field, value).value
            if quantity.symbol in assignments and assignments[quantity.symbol] != parsed:
                raise ModelInputError("Coordinate aliases must have the same value")
            assignments[quantity.symbol] = parsed
        if assignments.keys() != self.symbols:
            raise ModelInputError("Preview a value for every remaining coordinate")
        try:
            return self.family.where(*(Eq(symbol, value) for symbol, value in assignments.items())).one()
        except (SolutionSelectionError, UnresolvedSolutionError) as error:
            raise ModelInputError(
                "This point could not be checked against every equation, domain, and filter"
            ) from error

    def controls(self, paths: Sequence[str]) -> tuple[Quantity, ...]:
        if not 1 <= len(paths) <= 2 or len(set(paths)) != len(paths):
            raise ModelInputError("Choose one or two distinct remaining coordinates to plot")
        if any(all(path in self.layout.hidden for path in self.layout.aliases[symbol]) for symbol in self.symbols):
            raise ModelInputError("A background coordinate remains; it cannot be varied or replaced in the browser")
        controls = tuple(self.layout.quantity(path, context="controls") for path in paths)
        if len({quantity.symbol for quantity in controls}) != len(controls):
            raise ModelInputError("Choose distinct coordinates rather than two aliases of one quantity")
        for quantity in controls:
            if (
                quantity.path in self.layout.hidden
                or quantity.symbol not in self.symbols
                or quantity.symbol.is_real is not True
            ):
                raise ModelInputError(f"controls: {quantity.path!r} must be a visible, editable real coordinate")
        return controls

    def section(self, controls: Sequence[Quantity], reference: Modello | None) -> SolutionFamily:
        if len(controls) == len(self.family.parameters):
            return self.family
        if reference is None:
            raise ModelInputError("Preview a checked reference case before fixing the omitted coordinates")
        return self.family.section(*(quantity.symbol for quantity in controls), through=reference)

    def encode(self, assignment: Mapping[Symbol, Expr]) -> dict[str, str]:
        return {quantity.path: self.family.problem.format(assignment[quantity.symbol]) for quantity in self.quantities}

    def summary(self, reference: Modello | None) -> list[dict[str, Any]]:
        domain = self.family.parameter_domain
        while isinstance(domain, ConditionSet):
            domain = domain.base_set
        factors = domain.args if isinstance(domain, ProductSet) else (domain,)
        windows = (
            dict(zip((field.symbol for field in self.family.parameters), factors))
            if len(factors) == len(self.family.parameters)
            else {}
        )
        result = []
        for field in self.family.parameters:
            path = next((p for p in self.layout.aliases[field.symbol] if p not in self.layout.hidden), None)
            if path is None:
                continue
            quantity = self.layout.quantity(path)
            window = windows.get(field.symbol)
            display_window = ["", ""]
            if isinstance(window, Interval) and window.start.is_finite is True and window.end.is_finite is True:
                display_window = [quantity.input(window.start), quantity.input(window.end)]
            result.append(
                {
                    "path": path,
                    "title": quantity.info.title or path,
                    "previewable": True,
                    "real": field.symbol.is_real is True,
                    "window": display_window,
                    "point_value": quantity.input(quantity.read(reference)) if reference is not None else "",
                }
            )
        return result


@dataclass(frozen=True)
class PlotWindow:
    """An exact sampling interval with a finite extent in the quantity's display units."""

    lower: Expr
    upper: Expr
    display: tuple[float, float]

    @classmethod
    def for_quantity(cls, quantity: Quantity, endpoints: Sequence[ScalarValue]) -> PlotWindow:
        if len(endpoints) != 2:
            raise ModelInputError(f"windows: supply a lower and upper plotting bound for {quantity.path!r}")
        lower, upper = (FieldValue(quantity.field, value).value for value in endpoints)
        if not all(is_finite_real(value) for value in (lower, upper)):
            raise ModelInputError("Plotting windows must use finite real numeric bounds")
        if (lower < upper) is not S.true:
            raise ModelInputError("Each plotting lower bound must be less than its upper bound")
        if not all(math.isfinite(float(value)) for value in (lower, upper)):
            raise ModelInputError("The plotting window exceeds the numerical display range")
        start, end = quantity.number(lower), quantity.number(upper)
        if start is None or end is None:
            raise ModelInputError("The plotting window exceeds the numerical range of the displayed units")
        if not start < end or not math.isfinite(end - start):
            raise ModelInputError("The plotting window needs distinct finite bounds in the displayed units")
        return cls(lower, upper, (start, end))

    def samples(self, count: int) -> tuple[Expr, ...]:
        return tuple(self.lower + (self.upper - self.lower) * S(index) / (count - 1) for index in range(count))


class FamilyPlot:
    """A bounded view of a family or a section through a checked reference."""

    def __init__(
        self,
        coordinates: Coordinates,
        controls: Sequence[str],
        output: str,
        windows: Mapping[str, Sequence[ScalarValue]],
        reference: Modello | None,
    ) -> None:
        self.coordinates = coordinates
        self.controls = coordinates.controls(controls)
        self.output = coordinates.layout.quantity(output, context="output")
        if output in coordinates.layout.hidden:
            raise ModelInputError("Choose a visible output")
        self.family = coordinates.section(self.controls, reference)
        self.is_section = len(self.controls) < len(coordinates.family.parameters)
        self.windows = tuple(
            PlotWindow.for_quantity(quantity, windows.get(quantity.path, ())) for quantity in self.controls
        )
        self.grid_size = 41 if len(self.controls) == 1 else 21
        self.fixed = {
            quantity.path: quantity.present(quantity.read(reference))
            for quantity in coordinates.quantities
            if self.is_section and reference is not None and quantity.symbol not in {q.symbol for q in self.controls}
        }

    def sample(self) -> dict[str, Any]:
        expressions = {field.symbol: self.family.expression(field.symbol) for field in self.family.problem.fields}
        points = []
        rejected = unknown = 0
        grid = product(*(window.samples(self.grid_size) for window in self.windows))
        for index, values in enumerate(grid):
            substitutions = dict(zip((quantity.symbol for quantity in self.controls), values))
            assignment = {symbol: expression.xreplace(substitutions) for symbol, expression in expressions.items()}
            candidate = Candidate(
                tuple(FieldValue(field, assignment[field.symbol]) for field in self.family.problem.fields)
            )
            _, verdict = self.family._check_state(candidate)
            if verdict != "satisfied":
                rejected += verdict == "violated"
                unknown += verdict == "unknown"
                continue
            axes = [quantity.number(assignment[quantity.symbol]) for quantity in (*self.controls, self.output)]
            if any(value is None for value in axes):
                unknown += 1
                continue
            points.append({"index": index, "axes": axes, "point": self.coordinates.encode(assignment)})
        return {
            "controls": [quantity.path for quantity in self.controls],
            "output": self.output.path,
            "section": self.is_section,
            "points": points,
            "rejected": rejected,
            "unknown": unknown,
            "samples": self.grid_size ** len(self.controls),
            "grid_size": self.grid_size,
            "windows": [list(window.display) for window in self.windows],
            "fixed": self.fixed,
        }

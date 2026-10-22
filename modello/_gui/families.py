"""Conservative family views and bounded, independently audited chart samples."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import product
from operator import attrgetter
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from sympy import ConditionSet, Eq, Expr, FiniteSet, Interval, ProductSet, Range, S, Set, Symbol, Union, diff

from modello import Modello, ScalarValue, SolutionFamily
from modello._gui.formatting import display_number
from modello.diagnostics import Candidate, FieldValue, ModelField
from modello.errors import ModelInputError, SolutionSelectionError, UnresolvedSolutionError

if TYPE_CHECKING:
    from modello._gui.presentation import Layout

MAX_CASES = 32


class FamilyView:
    """Keep a family and an optional checked point together, without observing it."""

    def __init__(self, family: SolutionFamily, layout: Layout) -> None:
        domain = family.parameter_domain
        self.finite_count = (
            len(domain)
            if isinstance(domain, FiniteSet)
            else (int(domain.size) if isinstance(domain, Range) and domain.size.is_Integer else None)
        )
        if isinstance(domain, Range) and self.finite_count is not None and self.finite_count <= MAX_CASES:
            family = replace(family, parameter_domain=FiniteSet(*domain))
        self.family = family
        self.layout = layout
        self.fields: dict[Symbol, ModelField] = {}
        for path in (*layout.model_info.highlights, layout.model_info.initial_field, *layout.fields):
            if path in layout.fields and path not in layout.hidden:
                field = layout.fields[path]
                self.fields.setdefault(field.symbol, field)
        self.cases: tuple[Modello, ...] = ()
        self.reference: Modello | None = None
        self.selection: dict[str, Any] | None = None
        self.reason = ""
        if isinstance(family.parameter_domain, FiniteSet) and len(family.parameter_domain) <= MAX_CASES:
            try:
                self.cases = family.cases()
            except UnresolvedSolutionError as error:
                self.reason = error.reason
        if len(self.cases) == 1:
            self.select(case=0)

    def select(self, *, case: int | None = None, point: Mapping[str, ScalarValue] | None = None) -> None:
        if case is not None and point is not None:
            raise ModelInputError("Select a case or preview a coordinate point, not both")
        if case is not None:
            if isinstance(case, bool) or not isinstance(case, int) or not 0 <= case < len(self.cases):
                raise ModelInputError("Choose a case from the current checked alternatives")
            self.reference = self.cases[case]
            self.selection = {"kind": "case", "index": case}
        elif point is not None:
            coordinates = {field.symbol for field in self.family.parameters}
            assignments: dict[Symbol, Expr] = {}
            for path, value in point.items():
                field = self.layout.field(path, context="point")
                if path in self.layout.hidden or field.symbol not in coordinates:
                    raise ModelInputError(f"point: {path!r} must be a visible remaining coordinate")
                parsed = FieldValue(field, value).value
                if field.symbol in assignments and assignments[field.symbol] != parsed:
                    raise ModelInputError("Coordinate aliases must have the same value")
                assignments[field.symbol] = parsed
            if assignments.keys() != coordinates:
                raise ModelInputError("Preview a value for every remaining coordinate")
            try:
                self.reference = self.family.where(*(Eq(symbol, value) for symbol, value in assignments.items())).one()
            except (SolutionSelectionError, UnresolvedSolutionError) as error:
                raise ModelInputError(
                    "This point could not be checked against every equation, domain, and filter"
                ) from error
            self.selection = {
                "kind": "point",
                "coordinates": {
                    path: self.family.problem.format(FieldValue(self.layout.field(path), value).value)
                    for path, value in point.items()
                },
            }

    def assignment(self, instance: Modello) -> dict[Symbol, Expr]:
        return {
            field.symbol: FieldValue(field, attrgetter(".".join(field.path))(instance)).value
            for field in self.family.problem.fields
        }

    def values(self) -> dict[Symbol, Expr]:
        if self.reference is not None:
            return self.assignment(self.reference)
        return {field.symbol: self.family.expression(field.symbol) for field in self.family.problem.fields}

    def scalar(self, field: ModelField, expression: Expr) -> dict[str, Any]:
        path = ".".join(field.path)
        info = self.layout.metadata[path]
        scaled = expression * 100 if info.number_format == "percent" else expression
        display, approximate = display_number(scaled, percent=info.number_format == "percent")
        numeric = None
        if scaled.is_number is True and scaled.is_real is True and scaled.is_finite is True:
            value = float(scaled.evalf(16))
            numeric = value if math.isfinite(value) else None
        return {
            "value": self.family.problem.format(expression),
            "display_value": display or self.family.problem.format(scaled),
            "approximate": approximate,
            "number": numeric,
        }

    def possible(self, field: ModelField) -> dict[str, str]:
        values = self.family.values(field.symbol)
        return self.describe_set(field, values)

    def input_text(self, path: str, expression: Expr) -> str:
        if self.layout.metadata[path].number_format == "percent":
            return self.family.problem.format(expression * 100) + "%"
        return self.family.problem.format(expression)

    def describe_set(self, field: ModelField, values: Set) -> dict[str, str]:
        def scalar(expression: Expr) -> str:
            view = self.scalar(field, expression)
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
                return self.describe_set(field, FiniteSet(*values))
            kind = "finite"
            text = f"{scalar(values[0])}, {scalar(values[1])}, …, {scalar(values[-1])} ({values.size} values)"
        elif isinstance(values, Union):
            parts = [self.describe_set(field, value) for value in values.args]
            kind = "union" if all(part["kind"] != "implicit" for part in parts) else "implicit"
            text = " ∪ ".join(part["text"] for part in parts) if kind == "union" else "Conditional set"
        elif values in (S.Reals, S.Integers, S.Complexes):
            kind, text = (
                "domain",
                {S.Reals: "Real numbers", S.Integers: "Integers", S.Complexes: "Complex numbers"}[values],
            )
        else:
            kind, text = "implicit", "Conditional set"
        return {"kind": kind, "text": text, "exact": self.family.format(values)}

    def _windows(self) -> dict[Symbol, Interval]:
        domain = self.family.parameter_domain
        while isinstance(domain, ConditionSet):
            domain = domain.base_set
        factors = domain.args if isinstance(domain, ProductSet) else (domain,)
        if len(factors) != len(self.family.parameters):
            return {}
        return {
            field.symbol: factor
            for field, factor in zip(self.family.parameters, factors)
            if isinstance(factor, Interval) and factor.start.is_finite is True and factor.end.is_finite is True
        }

    def summary(self) -> dict[str, Any]:
        family = self.family
        count = len(family.parameters)
        kind = "curve" if count == 1 else "region" if count == 2 else "section" if count > 2 else "conditional"
        if family.is_empty is True:
            kind = "empty"
        elif self.cases:
            kind = "point" if len(self.cases) == 1 else "cases"
        elif self.finite_count is not None:
            kind = "finite" if not self.reason else "conditional"
        windows = self._windows()
        reference = self.assignment(self.reference) if self.reference is not None else {}
        coordinates = []
        for field in family.parameters:
            path = next((p for p in self.layout.aliases[field.symbol] if p not in self.layout.hidden), None)
            if path is None:
                continue
            window = windows.get(field.symbol)
            coordinates.append(
                {
                    "path": path,
                    "title": self.layout.metadata[path].title or path,
                    "previewable": path not in self.layout.hidden,
                    "real": field.symbol.is_real is True,
                    "window": [self.input_text(path, window.start), self.input_text(path, window.end)]
                    if window is not None
                    else ["", ""],
                    "point_value": self.input_text(path, reference[field.symbol]) if reference else "",
                }
            )
        canonical = list(self.fields.values())
        assignments = [self.assignment(case) for case in self.cases]
        distinguishing = [
            field
            for field in canonical
            if len(FiniteSet(*(assignment[field.symbol] for assignment in assignments))) > 1
        ]
        # Prefer highlighted, measurable outputs over coordinates or duplicate aliases.
        highlight_order = {path: index for index, path in enumerate(self.layout.model_info.highlights)}
        ordered = sorted(
            distinguishing,
            key=lambda f: (
                highlight_order.get(".".join(f.path), len(highlight_order)),
                f in family.parameters,
                ".".join(f.path),
            ),
        )
        partitions: set[tuple[int, ...]] = set()
        suggestions = []
        for field in ordered:
            if ".".join(field.path) in self.layout.hidden:
                continue
            values = [assignment[field.symbol] for assignment in assignments]
            partition = tuple(values.index(value) for value in values)
            if partition in partitions:
                continue
            partitions.add(partition)
            suggestions.append(".".join(field.path))
        coordinate_symbols = {field.symbol for field in family.parameters}
        invariants = [
            ".".join(field.path)
            for field in canonical
            if kind not in {"empty", "point"}
            and (
                (
                    not family.expression(field.symbol).free_symbols & coordinate_symbols
                    and family.expression(field.symbol).is_number is True
                )
                or (
                    bool(assignments) and len(FiniteSet(*(assignment[field.symbol] for assignment in assignments))) == 1
                )
            )
            and not any(c.left == field.symbol and c.kind != "definition" for c in family.problem.constraints)
        ]
        hidden_symbols = {self.layout.fields[p].symbol for p in self.layout.hidden if p in self.layout.fields}
        conditions = [family.problem.format(c) for c in family.conditions if not c.free_symbols & hidden_symbols]
        # Only constant exact derivatives support a claim across the whole coordinate expression.
        # Matching display units avoid implying conversions or comparing incompatible scales.
        relations = []
        if kind == "curve":
            coordinate = family.parameters[0].symbol
            rates = {field.symbol: diff(family.expression(field.symbol), coordinate) for field in canonical}
            output_path = self.layout.model_info.initial_field
            output = self.layout.fields.get(output_path)
            if output is not None:
                output_info = self.layout.metadata[output_path]
                rate = rates.get(output.symbol)
                if rate is not None and rate.is_number is True and rate.is_real is True and rate.is_finite is True:
                    for field in canonical:
                        source = ".".join(field.path)
                        source_info = self.layout.metadata[source]
                        source_rate = rates[field.symbol]
                        if (
                            field.symbol != output.symbol
                            and output_info.unit
                            and source_info.unit == output_info.unit
                            and source_info.number_format == output_info.number_format
                            and source_rate.is_number is True
                            and source_rate.is_real is True
                            and source_rate.is_finite is True
                            and source_rate.is_zero is False
                        ):
                            ratio = rate / source_rate
                            display, approximate = display_number(ratio)
                            relations.append(
                                {
                                    "output": output_path,
                                    "source": source,
                                    "ratio": str(ratio),
                                    "display_value": display,
                                    "approximate": approximate,
                                }
                            )
                            if len(relations) == 2:
                                break
        violations: set[str] = set()
        if kind == "empty":
            audit = family.problem.check(
                Candidate(tuple(FieldValue(field, family.expression(field.symbol)) for field in family.problem.fields))
            )
            violations.update(
                ".".join(check.value.field.path)
                for check in audit.assumptions
                if check.actual not in (None, check.expected)
            )
            violations.update(
                ".".join(check.constraint.path) for check in audit.residuals if check.verdict == "violated"
            )
        return {
            "kind": kind,
            "coordinate_count": count,
            "coordinates": coordinates,
            "case_count": len(self.cases) if self.cases or kind == "empty" else None,
            "candidate_count": self.finite_count,
            "case_limit": MAX_CASES,
            "cases": [
                {
                    "index": index,
                    "values": {
                        ".".join(field.path): self.scalar(field, assignment[field.symbol]) for field in canonical
                    },
                }
                for index, assignment in enumerate(assignments)
            ],
            "selection": self.selection,
            "distinguishing": [".".join(field.path) for field in distinguishing],
            "suggestions": suggestions[:3],
            "invariants": invariants,
            "conditions": conditions,
            "background_conditions": len(family.conditions) - len(conditions),
            "reason": self.reason,
            "issues": [f"{issue.operation}: {type(issue.error).__name__}" for issue in family.issues],
            "violations": sorted(path for path in violations if path not in self.layout.hidden),
            "relations": relations,
        }

    def plot(
        self, controls: Sequence[str], output: str, windows: Mapping[str, Sequence[ScalarValue]]
    ) -> dict[str, Any]:
        """Sample real coordinates inside a display window, retaining gaps and unknowns."""
        if not 1 <= len(controls) <= 2 or len(set(controls)) != len(controls):
            raise ModelInputError("Choose one or two distinct remaining coordinates to plot")
        coordinate_symbols = {field.symbol for field in self.family.parameters}
        if any(
            all(path in self.layout.hidden for path in self.layout.aliases[symbol]) for symbol in coordinate_symbols
        ):
            raise ModelInputError("A background coordinate remains; it cannot be varied or replaced in the browser")
        fields = [self.layout.field(path, context="controls") for path in controls]
        if len({field.symbol for field in fields}) != len(fields):
            raise ModelInputError("Choose distinct coordinates rather than two aliases of one quantity")
        for path, field in zip(controls, fields):
            if path in self.layout.hidden or field.symbol not in coordinate_symbols or field.symbol.is_real is not True:
                raise ModelInputError(f"controls: {path!r} must be a visible, editable real coordinate")
        output_field = self.layout.field(output, context="output")
        if output in self.layout.hidden:
            raise ModelInputError("Choose a visible output")
        family = self.family
        section = len(fields) < len(family.parameters)
        if section:
            if self.reference is None:
                raise ModelInputError("Preview a checked reference case before fixing the omitted coordinates")
            family = family.section(*(field.symbol for field in fields), through=self.reference)
        grids = []
        window_axes = []
        samples = 41 if len(fields) == 1 else 11
        for path, field in zip(controls, fields):
            endpoints = windows.get(path, ())
            if len(endpoints) != 2:
                raise ModelInputError(f"windows: supply a lower and upper plotting bound for {path!r}")
            lower, upper = (FieldValue(field, value).value for value in endpoints)
            if any(v.is_real is not True or v.is_finite is not True or v.is_number is not True for v in (lower, upper)):
                raise ModelInputError("Plotting windows must use finite real numeric bounds")
            if (lower < upper) is not S.true:
                raise ModelInputError("Each plotting lower bound must be less than its upper bound")
            if not all(math.isfinite(float(v)) for v in (lower, upper)):
                raise ModelInputError("The plotting window exceeds the numerical display range")
            grids.append([lower + (upper - lower) * S(i) / (samples - 1) for i in range(samples)])
            window_axes.append([self.scalar(field, value)["number"] for value in (lower, upper)])
            if any(value is None for value in window_axes[-1]):
                raise ModelInputError("The plotting window exceeds the numerical range of the displayed units")
            start, end = window_axes[-1]
            if not start < end or not math.isfinite(end - start):
                raise ModelInputError("The plotting window needs distinct finite bounds in the displayed units")
        expressions = {field.symbol: family.expression(field.symbol) for field in family.problem.fields}
        result = []
        rejected = unknown = 0
        reference = self.assignment(self.reference) if self.reference is not None else {}
        for index, coordinates in enumerate(product(*grids)):
            substitutions = dict(zip((field.symbol for field in fields), coordinates))
            values = {symbol: expression.xreplace(substitutions) for symbol, expression in expressions.items()}
            candidate = Candidate(tuple(FieldValue(field, values[field.symbol]) for field in family.problem.fields))
            _, verdict = family._check_state(candidate)
            if verdict != "satisfied":
                rejected += verdict == "violated"
                unknown += verdict == "unknown"
                continue
            plotted = [self.scalar(field, values[field.symbol]) for field in (*fields, output_field)]
            if any(value["number"] is None for value in plotted):
                unknown += 1
                continue
            # The full coordinate assignment lets a chart point reopen in the original family.
            point = {
                ".".join(field.path): family.problem.format(values[field.symbol]) for field in self.family.parameters
            }
            result.append({"index": index, "axes": [value["number"] for value in plotted], "point": point})
        return {
            "controls": list(controls),
            "output": output,
            "section": section,
            "points": result,
            "rejected": rejected,
            "unknown": unknown,
            "samples": samples ** len(fields),
            "windows": window_axes,
            "fixed": {
                ".".join(field.path): self.scalar(field, reference[field.symbol])
                for field in self.family.parameters
                if field.symbol not in {f.symbol for f in fields} and section
            },
        }

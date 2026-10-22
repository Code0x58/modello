"""Solution-family presentation: checked alternatives, previews, and insights."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from sympy import Expr, FiniteSet, Range, Symbol, diff

from modello import Modello, ScalarValue, SolutionFamily
from modello._gui.formatting import display_number
from modello._gui.quantities import Quantity, is_finite_real
from modello._gui.sampling import Coordinates, FamilyPlot
from modello.diagnostics import Candidate, ConstraintKind, FieldValue, ModelField
from modello.errors import ModelInputError, UnresolvedSolutionError

if TYPE_CHECKING:
    from modello._gui.presentation import Layout

MAX_CASES = 32


class CheckedCases:
    """Small audited alternatives and the observations that distinguish them."""

    def __init__(self, family: SolutionFamily, quantities: tuple[Quantity, ...]) -> None:
        domain = family.parameter_domain
        self.candidate_count = (
            len(domain)
            if isinstance(domain, FiniteSet)
            else int(domain.size)
            if isinstance(domain, Range) and domain.size.is_Integer
            else None
        )
        if isinstance(domain, Range) and self.candidate_count is not None and self.candidate_count <= MAX_CASES:
            family = replace(family, parameter_domain=FiniteSet(*domain))
        self.family = family
        self.quantities = quantities
        self.models: tuple[Modello, ...] = ()
        self.reason = ""
        if isinstance(family.parameter_domain, FiniteSet) and len(family.parameter_domain) <= MAX_CASES:
            try:
                self.models = family.cases()
            except UnresolvedSolutionError as error:
                self.reason = error.reason
        self.assignments = tuple({q.symbol: q.read(model) for q in quantities} for model in self.models)
        self.possibilities = {
            quantity.symbol: FiniteSet(*(assignment[quantity.symbol] for assignment in self.assignments))
            for quantity in quantities
        }

    def values(self, quantity: Quantity) -> FiniteSet:
        return self.possibilities[quantity.symbol]

    @property
    def distinguishing(self) -> tuple[Quantity, ...]:
        return tuple(quantity for quantity in self.quantities if len(self.values(quantity)) > 1)

    def suggestions(self, highlights: Sequence[str]) -> list[str]:
        priority = {path: index for index, path in enumerate(highlights)}
        ordered = sorted(
            self.distinguishing,
            key=lambda q: (priority.get(q.path, len(priority)), q.field in self.family.parameters, q.path),
        )
        partitions: set[tuple[int, ...]] = set()
        suggestions = []
        for quantity in ordered:
            values = [assignment[quantity.symbol] for assignment in self.assignments]
            partition = tuple(values.index(value) for value in values)
            if partition not in partitions:
                partitions.add(partition)
                suggestions.append(quantity.path)
        return suggestions[:3]

    def summary(self) -> list[dict[str, Any]]:
        return [
            {"index": index, "values": {q.path: q.present(assignment[q.symbol]) for q in self.quantities}}
            for index, assignment in enumerate(self.assignments)
        ]


class FamilyView:
    """Keep the whole family, its checked alternatives, and a separate point preview."""

    def __init__(self, family: SolutionFamily, layout: Layout) -> None:
        self.layout = layout
        self.quantities = layout.visible_quantities()
        self.alternatives = CheckedCases(family, self.quantities)
        self.family = self.alternatives.family
        self.coordinates = Coordinates(self.family, layout)
        self.reference: Modello | None = None
        self.selection: dict[str, Any] | None = None
        if len(self.cases) == 1:
            self.select(case=0)

    @property
    def cases(self) -> tuple[Modello, ...]:
        return self.alternatives.models

    @property
    def kind(self) -> str:
        if self.family.is_empty is True:
            return "empty"
        if self.cases:
            return "point" if len(self.cases) == 1 else "cases"
        if self.alternatives.candidate_count is not None:
            return "conditional" if self.alternatives.reason else "finite"
        count = len(self.family.parameters)
        return {1: "curve", 2: "region"}.get(count, "section" if count > 2 else "conditional")

    @property
    def verdict(self) -> str:
        if self.family.is_empty is True:
            return "empty"
        if self.reference is not None:
            return "satisfied"
        return "alternatives" if self.cases else "unknown"

    def select(self, *, case: int | None = None, point: Mapping[str, ScalarValue] | None = None) -> None:
        if case is not None and point is not None:
            raise ModelInputError("Select a case or preview a coordinate point, not both")
        if case is not None:
            if isinstance(case, bool) or not isinstance(case, int) or not 0 <= case < len(self.cases):
                raise ModelInputError("Choose a case from the current checked alternatives")
            self.reference = self.cases[case]
            self.selection = {"kind": "case", "index": case}
        elif point is not None:
            self.reference = self.coordinates.check_point(point)
            self.selection = {
                "kind": "point",
                "coordinates": {
                    path: self.family.problem.format(FieldValue(self.layout.field(path), value).value)
                    for path, value in point.items()
                },
            }

    def values(self) -> dict[Symbol, Expr]:
        return {
            field.symbol: self.layout.quantity(".".join(field.path)).read(self.reference)
            if self.reference is not None
            else self.family.expression(field.symbol)
            for field in self.family.problem.fields
        }

    def possible(self, field: ModelField) -> dict[str, str]:
        values = self.family.values(field.symbol)
        return {**self.layout.quantity(".".join(field.path)).describe(values), "exact": self.family.format(values)}

    @property
    def invariants(self) -> list[str]:
        if self.kind in {"empty", "point"}:
            return []
        observed = {c.left for c in self.family.problem.constraints if c.kind is not ConstraintKind.DEFINITION}
        invariants = []
        for quantity in self.quantities:
            expression = self.family.expression(quantity.symbol)
            constant = not expression.free_symbols & self.coordinates.symbols and expression.is_number is True
            identical = bool(self.cases) and len(self.alternatives.values(quantity)) == 1
            if quantity.symbol not in observed and (constant or identical):
                invariants.append(quantity.path)
        return invariants

    @property
    def relations(self) -> list[dict[str, Any]]:
        """Only constant exact responses with matching display units support an insight."""
        output_path = self.layout.model_info.initial_field
        if self.kind != "curve" or output_path not in self.layout.fields:
            return []
        output = self.layout.quantity(output_path)
        coordinate = self.family.parameters[0].symbol
        rates = {q.symbol: diff(self.family.expression(q.symbol), coordinate) for q in self.quantities}
        rate = rates.get(output.symbol)
        if rate is None or not is_finite_real(rate):
            return []
        relations = []
        for source in self.quantities:
            source_rate = rates[source.symbol]
            if (
                source.symbol == output.symbol
                or not source.same_units(output)
                or not is_finite_real(source_rate)
                or source_rate.is_zero is not False
            ):
                continue
            ratio = rate / source_rate
            display, approximate = display_number(ratio)
            relations.append(
                {
                    "output": output_path,
                    "source": source.path,
                    "ratio": str(ratio),
                    "display_value": display,
                    "approximate": approximate,
                }
            )
            if len(relations) == 2:
                break
        return relations

    @property
    def conditions(self) -> list[str]:
        hidden = {self.layout.fields[path].symbol for path in self.layout.hidden if path in self.layout.fields}
        return [
            self.family.problem.format(condition)
            for condition in self.family.conditions
            if not condition.free_symbols & hidden
        ]

    @property
    def violations(self) -> list[str]:
        if self.kind != "empty":
            return []
        candidate = Candidate(
            tuple(FieldValue(field, self.family.expression(field.symbol)) for field in self.family.problem.fields)
        )
        audit = self.family.problem.check(candidate)
        paths = {
            ".".join(check.value.field.path)
            for check in audit.assumptions
            if check.actual not in (None, check.expected)
        }
        paths.update(".".join(check.constraint.path) for check in audit.residuals if check.verdict == "violated")
        return sorted(paths - self.layout.hidden)

    def summary(self) -> dict[str, Any]:
        conditions = self.conditions
        return {
            "kind": self.kind,
            "coordinate_count": len(self.family.parameters),
            "coordinates": self.coordinates.summary(self.reference),
            "case_count": len(self.cases) if self.cases or self.kind == "empty" else None,
            "candidate_count": self.alternatives.candidate_count,
            "case_limit": MAX_CASES,
            "cases": self.alternatives.summary(),
            "selection": self.selection,
            "distinguishing": [q.path for q in self.alternatives.distinguishing],
            "suggestions": self.alternatives.suggestions(self.layout.model_info.highlights),
            "invariants": self.invariants,
            "conditions": conditions,
            "background_conditions": len(self.family.conditions) - len(conditions),
            "reason": self.alternatives.reason,
            "issues": [f"{issue.operation}: {type(issue.error).__name__}" for issue in self.family.issues],
            "violations": self.violations,
            "relations": self.relations,
        }

    def plot(
        self, controls: Sequence[str], output: str, windows: Mapping[str, Sequence[ScalarValue]]
    ) -> dict[str, Any]:
        return FamilyPlot(self.coordinates, controls, output, windows, self.reference).sample()

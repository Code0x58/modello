"""Reusable, experimental reconciliation workflow, independent of trade models.

Try configured entry-error hypotheses before explicitly releasing an input.
Every trial starts from the same immutable snapshot and changes one scalar
observation. Diagnostic witnesses establish feasibility, never the true cause.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Generic, TypeVar, cast

from sympy import Expr, Float, Rational, sympify

from modello import ConstraintKind, Modello, ModelProblem, ScalarValue, SolutionFamily
from modello.diagnostics import ModelConstraint, ModelField, ProblemAnalysis

Model = TypeVar("Model", bound=Modello)


@dataclass(frozen=True, init=False)
class Scale:
    """A labelled, exact multiplicative hypothesis, including a sign reversal."""

    label: str
    factor: Expr

    def __init__(self, label: str, factor: ScalarValue) -> None:
        value = sympify(factor)
        if not label.strip() or not isinstance(value, Expr) or value.is_Rational is not True or value in (0, 1):
            raise ValueError("Supply a label and an exact rational factor other than 0 or 1; use Rational for decimals")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "factor", value)


DECIMAL_SHIFTS = (
    Scale("one missing zero (x10)", 10),
    Scale("one extra zero (/10)", Rational(1, 10)),
    Scale("two missing zeros (x100)", 100),
    Scale("two extra zeros (/100)", Rational(1, 100)),
)


@dataclass(frozen=True, init=False)
class Suspect:
    """An actual problem field and the hypotheses permitted for that field."""

    field: ModelField
    mutations: tuple[Scale, ...]

    def __init__(self, field: ModelField, mutations: Iterable[Scale] = DECIMAL_SHIFTS) -> None:
        selected = tuple(mutations)
        if not all(isinstance(mutation, Scale) for mutation in selected):
            raise TypeError("Supply Scale hypotheses for each suspect field")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "mutations", selected)


@dataclass(frozen=True)
class MutationTrial(Generic[Model]):
    """One proposed value, its possible explanations, and the full checked trial."""

    field: ModelField
    original: Expr
    proposed: Expr
    mutations: tuple[Scale, ...]
    problem: ModelProblem[Model]
    analysis: ProblemAnalysis


@dataclass(frozen=True)
class MutationReport(Generic[Model]):
    """Retain baseline evidence and every unique, nontrivial mutation outcome."""

    baseline: ProblemAnalysis
    trials: tuple[MutationTrial[Model], ...]

    @property
    def matches(self) -> tuple[MutationTrial[Model], ...]:
        """Proposals admitting an independently checked, closed witness."""
        return tuple(trial for trial in self.trials if trial.analysis.status == "consistent")

    @property
    def unresolved(self) -> tuple[MutationTrial[Model], ...]:
        """Undecided trials remain visible; they are not rejected hypotheses."""
        return tuple(trial for trial in self.trials if trial.analysis.status == "unknown")


@dataclass(frozen=True, init=False)
class Reconciliation(Generic[Model]):
    """An explicit suspect list; all other recorded observations stay fixed.

    Each suspect must identify exactly one directly recorded scalar input.
    Nested fields and aliases use their actual bound symbol identities.
    Copied inferred results stay fixed too: prepare from raw observations when
    such results should be recalculated. No source model or snapshot is mutated.
    """

    problem: ModelProblem[Model]
    suspects: tuple[Suspect, ...]

    def __init__(self, problem: ModelProblem[Model], suspects: Iterable[Suspect]) -> None:
        selected = tuple(suspects)
        if not all(isinstance(suspect, Suspect) for suspect in selected):
            raise TypeError("Supply Suspect records for this problem")
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "suspects", selected)
        symbols = [suspect.field.symbol for suspect in self.suspects]
        if len(set(symbols)) != len(symbols):
            raise ValueError("Declare each suspect quantity once, including its aliases")
        for suspect in self.suspects:
            observation = self._observation(suspect.field)
            value = observation.right
            if (
                not isinstance(value, Expr)
                or value.free_symbols
                or value.has(Float)
                or value.is_number is not True
                or value.is_real is not True
                or value.is_finite is not True
            ):
                raise ValueError("Suspect inputs must be exact, finite real observations; use Rational for decimals")

    def _observation(self, field: ModelField) -> ModelConstraint:
        if field not in self.problem.fields:
            raise ValueError("Select an actual field of this problem")
        observations = tuple(
            constraint
            for constraint in self.problem.constraints
            if constraint.left == field.symbol and constraint.kind is ConstraintKind.INPUT
        )
        if len(observations) != 1:
            raise ValueError("A suspect must have exactly one directly recorded scalar observation")
        return observations[0]

    def scan(self) -> MutationReport[Model]:
        """Check concrete mutations individually, without solving backwards.

        An already witnessed baseline returns no proposed changes. Otherwise
        each distinct replacement is bound before diagnostic analysis starts.
        Declaration order is preserved; it does not rank likelihood or cost.
        """
        baseline = self.problem.analyse()
        if baseline.status == "consistent":
            return MutationReport(baseline, ())
        trials: list[MutationTrial[Model]] = []
        for suspect in self.suspects:
            observation = self._observation(suspect.field)
            original = cast(Expr, observation.right)
            proposals: dict[Expr, list[Scale]] = {}
            for mutation in suspect.mutations:
                proposed = original * mutation.factor
                if proposed != original:
                    explanations = proposals.setdefault(proposed, [])
                    if mutation not in explanations:
                        explanations.append(mutation)
            for proposed, mutations in proposals.items():
                trial = replace(
                    self.problem,
                    constraints=tuple(
                        replace(constraint, right=proposed) if constraint is observation else constraint
                        for constraint in self.problem.constraints
                    ),
                    candidates=(),
                )
                trials.append(
                    MutationTrial(suspect.field, original, proposed, tuple(mutations), trial, trial.analyse())
                )
        return MutationReport(baseline, tuple(trials))

    def infer(self, field: ModelField) -> SolutionFamily[Model]:
        """Explicit second stage: release one suspect's recorded scalar input.

        Keep all other observations, definitions, domains, and copied results.
        The returned family can contain zero, one, several, or unresolved states.
        """
        if field not in self.problem.fields or not any(
            field.symbol == suspect.field.symbol for suspect in self.suspects
        ):
            raise ValueError("Select a declared suspect field of this problem")
        observation = self._observation(field)
        trial = replace(
            self.problem,
            constraints=tuple(constraint for constraint in self.problem.constraints if constraint is not observation),
            candidates=(),
        )
        return trial.solutions()

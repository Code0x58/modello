"""Typed Modello failures, separate from expensive diagnostic analysis."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modello.diagnostics import Candidate, ModelProblem


class ModelloError(Exception):
    """Base for Modello's validation and model-construction failures."""


class ModelInputError(ModelloError, TypeError):
    """A constructor, declaration, or diagnostic input has an invalid type."""


class ModelValidationError(ModelloError, ValueError):
    """An argument or model operation violates a Modello requirement."""


class ModelDefinitionError(ModelValidationError):
    """A model declaration contains incompatible fields or references."""


class ModelSolveError(ModelloError, ValueError):
    """A solve failure retaining its problem; analysis is explicitly opt-in.

    ``solution_count`` counts retained solver candidates, not numeric points.
    ``None`` means the backend failed before a count was available.
    """

    def __init__(self, problem: ModelProblem, count: int | None, candidates: tuple[Candidate, ...] = ()) -> None:
        self.problem = replace(problem, candidates=candidates)
        self.solution_count = count
        prefix = "Solver failed" if count is None else f"{count} solutions reported"
        super().__init__(
            f"{prefix} for {problem.model.__name__}({problem.name!r})\n"
            "Inspect .problem or call .problem.analyse() for an opt-in diagnostic report."
        )


class SolutionCountError(ModelSolveError):
    """Construction retained zero or several candidates instead of one.

    A zero count alone does not establish inconsistency. Domain filtering can
    reject candidates that remain available in ``problem.candidates``.
    """

    __match_args__ = ("solution_count",)
    solution_count: int

    def __init__(self, problem: ModelProblem, solution_count: int, candidates: tuple[Candidate, ...] = ()) -> None:
        super().__init__(problem, solution_count, candidates)

    @property
    def returned_count(self) -> int:
        """Count captured solver candidates before domain filtering (if any)."""
        return len(self.problem.candidates)


class SolverBackendError(ModelSolveError):
    """The joint solver raised; its original exception is also the cause."""

    __match_args__ = ("backend_error",)

    def __init__(self, problem: ModelProblem, backend_error: Exception) -> None:
        super().__init__(problem, None)
        self.backend_error = backend_error
        self.__cause__ = backend_error

"""Symbolic models whose named children form a single system of equations."""

from __future__ import annotations

from typing import ClassVar, Mapping, TypeVar, cast

from sympy import Basic, Eq, Expr, S, Symbol, check_assumptions, solve

from modello._collections import ModelObservations as ModelObservations
from modello._expressions import _evaluate, _substitute
from modello._model import (
    BoundInstanceDummy as BoundInstanceDummy,
    InstanceDummy,
    ModelloMeta,
    ModelloMetaNamespace,
    ScalarValue,
    SymbolicCondition as SymbolicCondition,
    SymbolicScalar as SymbolicScalar,
    VectorValue as VectorValue,
    _BoundField,
    _ModelDefinition,
    _ModelReference as _ModelReference,
    _populate,
)
from modello._operators import _definitely_undefined
from modello._problem import (
    Candidate,
    ConstraintKind as ConstraintKind,
    FieldPath as FieldPath,
    FieldValue,
    ModelConstraint as ModelConstraint,
    ModelField as ModelField,
    _algebra,
)
from modello._symbols import Coordinate as Coordinate, Index as Index
from modello.annotations import Annotation as Annotation, annotate as annotate, annotate_model as annotate_model
from modello.diagnostics import ModelProblem, _prepare_problem
from modello.errors import (
    ModelDefinitionError as ModelDefinitionError,
    ModelInputError as ModelInputError,
    ModelloError as ModelloError,
    ModelSolveError as ModelSolveError,
    ModelValidationError as ModelValidationError,
    SolutionCountError,
    SolutionSelectionError as SolutionSelectionError,
    SolverBackendError,
    UnresolvedSolutionError as UnresolvedSolutionError,
)
from modello.model_vectors import (
    ModelRecurrence as ModelRecurrence,
    ModelVector as ModelVector,
    ModelVectorValue as ModelVectorValue,
    model_recurrence as model_recurrence,
    model_vector as model_vector,
)
from modello.series import Recurrence as Recurrence, Series as Series
from modello.solutions import SolutionFamily as SolutionFamily, SolutionIssue as SolutionIssue
from modello.vectors import Vector as Vector, vector as vector


class Modello(metaclass=ModelloMeta):
    """Build a symbolic model, solving its named children together with it."""

    _modello_namespace: ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace("", ())
    _modello_definition: ClassVar[_ModelDefinition]
    _modello_bound_fields: dict[InstanceDummy, _BoundField]
    _modello_bound_templates: dict[str, Basic]
    _modello_bound_functions: dict[Symbol, Symbol]
    __modello_annotations__: ClassVar[Mapping[type[Annotation], Mapping[str, Annotation]]]

    def __init__(self, name: str, **value_map: ScalarValue | VectorValue | Modello | ModelVectorValue[Modello]) -> None:
        """Accept scalars, finite vectors, and compatible child model instances."""
        problem = inspect_model(type(self), name, **value_map)
        reported = _solve_problem(problem)
        solution = _select_candidate(problem, reported)
        _populate(self, problem, solution)


_ModelT = TypeVar("_ModelT", bound=Modello)


def component(model: type[_ModelT]) -> _ModelT:
    """Declare a named child with its instance type in a model class body.

    Use ``child = component(Child)``, then supply a ``Child`` instance when
    constructing the parent model.
    """
    # The class namespace turns this declaration into a scoped reference;
    # populated models expose an instance of this type instead.
    return cast(_ModelT, model)


def inspect_model(
    model: type[_ModelT], name: str, /, **value_map: ScalarValue | VectorValue | Modello | ModelVectorValue[Modello]
) -> ModelProblem[_ModelT]:
    """Prepare the constructor's actual problem without invoking the solver.

    Supply the same scalar values and typed component instances as a normal
    construction. The snapshot retains all aliases, constraint origins, and
    whole-argument observation groups, including copied component values.
    """
    return _prepare_problem(model, name, **value_map)


def _solve_problem(problem: ModelProblem, *, check: bool = True) -> tuple[dict[Symbol, Expr], ...]:
    """Generate joint candidates after expanding directed model definitions.

    Construction and interactive previews share the same equations and ordering.
    Callers choose their candidate checks; an empty result is not itself a proof
    of inconsistency.
    """
    try:
        algebra = _algebra(problem.constraints, problem.templates.values())
    except NotImplementedError as error:
        raise ModelInputError(str(error)) from error
    try:
        fields = tuple(dict.fromkeys(field.symbol for field in problem.fields))
        expressions = {symbol: algebra.definitions.get(symbol, symbol) for symbol in fields}
        equations = [Eq(expressions[left], right) for left, right in algebra.equations]
        if S.false in equations:
            return ()
        equations = [eq for eq in equations if eq is not S.true]
        symbols = list(dict.fromkeys([*algebra.cyclic, *fields]))
        solutions = solve(equations, symbols, dict=True, check=check) if equations else [{}]
        return tuple(
            {symbol: _evaluate(_substitute(expression, result)) for symbol, expression in expressions.items()}
            for result in solutions
        )
    except ModelInputError:
        raise
    except Exception as error:
        raise SolverBackendError(problem, error) from error


def _select_candidate(problem: ModelProblem, reported: tuple[dict[Symbol, Expr], ...]) -> dict[Symbol, Expr]:
    """Select one defined result compatible with the constructor's assumptions.

    Undecided assumptions remain acceptable for symbolic construction. This is
    distinct from the closed, exact witness required by a diagnostic audit.
    """
    candidates = []
    for resolved in reported:
        if any(_definitely_undefined(value) for value in resolved.values()):
            continue
        if all(check_assumptions(value, against=dummy) is not False for dummy, value in resolved.items()):
            candidates.append(resolved)
    if len(candidates) != 1:
        snapshots = tuple(
            Candidate(tuple(FieldValue(field, values[field.symbol]) for field in problem.fields)) for values in reported
        )
        raise SolutionCountError(problem, len(candidates), snapshots)
    return candidates[0]

"""Symbolic models whose named children form a single system of equations."""

from __future__ import annotations

from typing import ClassVar, Mapping, TypeVar, cast

from sympy import Basic, Eq, Expr, S, Symbol, solve

from modello._model import (
    BoundInstanceDummy as BoundInstanceDummy,
    InstanceDummy,
    ModelloMeta,
    ModelloMetaNamespace,
    ScalarValue,
    SymbolicCondition as SymbolicCondition,
    SymbolicScalar as SymbolicScalar,
    _BoundField,
    _ModelReference as _ModelReference,
    _populate,
)
from modello._problem import (
    Candidate,
    ConstraintKind as ConstraintKind,
    FieldPath,
    FieldValue,
    ModelConstraint as ModelConstraint,
    ModelField as ModelField,
)
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
from modello.solutions import SolutionFamily as SolutionFamily, SolutionIssue as SolutionIssue


class Modello(metaclass=ModelloMeta):
    """Build a symbolic model, solving its named children together with it."""

    _modello_namespace: ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace("", ())
    _modello_fields: ClassVar[dict[FieldPath, InstanceDummy]] = {}
    _modello_class_constraints: ClassVar[tuple[tuple[InstanceDummy, Basic], ...]] = ()
    _modello_constraint_paths: ClassVar[tuple[FieldPath, ...]] = ()
    _modello_bound_fields: dict[InstanceDummy, _BoundField]
    __modello_annotations__: ClassVar[Mapping[type[Annotation], Mapping[str, Annotation]]]

    def __init__(self, name: str, **value_map: ScalarValue | Modello) -> None:
        """Accept scalar values and compatible child model instances."""
        problem = inspect_model(type(self), name, **value_map)
        reported = _solve_problem(problem)
        candidates = []
        for resolved in reported:
            if any(value.has(S.NaN, S.ComplexInfinity) for value in resolved.values()):
                continue
            if all(
                getattr(value, f"is_{assumption}", None) in (None, expected)
                for dummy, value in resolved.items()
                for assumption, expected in dummy.assumptions0.items()
            ):
                candidates.append(resolved)
        if len(candidates) != 1:
            snapshots = tuple(
                Candidate(tuple(FieldValue(field, values[field.symbol]) for field in problem.fields))
                for values in reported
            )
            raise SolutionCountError(problem, len(candidates), snapshots)
        solution = candidates[0]
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


def inspect_model(model: type[_ModelT], name: str, /, **value_map: ScalarValue | Modello) -> ModelProblem[_ModelT]:
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
    constraints = [Eq(c.left, c.right) for c in problem.constraints]
    if S.false in constraints:
        return ()
    constraints = [eq for eq in constraints if eq is not S.true]
    fields = tuple(dict.fromkeys(field.symbol for field in problem.fields))
    expressions = {symbol: symbol for symbol in fields}
    pending = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.DEFINITION}
    while pending:
        ready = {left: right for left, right in pending.items() if not right.free_symbols & pending.keys()}
        if not ready:
            break  # Cyclic definitions belong in the joint solve.
        for left, right in ready.items():
            expressions[left] = right.xreplace(expressions)
            del pending[left]
    equations = [eq.xreplace(expressions) for eq in constraints]
    if S.false in equations:
        return ()
    equations = [eq for eq in equations if eq is not S.true]
    symbols = list(dict.fromkeys([*pending, *fields]))
    try:
        solutions = solve(equations, symbols, dict=True, check=check) if equations else [{}]
    except Exception as error:
        raise SolverBackendError(problem, error) from error
    return tuple(
        {symbol: expression.xreplace(result) for symbol, expression in expressions.items()} for result in solutions
    )

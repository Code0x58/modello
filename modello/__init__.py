"""Symbolic models whose named children form a single system of equations."""

from __future__ import annotations

from typing import ClassVar, TypeVar, cast

from sympy import Basic, Eq, S, solve

from modello._model import BoundInstanceDummy as BoundInstanceDummy
from modello._model import InstanceDummy as InstanceDummy
from modello._model import ModelloMeta as ModelloMeta
from modello._model import ModelloMetaNamespace as ModelloMetaNamespace
from modello._model import ScalarValue as ScalarValue
from modello._model import SymbolicCondition as SymbolicCondition
from modello._model import SymbolicScalar as SymbolicScalar
from modello._model import _BoundField as _BoundField
from modello._model import _ModelReference as _ModelReference
from modello._model import _populate as _populate
from modello._problem import Candidate as Candidate
from modello._problem import ConstraintKind as ConstraintKind
from modello._problem import FieldPath as FieldPath
from modello._problem import FieldValue as FieldValue
from modello._problem import ModelConstraint as ModelConstraint
from modello._problem import ModelField as ModelField
from modello.diagnostics import ModelProblem as ModelProblem
from modello.diagnostics import _prepare_problem
from modello.errors import ModelDefinitionError as ModelDefinitionError
from modello.errors import ModelInputError as ModelInputError
from modello.errors import ModelloError as ModelloError
from modello.errors import ModelSolveError as ModelSolveError
from modello.errors import ModelValidationError as ModelValidationError
from modello.errors import SolutionCountError as SolutionCountError
from modello.errors import SolutionSelectionError as SolutionSelectionError
from modello.errors import SolverBackendError as SolverBackendError
from modello.errors import UnresolvedSolutionError as UnresolvedSolutionError
from modello.solutions import SolutionFamily as SolutionFamily
from modello.solutions import SolutionIssue as SolutionIssue


class Modello(metaclass=ModelloMeta):
    """Build a symbolic model, solving its named children together with it."""

    _modello_namespace: ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace("", ())
    _modello_fields: ClassVar[dict[FieldPath, InstanceDummy]] = {}
    _modello_class_constraints: ClassVar[tuple[tuple[InstanceDummy, Basic], ...]] = ()
    _modello_constraint_paths: ClassVar[tuple[FieldPath, ...]] = ()
    _modello_bound_fields: dict[InstanceDummy, _BoundField]

    def __init__(self, name: str, **value_map: ScalarValue | Modello) -> None:
        """Accept scalar values and compatible child model instances."""
        problem = inspect_model(type(self), name, **value_map)
        fields = {field.path: cast(BoundInstanceDummy, field.symbol) for field in problem.fields}
        bindings = {dummy: fields[path] for path, dummy in self._modello_fields.items()}
        constraints = [Eq(c.left, c.right) for c in problem.constraints]
        if S.false in constraints:
            raise SolutionCountError(problem, 0)
        constraints = [eq for eq in constraints if eq is not S.true]
        # Expand directed definitions first. Otherwise a general solver may use
        # an output as a free parameter and unnecessarily solve for its inputs.
        expressions = {d: d for d in bindings.values()}
        pending = {bindings[d]: expr.xreplace(bindings) for d, expr in self._modello_class_constraints}
        while pending:
            ready = {d: expr for d, expr in pending.items() if not expr.free_symbols & pending.keys()}
            if not ready:
                break  # A cyclic set of definitions still belongs in the joint solve.
            for dummy, expr in ready.items():
                expressions[dummy] = expr.xreplace(expressions)
                del pending[dummy]
        equations = [eq.xreplace(expressions) for eq in constraints]
        if S.false in equations:
            raise SolutionCountError(problem, 0)
        equations = [eq for eq in equations if eq is not S.true]
        symbols = list(dict.fromkeys([*pending, *bindings.values()]))
        try:
            solutions = solve(equations, symbols, dict=True) if equations else [{}]
        except Exception as error:
            raise SolverBackendError(problem, error) from error
        candidates = []
        reported = []
        for result in solutions:
            resolved = {d: expr.xreplace(result) for d, expr in expressions.items()}
            reported.append(resolved)
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

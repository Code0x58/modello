#!/usr/bin/env python
"""Module for symbolic modeling of systems."""
import typing
from dataclasses import dataclass, field
from enum import StrEnum

from sympy import Basic, Dummy, Eq, FiniteSet, linsolve, nonlinsolve, solve, solveset
# more verbose path as mypy sees sympy.simplify as a module
from sympy.simplify.simplify import simplify


class ModelloSentinelClass:
    """This class is used for quick type.mro() checks."""


class InstanceDummy(Dummy):
    """Dummy which will create a bound bummy on Modello instantiation."""

    def bound(self, model_name: str) -> "BoundInstanceDummy":
        """Return a dummy bound to a modello instance."""
        return BoundInstanceDummy(model_name + "_" + self.name, **self.assumptions0)


#    # for debugging
#    def _sympystr(self, printer):
#        return "%s[%s]" % (self.name, self.dummy_index)


class BoundInstanceDummy(InstanceDummy):
    """Dummy associated with a Modello instance."""


class SolverStrategy(StrEnum):
    """Constraint solver strategy options."""

    LEGACY = "legacy"
    SET = "set"


class SolutionMode(StrEnum):
    """How non-unique solutions are handled."""

    STRICT = "strict"
    PERMISSIVE = "permissive"


class SolutionKind(StrEnum):
    """Normalized solver result kind."""

    UNIQUE = "unique"
    MULTIPLE = "multiple"
    PARAMETRIC = "parametric"


@dataclass
class SolveResult:
    """Normalized result for solver outputs."""

    kind: SolutionKind
    selected: typing.Dict[BoundInstanceDummy, Basic] = field(default_factory=dict)
    solutions: typing.List[typing.Dict[BoundInstanceDummy, Basic]] = field(
        default_factory=list
    )
    sets: typing.Optional[typing.Dict[Basic, Basic]] = None


class ModelloMetaNamespace(dict):
    """This is so that Modello class definitions implicitly define symbols."""

    def __init__(self, name: str, bases: typing.Tuple[type, ...]) -> None:
        """Create a namespace for a Modello class to use."""
        super().__init__()
        self.name = name
        # map of attributes to sympy Basic (e.g expression, value) objects
        self.attrs: typing.Dict[str, Basic] = {}
        # map of attributes to InstanceDummy instances - metadata used by derived classes
        self.dummies: typing.Dict[str, InstanceDummy] = {}
        # map of attributes to non-modello managed objects
        self.other_attrs: typing.Dict[str, object] = {}
        # map of dummies to dummies that override them - metadata used by derived classes
        self.dummy_overrides: typing.Dict[Dummy, Dummy] = {}

        # build up the attributes from the base classes
        for base in bases:
            if ModelloSentinelClass not in base.mro():
                continue
            parent_namespace = getattr(base, "_modello_namespace", None)
            # TODO: read the following (regarding python's method resolution order) and make sure all is ok:
            #  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.3910&rep=rep1&type=pdf

            if parent_namespace:
                # find which dummies are overridden by this base modello
                for attr in self.dummies.keys() & parent_namespace.dummies.keys():
                    override_dummy = parent_namespace.dummies[attr]
                    base_dummy = self.dummies[attr]
                    self.dummies[attr] = override_dummy
                    self.dummy_overrides[base_dummy] = override_dummy

                self.attrs.update(parent_namespace.attrs)
                self.dummies.update(parent_namespace.dummies)
                self.other_attrs.update(parent_namespace.other_attrs)
                self.update(parent_namespace)
            # substitute overridden dummies in the attributes
            if self.dummy_overrides:
                for attr, value in self.attrs.items():
                    self.attrs[attr] = value.subs(self.dummy_overrides)

    def __setitem__(self, key: str, value: object) -> None:
        """Manage modello attributes as values are assigned."""
        if isinstance(value, Basic):
            if key in self:
                dummy = self.dummies[key]
            elif isinstance(value, InstanceDummy):
                dummy = value
            else:
                dummy = InstanceDummy(key, **value.assumptions0)
            self.attrs[key] = simplify(value).subs(self.dummy_overrides)
            self.dummies[key] = dummy
            value = dummy
        elif key in self.attrs:
            # cannot overried a part of inherited expressions with a non-expression
            raise ValueError(
                "Cannot assign %s.%s to a non-expression" % (self.name, key)
            )
        else:
            self.other_attrs[key] = value
        super().__setitem__(key, value)


class ModelloMeta(type):
    """Used to make Modello class definitions use dummies."""

    @classmethod
    def __prepare__(
        metacls, __name: str, __bases: typing.Tuple[type, ...], **kwds: typing.Any
    ) -> typing.MutableMapping[str, typing.Any]:
        """Return a ModelloMetaNamespace instead of a plain dict to accumlate attributes on."""
        return ModelloMetaNamespace(__name, __bases)

    def __new__(
        mcs,
        name: str,
        bases: typing.Tuple[type, ...],
        meta_namespace: ModelloMetaNamespace,
    ) -> typing.Any:
        """Return a new class with modello attributes."""
        namespace = dict(meta_namespace)
        # could follow django's model of _meta? conflicts?
        namespace["_modello_namespace"] = meta_namespace
        namespace["_modello_class_constraints"] = {
            dummy: meta_namespace.attrs[attr]
            for attr, dummy in meta_namespace.dummies.items()
            if meta_namespace.attrs[attr] is not dummy
        }
        return super().__new__(mcs, name, bases, namespace)


class Modello(ModelloSentinelClass, metaclass=ModelloMeta):
    """Base class for building symbolic models."""

    _modello_namespace: typing.ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace(
        "", ()
    )
    _modello_class_constraints: typing.Dict[InstanceDummy, Basic] = {}

    def __init__(
        self,
        name: str,
        *,
        solver_strategy: SolverStrategy = SolverStrategy.LEGACY,
        solution_mode: SolutionMode = SolutionMode.STRICT,
        solution_selector: typing.Optional[
            typing.Callable[[typing.List[typing.Dict[BoundInstanceDummy, Basic]]], typing.Dict[BoundInstanceDummy, Basic]]
        ] = None,
        **value_map: Basic,
    ) -> None:
        """Initialise a model instance and solve for each attribute."""
        instance_dummies = {
            class_dummy: class_dummy.bound(name)
            for class_dummy in self._modello_namespace.dummies.values()
        }
        self._modello_instance_dummies = instance_dummies

        instance_constraints = {}
        for attr, value in value_map.items():
            value = simplify(value).subs(instance_dummies)
            value_map[attr] = value
            class_dummy = getattr(self, attr)
            instance_dummy = instance_dummies[class_dummy]
            instance_constraints[instance_dummy] = value
        self._modello_instance_constraints: typing.Dict[
            BoundInstanceDummy, Basic
        ] = instance_constraints

        constraints = [
            Eq(instance_dummies[class_dummy], value.subs(instance_dummies))
            for class_dummy, value in self._modello_class_constraints.items()
        ]
        constraints.extend(
            Eq(instance_dummy, value)
            for instance_dummy, value in instance_constraints.items()
        )
        # handy for debugging
        self._modello_constraints: typing.List[Eq] = constraints

        solve_result = self._solve_constraints(
            constraints,
            solver_strategy=solver_strategy,
            solution_mode=solution_mode,
            solution_selector=solution_selector,
        )
        solution = solve_result.selected

        for attr, class_dummy in self._modello_namespace.dummies.items():
            instance_dummy = instance_dummies[class_dummy]
            if instance_dummy in solution:
                value = solution[instance_dummy]
            elif instance_dummy in instance_constraints:
                value = instance_constraints[instance_dummy]
            elif class_dummy in self._modello_class_constraints:
                value = self._modello_class_constraints[class_dummy].subs(
                    instance_dummies
                )
            else:
                value = instance_dummy
            setattr(self, attr, value)

    def _solve_constraints(
        self,
        constraints: typing.List[Eq],
        *,
        solver_strategy: SolverStrategy,
        solution_mode: SolutionMode,
        solution_selector: typing.Optional[
            typing.Callable[[typing.List[typing.Dict[BoundInstanceDummy, Basic]]], typing.Dict[BoundInstanceDummy, Basic]]
        ],
    ) -> SolveResult:
        if not constraints:
            return SolveResult(kind=SolutionKind.UNIQUE, solutions=[{}])

        solver_strategy = SolverStrategy(solver_strategy)
        solution_mode = SolutionMode(solution_mode)

        if solver_strategy is SolverStrategy.LEGACY:
            solutions = solve(constraints, particular=True, dict=True)
        elif solver_strategy is SolverStrategy.SET:
            symbols = sorted(
                set().union(*(eq.free_symbols for eq in constraints)), key=lambda s: s.name
            )
            exprs = [eq.lhs - eq.rhs for eq in constraints]
            if all(expr.is_polynomial(*symbols) and expr.total_degree() <= 1 for expr in exprs):
                set_solution = linsolve(exprs, symbols)
            else:
                set_solution = nonlinsolve(exprs, symbols)

            if isinstance(set_solution, FiniteSet):
                solutions = [dict(zip(symbols, sol)) for sol in set_solution]
            else:
                # fallback to per-symbol solvesets for symbolic set outputs
                solution_sets = {
                    symbol: solveset(exprs[0], symbol) for symbol in symbols
                }
                return SolveResult(kind=SolutionKind.PARAMETRIC, sets=solution_sets)
        else:
            raise ValueError("Unknown solver_strategy: %s" % solver_strategy)

        if len(solutions) == 1:
            return SolveResult(
                kind=SolutionKind.UNIQUE, selected=solutions[0], solutions=solutions
            )
        if len(solutions) > 1:
            if solution_selector is not None:
                selected = solution_selector(solutions)
                return SolveResult(
                    kind=SolutionKind.MULTIPLE,
                    selected=selected,
                    solutions=solutions,
                )
            if solution_mode is SolutionMode.STRICT:
                raise ValueError("%s solutions" % len(solutions))
            return SolveResult(kind=SolutionKind.MULTIPLE, solutions=solutions)
        return SolveResult(kind=SolutionKind.PARAMETRIC, solutions=solutions)

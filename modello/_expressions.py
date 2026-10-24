"""Preserve declared formulas and evaluate them only at algebra boundaries."""

from __future__ import annotations

from functools import cached_property
from typing import Iterable, Mapping

from sympy import (
    And,
    Basic,
    Derivative,
    Expr,
    Integer,
    Integral,
    Limit,
    Piecewise,
    Product,
    S,
    Subs,
    Sum,
    Symbol,
    Tuple,
    default_sort_key,
    simplify,
)

from modello._evaluation import _OPERATORS, _evaluate as _evaluate, _rebuild, _substitute as _substitute
from modello._operators import (
    _conditioned,
    _defined_part,
    _denominator_condition,
    _scalar_atoms,
    _ScopedExpression,
    _ScopedOperator,
    _SymbolicTemplate,
)
from modello.errors import ModelInputError
from modello.series import _project_series, _RecurrenceValue, _SeriesTerm
from modello.stochastic import ItoProcess, _ItoQuantity, _ProcessContext
from modello.vectors import _VectorFormula


def _declaration(expression: Basic) -> Basic:
    """Retain operators in the user-facing definition; simplify ordinary scalars."""
    return expression if expression.has(*_OPERATORS) else simplify(expression)


class _DefinitionResolver:
    """Expand formulas with local binders, sharing work within this problem.

    A differentiation variable remains independent inside its body. Observations
    are applied later. Memoization includes that binder scope: a long shared
    dependency graph must not be re-expanded exponentially for each output.
    """

    def __init__(
        self,
        definitions: Mapping[Symbol, Basic],
        cyclic: frozenset[Symbol],
        ambiguous: frozenset[Symbol],
        observations: Mapping[Symbol, Basic],
        templates: Iterable[_SymbolicTemplate] = (),
        *,
        evaluate_operators: bool = True,
        attach_process_context: bool = True,
    ) -> None:
        self.definitions = definitions
        self.cyclic = cyclic
        self.ambiguous = ambiguous
        self.observations = observations
        self.templates = tuple(templates)
        self.evaluate_operators = evaluate_operators
        self.attach_process_context = attach_process_context
        self._memo: dict[tuple[Basic, frozenset[Basic]], Basic] = {}
        self._active: set[tuple[Basic, frozenset[Basic]]] = set()
        self._series_points: dict[tuple[Basic, Basic], list[Integer]] = {}

    def symbolic(self) -> _DefinitionResolver:
        """Expand laws while retaining parameters for later observations or copies."""
        return _DefinitionResolver(
            self.definitions, self.cyclic, self.ambiguous, {}, self.templates, evaluate_operators=False
        )

    @cached_property
    def _raw_symbolic(self) -> _DefinitionResolver:
        return _DefinitionResolver(
            self.definitions,
            self.cyclic,
            self.ambiguous,
            {},
            self.templates,
            evaluate_operators=False,
            attach_process_context=False,
        )

    def __call__(self, expression: Basic, protected: frozenset[Basic] = frozenset()) -> Basic:
        key = (expression, protected)
        if key not in self._memo:
            if key in self._active:
                raise NotImplementedError("Cyclic symbolic queries cannot be evaluated")
            points = None
            if isinstance(expression, _RecurrenceValue) and isinstance(expression.args[1], Integer):
                law = expression.args[0]
                series = (law.args[4], law.target.series.symbol)
                point = expression.args[1]
                points = self._series_points.setdefault(series, [])
                if points and point >= points[-1]:
                    raise NotImplementedError("Composed recurrence queries must decrease in time around a cycle")
                points.append(point)
            self._active.add(key)
            try:
                self._memo[key] = self._resolve(expression, protected)
            finally:
                self._active.remove(key)
                if points is not None:
                    points.pop()
        return self._memo[key]

    def operators(self, expression: Basic) -> Basic:
        """Resolve operator bodies while retaining ordinary field references outside them."""
        if not expression.has(*_OPERATORS):
            return expression
        if isinstance(expression, _OPERATORS):
            return self(expression)
        args = tuple(self.operators(arg) for arg in expression.args)
        return expression.func(*args) if args != expression.args else expression

    def _resolve(self, expression: Basic, protected: frozenset[Basic]) -> Basic:
        if expression in protected:
            return expression
        if isinstance(expression, _SeriesTerm) and not expression.index.free_symbols & protected:
            projected = _project_series(expression, self.templates)
            if projected != expression:
                return self(projected, protected)
        if isinstance(expression, Symbol):
            if expression in self.definitions:
                return self(self.definitions[expression], protected)
            return self.observations.get(expression, expression)
        if isinstance(expression, Piecewise):
            branches = []
            for value, condition in expression.args:
                condition = self(condition, protected)
                if condition is S.false:
                    continue
                branches.append((self(value, protected), condition))
                if condition is S.true:
                    break
            return Piecewise(*branches)
        if isinstance(expression, _ScopedExpression):
            if self.evaluate_operators and isinstance(expression, _ItoQuantity):
                self._check_stochastic(expression, protected)
            if isinstance(expression, _SymbolicTemplate) and not (
                isinstance(expression, ItoProcess) and expression._context is not None
            ):
                expression = next(
                    (template for template in self.templates if template._same_declaration(expression)), expression
                )
            scoped = protected | frozenset(expression.protected_symbols)
            if expression.calculus_expression is not None:
                self._check_calculus(expression.calculus_expression, scoped)
            args = tuple(
                self(arg, scoped if i in expression.scoped_arguments else protected)
                for i, arg in enumerate(expression.args)
            )
            operator = expression.func(*args) if args != expression.args else expression
            if isinstance(operator, ItoProcess) and self.attach_process_context and operator._context is None:
                operator = operator._with_context(self._process_context(expression, protected))
            if not isinstance(operator, _ScopedOperator) or not self.evaluate_operators:
                return operator
            evaluated = _evaluate(operator)
            return self(evaluated, protected) if evaluated != operator else operator
        if isinstance(expression, _VectorFormula):
            variable, first, last = expression.limits
            return _VectorFormula(
                self(expression.function, protected | {variable}),
                (variable, self(first, protected), self(last, protected)),
                expression.args[2],
            )
        if isinstance(expression, (Sum, Product, Integral)):
            variables = frozenset(expression.variables)
            limits = []
            for i, limit in enumerate(expression.limits):
                # Inner bounds may depend on variables bound by outer operators.
                outer = frozenset(item[0] for item in expression.limits[i + 1 :])
                limits.append((limit[0], *(self(bound, protected | outer) for bound in limit[1:])))
            if isinstance(expression, (Sum, Product)) and any(
                len(limit) == 3 and simplify(limit[2] - limit[1]) == -1 for limit in limits
            ):
                return S.Zero if isinstance(expression, Sum) else S.One
            if isinstance(expression, Integral):
                self._check_calculus(expression.function, protected | variables)
            body = self(expression.function, protected | variables)
            operator = expression.func(body, *limits)
        elif isinstance(expression, Derivative):
            variables = frozenset(expression.variables)
            self._check_calculus(expression.expr, protected | variables)
            body = self(expression.expr, protected | variables)
            operator = Derivative(body, *expression.variable_count, evaluate=False)
        elif isinstance(expression, Limit):
            body, variable, point, direction = expression.args
            self._check_calculus(body, protected | {variable})
            operator = Limit(self(body, protected | {variable}), variable, self(point, protected), dir=str(direction))
        elif isinstance(expression, Subs):
            variables = frozenset(expression.variables)
            body = self(expression.expr, protected | variables)
            operator = Subs(body, expression.variables, tuple(self(point, protected) for point in expression.point))
        else:
            args = tuple(self(arg, protected) for arg in expression.args)
            if not isinstance(expression, Expr):
                return expression.func(*args) if args != expression.args else expression
            condition = And(
                self(_denominator_condition(expression), protected),
                *(_denominator_condition(arg) for arg in args if isinstance(arg, Expr)),
            )
            if condition is S.false:
                return S.NaN
            values = tuple(_defined_part(arg) if isinstance(arg, Expr) else arg for arg in args)
            value = _rebuild(expression, values)
            return _conditioned(value, condition)
        evaluated = _evaluate(operator) if self.evaluate_operators else operator
        return self(evaluated, protected) if evaluated != operator else operator

    def _process_context(self, law: ItoProcess, protected: frozenset[Basic]) -> _ProcessContext:
        processes = tuple(template for template in self.templates if isinstance(template, ItoProcess))
        dynamic = frozenset(process.state for process in processes)
        unresolved = self.cyclic | self.ambiguous
        source = self._raw_symbolic(law, protected | dynamic | unresolved)
        declarations = Tuple(
            *sorted({Tuple(process._state_domain, process.state) for process in processes}, key=default_sort_key)
        )
        context = _ProcessContext(
            declarations,
            Tuple(*source.bound_symbols),
            Tuple(source.drift_expression, source.diffusion_expression),
            Tuple(*sorted(unresolved, key=default_sort_key)),
        )
        return _substitute(context, self.observations)

    def _check_stochastic(self, operator: _ItoQuantity, protected: frozenset[Basic]) -> None:
        """One-state queries must not silently freeze another declared process."""
        processes = tuple(template for template in self.templates if isinstance(template, ItoProcess))
        if not processes:
            return
        dynamic = frozenset(process.state for process in processes)
        resolved = self._raw_symbolic(operator, protected | dynamic)
        law = resolved.args[0]
        state, time = resolved.args[3]
        expression = resolved.args[1].xreplace({state: law.state, time: law.time})
        expression = self(expression, protected | dynamic | frozenset(law.bound_symbols))
        others = dynamic - {law.state}
        states = expression.free_symbols | _scalar_atoms(expression, _SeriesTerm)
        if states & others:
            raise ModelInputError(
                "A one-state Itô transformation cannot differentiate another registered process state"
            )

    def _check_calculus(self, body: Basic, protected: frozenset[Basic]) -> None:
        pending = set(body.free_symbols - protected)
        visited = set()
        while pending:
            symbol = pending.pop()
            if symbol in self.cyclic:
                raise NotImplementedError("Calculus through cyclic model definitions is not supported")
            if symbol in self.ambiguous:
                raise NotImplementedError("Calculus through multiply defined model quantities is not supported")
            visited.add(symbol)
            if symbol in self.definitions:
                pending.update(self.definitions[symbol].free_symbols - protected - visited)


def _expand_definitions(
    definitions: Mapping[Symbol, Basic],
    ambiguous: frozenset[Symbol] = frozenset(),
    observations: Mapping[Symbol, Basic] | None = None,
    templates: Iterable[_SymbolicTemplate] = (),
) -> tuple[dict[Symbol, Basic], tuple[Symbol, ...], _DefinitionResolver]:
    """Resolve the directed part; leave algebraic cycles for the joint solver."""
    pending = dict(definitions)
    ordered: dict[Symbol, Basic] = {}
    while pending:
        ready = {left: right for left, right in pending.items() if not right.free_symbols & pending.keys()}
        if not ready:
            break
        for left, right in ready.items():
            ordered[left] = right
            del pending[left]
    resolve = _DefinitionResolver(ordered, frozenset(pending), ambiguous, observations or {}, templates)
    return {left: resolve(right) for left, right in ordered.items()}, tuple(pending), resolve

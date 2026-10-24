"""Preserve declared formulas and evaluate them only at algebra boundaries."""

from __future__ import annotations

from typing import Iterable, Mapping

from sympy import Basic, Derivative, Integer, Integral, Limit, Piecewise, Product, S, Sum, Symbol, simplify

from modello._evaluation import _OPERATORS, _evaluate as _evaluate, _substitute as _substitute
from modello._operators import _ScopedExpression, _ScopedOperator, _SymbolicTemplate
from modello.series import _project_series, _RecurrenceValue, _SeriesTerm
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
    ) -> None:
        self.definitions = definitions
        self.cyclic = cyclic
        self.ambiguous = ambiguous
        self.observations = observations
        self.templates = tuple(templates)
        self.evaluate_operators = evaluate_operators
        self._memo: dict[tuple[Basic, frozenset[Basic]], Basic] = {}
        self._active: set[tuple[Basic, frozenset[Basic]]] = set()
        self._series_points: dict[Basic, list[Integer]] = {}

    def symbolic(self) -> _DefinitionResolver:
        """Expand laws while retaining parameters for later observations or copies."""
        return _DefinitionResolver(
            self.definitions, self.cyclic, self.ambiguous, {}, self.templates, evaluate_operators=False
        )

    def __call__(self, expression: Basic, protected: frozenset[Basic] = frozenset()) -> Basic:
        key = (expression, protected)
        if key not in self._memo:
            if key in self._active:
                raise NotImplementedError("Cyclic symbolic queries cannot be evaluated")
            points = None
            if isinstance(expression, _RecurrenceValue) and isinstance(expression.args[1], Integer):
                series = expression.args[0].target.series.symbol
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
            if isinstance(expression, _SymbolicTemplate):
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
            if isinstance(expression, Integral):
                self._check_calculus(expression.function, protected | variables)
            body = self(expression.function, protected | variables)
            limits = []
            for i, limit in enumerate(expression.limits):
                # Inner bounds may depend on variables bound by outer operators.
                outer = frozenset(item[0] for item in expression.limits[i + 1 :])
                limits.append((limit[0], *(self(bound, protected | outer) for bound in limit[1:])))
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
        else:
            args = tuple(self(arg, protected) for arg in expression.args)
            return expression.func(*args) if args != expression.args else expression
        evaluated = _evaluate(operator) if self.evaluate_operators else operator
        return self(evaluated, protected) if evaluated != operator else operator

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

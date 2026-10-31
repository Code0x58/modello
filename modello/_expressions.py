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
    Subs,
    Sum,
    Symbol,
    Tuple,
    simplify,
)
from sympy.logic.boolalg import Boolean

from modello._evaluation import (
    _OPERATORS,
    _construct,
    _evaluate as _evaluate,
    _expression_key,
    _predicate_parts,
    _rebuild,
    _resolve_piecewise,
    _substitute as _substitute,
)
from modello._operators import (
    _conditioned,
    _denominator_condition,
    _empty_reduction,
    _held_reduction,
    _ScopedExpression,
    _ScopedOperator,
    _SymbolicTemplate,
)
from modello.series import _project_series, _RecurrenceValue, _SeriesTerm
from modello.vectors import _VectorFormula


def _declaration(expression: Basic) -> Basic:
    """Retain operators in the user-facing definition; simplify ordinary scalars."""
    if expression.has(*_OPERATORS, Piecewise):
        return expression
    condition = _denominator_condition(expression)
    value = simplify(expression)
    return value if _denominator_condition(value) == condition else _conditioned(value, condition)


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
        attach_context: bool = True,
    ) -> None:
        self.definitions = definitions
        self.cyclic = cyclic
        self.ambiguous = ambiguous
        self.observations = observations
        self.templates = tuple(templates)
        self.evaluate_operators = evaluate_operators
        self.attach_context = attach_context
        self._memo: dict[tuple, Basic] = {}
        self._active: set[tuple] = set()
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
            attach_context=False,
        )

    def __call__(self, expression: Basic, protected: frozenset[Basic] = frozenset()) -> Basic:
        key = (_expression_key(expression), protected)
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
        if (
            self.evaluate_operators
            and isinstance(expression, Boolean)
            and not isinstance(expression, Expr)
            and not expression.is_Atom
        ):
            return And(*_predicate_parts(expression, self.operators))
        args = tuple(self.operators(arg) for arg in expression.args)
        if not isinstance(expression, Expr):
            return _construct(expression, args)
        return _rebuild(expression, args, self.operators(_denominator_condition(expression)))

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
            return _resolve_piecewise(expression, lambda value: self(value, protected))
        if isinstance(expression, _ScopedExpression):
            if isinstance(expression, _SymbolicTemplate) and not expression._has_context:
                expression = next(
                    (template for template in self.templates if template._same_declaration(expression)), expression
                )
            expression = expression._in_context(self, protected)
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
            limits = []
            for i, limit in enumerate(expression.limits):
                # Inner bounds may depend on variables bound by outer operators.
                outer = frozenset(item[0] for item in expression.limits[i + 1 :])
                limits.append((limit[0], *(self(bound, protected | outer) for bound in limit[1:])))
            held = _held_reduction(expression, (expression.function, *(Tuple(*limit) for limit in limits)))
            empty = _empty_reduction(held)
            if empty is not None:
                return empty if self.evaluate_operators else held
            if isinstance(expression, Integral):
                self._check_calculus(expression.function, protected | variables)
            body = self(expression.function, protected | variables)
            operator = (
                expression.func(body, *limits)
                if self.evaluate_operators
                else _held_reduction(expression, (body, *(Tuple(*limit) for limit in limits)))
            )
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
                return _construct(expression, args)
            return _rebuild(expression, args, self(_denominator_condition(expression), protected))
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


def _definition_resolver(
    definitions: Mapping[Symbol, Basic],
    ambiguous: frozenset[Symbol] = frozenset(),
    observations: Mapping[Symbol, Basic] | None = None,
    templates: Iterable[_SymbolicTemplate] = (),
    *,
    evaluate_operators: bool = True,
) -> _DefinitionResolver:
    """Prepare directed expansion without evaluating the declared expressions."""
    pending = dict(definitions)
    ordered: dict[Symbol, Basic] = {}
    while pending:
        ready = {left: right for left, right in pending.items() if not right.free_symbols & pending.keys()}
        if not ready:
            break
        for left, right in ready.items():
            ordered[left] = right
            del pending[left]
    return _DefinitionResolver(
        ordered, frozenset(pending), ambiguous, observations or {}, templates, evaluate_operators=evaluate_operators
    )


def _expand_definitions(
    definitions: Mapping[Symbol, Basic],
    ambiguous: frozenset[Symbol] = frozenset(),
    observations: Mapping[Symbol, Basic] | None = None,
    templates: Iterable[_SymbolicTemplate] = (),
) -> tuple[dict[Symbol, Basic], tuple[Symbol, ...], _DefinitionResolver]:
    """Resolve the directed part; leave algebraic cycles for the joint solver."""
    resolve = _definition_resolver(definitions, ambiguous, observations, templates)
    return (
        {left: resolve(right) for left, right in resolve.definitions.items()},
        tuple(left for left in definitions if left in resolve.cyclic),
        resolve,
    )

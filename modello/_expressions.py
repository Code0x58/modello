"""Preserve declared formulas and evaluate them only at algebra boundaries."""

from __future__ import annotations

from typing import Mapping

from sympy import Basic, Derivative, Integral, Limit, Product, Sum, Symbol, simplify

from modello.vectors import _VectorElement, _VectorFormula

_OPERATORS = (Sum, Product, Integral, Derivative, Limit, _VectorElement)


def _declaration(expression: Basic) -> Basic:
    """Retain operators in the user-facing definition; simplify ordinary scalars."""
    return expression if expression.has(*_OPERATORS) else simplify(expression)


def _evaluate(expression: Basic) -> Basic:
    """Use SymPy's own evaluation for sums, products, calculus, and selections."""
    return expression.doit() if expression.has(*_OPERATORS) else expression


def _substitute(expression: Basic, values: Mapping[Symbol, Basic]) -> Basic:
    """Preserve remaining binders when observations replace their variables.

    SymPy uses Subs for an unresolved derivative evaluated at a numeric point;
    xreplace alone would attempt to differentiate with respect to that number.
    """
    if expression.has(Sum, Product, Integral, Derivative, Limit, _VectorFormula):
        for operator in expression.atoms(Sum, Product, Integral, Derivative, Limit, _VectorFormula):
            variables = [operator.args[1]] if isinstance(operator, Limit) else getattr(operator, "variables", ())
            if isinstance(operator, _VectorFormula):
                variables = operator.bound_symbols
            if any(variable in values and values[variable] != variable for variable in variables):
                return expression.subs(values, simultaneous=True)
    return expression.xreplace(values)


class _DefinitionResolver:
    """Expand formulas with local binders, sharing work within this problem.

    A differentiation variable remains independent inside its body. Observations
    are applied later. Memoization includes that binder scope: a long shared
    dependency graph must not be re-expanded exponentially for each output.
    """

    def __init__(self, definitions: Mapping[Symbol, Basic], cyclic: frozenset[Symbol]) -> None:
        self.definitions = definitions
        self.cyclic = cyclic
        self._memo: dict[tuple[Basic, frozenset[Symbol]], Basic] = {}

    def __call__(self, expression: Basic, protected: frozenset[Symbol] = frozenset()) -> Basic:
        key = (expression, protected)
        if key not in self._memo:
            self._memo[key] = self._resolve(expression, protected)
        return self._memo[key]

    def operators(self, expression: Basic) -> Basic:
        """Resolve operator bodies while retaining ordinary field references outside them."""
        if not expression.has(*_OPERATORS):
            return expression
        if isinstance(expression, _OPERATORS):
            return self(expression)
        args = tuple(self.operators(arg) for arg in expression.args)
        return expression.func(*args) if args != expression.args else expression

    def _resolve(self, expression: Basic, protected: frozenset[Symbol]) -> Basic:
        if isinstance(expression, Symbol):
            if expression in self.definitions and expression not in protected:
                return self(self.definitions[expression], protected)
            return expression
        if isinstance(expression, _VectorFormula):
            variable, first, last = expression.limits
            return _VectorFormula(self(expression.function, protected | {variable}), (variable, first, last))
        if isinstance(expression, (Sum, Product, Integral)):
            variables = frozenset(expression.variables)
            self._check_cycles(expression, expression.function, protected | variables)
            body = self(expression.function, protected | variables)
            limits = []
            for i, limit in enumerate(expression.limits):
                # Inner bounds may depend on variables bound by outer operators.
                outer = frozenset(item[0] for item in expression.limits[i + 1 :])
                limits.append((limit[0], *(self(bound, protected | outer) for bound in limit[1:])))
            operator = expression.func(body, *limits)
        elif isinstance(expression, Derivative):
            variables = frozenset(expression.variables)
            self._check_cycles(expression, expression.expr, protected | variables)
            body = self(expression.expr, protected | variables)
            operator = Derivative(body, *expression.variable_count, evaluate=False)
        elif isinstance(expression, Limit):
            body, variable, point, direction = expression.args
            self._check_cycles(expression, body, protected | {variable})
            operator = Limit(self(body, protected | {variable}), variable, self(point, protected), dir=str(direction))
        else:
            args = tuple(self(arg, protected) for arg in expression.args)
            return expression.func(*args) if args != expression.args else expression
        evaluated = operator.doit()
        return self(evaluated, protected) if evaluated != operator else operator

    def _check_cycles(self, operator: Basic, body: Basic, protected: frozenset[Symbol]) -> None:
        if isinstance(operator, (Derivative, Integral, Limit)) and (body.free_symbols - protected) & self.cyclic:
            raise NotImplementedError("Calculus through cyclic model definitions is not supported")


def _expand_definitions(
    definitions: Mapping[Symbol, Basic],
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
    resolve = _DefinitionResolver(ordered, frozenset(pending))
    return {left: resolve(right) for left, right in ordered.items()}, tuple(pending), resolve

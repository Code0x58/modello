"""Preserve declared formulas and evaluate them only at algebra boundaries."""

from __future__ import annotations

from typing import Mapping

from sympy import Add, Basic, Derivative, Integer, Integral, Limit, Mul, Product, S, Sum, Symbol, simplify

from modello.vectors import _VectorElement, _VectorFormula

_OPERATORS = (Sum, Product, Integral, Derivative, Limit, _VectorElement)


def _declaration(expression: Basic) -> Basic:
    """Retain operators in the user-facing definition; simplify ordinary scalars."""
    return expression if expression.has(*_OPERATORS) else simplify(expression)


def _evaluate(expression: Basic) -> Basic:
    """Evaluate native operators, directly reducing explicitly finite vector data."""
    if not expression.has(*_OPERATORS):
        return expression
    expression = expression.replace(_finite_vector_reduction, _reduce_vector)
    if expression.has(Limit):
        # A generic parameterized limit can discard exceptional values, even
        # when its endpoint is fixed: lim (x+a)/(x+2*a) is 1/2 for a != 0 but
        # is 1 for a == 0. Keep it intact until substitutions close the limit.
        # Evaluate siblings independently without differentiating or integrating
        # through a deferred limit as if it were an independent placeholder.
        args = tuple(_evaluate(arg) for arg in expression.args)
        if args != expression.args:
            expression = expression.func(*args)
        if isinstance(expression, Limit):
            if expression.free_symbols:
                return expression
            return expression.doit()
        elif expression.has(Limit):
            return expression
        return expression.doit(deep=False)
    return expression.doit()


def _finite_vector_reduction(expression: Basic) -> bool:
    """Indexed data have explicit entries, rather than a closed-form summand."""
    return (
        isinstance(expression, (Sum, Product))
        and len(expression.limits[-1]) == 3
        and all(isinstance(bound, Integer) for bound in expression.limits[-1][1:])
        and expression.function.has(_VectorElement)
    )


def _reduce_vector(operator: Sum | Product) -> Basic:
    variable, first, last = operator.limits[-1]
    first, last = int(first), int(last)
    reversed_limits = last < first
    if reversed_limits:
        first, last = last + 1, first - 1
    body = operator.func(operator.function, *operator.limits[:-1]) if len(operator.limits) > 1 else operator.function
    terms = []
    for index in range(first, last + 1):
        term = _evaluate(_substitute(body, {variable: Integer(index)}))
        if term is S.NaN:
            return S.NaN
        terms.append(term)
    result = Add(*terms) if isinstance(operator, Sum) else Mul(*terms)
    if reversed_limits:
        return -result if isinstance(operator, Sum) else 1 / result
    return result


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

    def __init__(
        self, definitions: Mapping[Symbol, Basic], cyclic: frozenset[Symbol], ambiguous: frozenset[Symbol]
    ) -> None:
        self.definitions = definitions
        self.cyclic = cyclic
        self.ambiguous = ambiguous
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
            self._check_calculus(expression, expression.function, protected | variables)
            body = self(expression.function, protected | variables)
            limits = []
            for i, limit in enumerate(expression.limits):
                # Inner bounds may depend on variables bound by outer operators.
                outer = frozenset(item[0] for item in expression.limits[i + 1 :])
                limits.append((limit[0], *(self(bound, protected | outer) for bound in limit[1:])))
            operator = expression.func(body, *limits)
        elif isinstance(expression, Derivative):
            variables = frozenset(expression.variables)
            self._check_calculus(expression, expression.expr, protected | variables)
            body = self(expression.expr, protected | variables)
            operator = Derivative(body, *expression.variable_count, evaluate=False)
        elif isinstance(expression, Limit):
            body, variable, point, direction = expression.args
            self._check_calculus(expression, body, protected | {variable})
            operator = Limit(self(body, protected | {variable}), variable, self(point, protected), dir=str(direction))
        else:
            args = tuple(self(arg, protected) for arg in expression.args)
            return expression.func(*args) if args != expression.args else expression
        evaluated = _evaluate(operator)
        return self(evaluated, protected) if evaluated != operator else operator

    def _check_calculus(self, operator: Basic, body: Basic, protected: frozenset[Symbol]) -> None:
        if not isinstance(operator, (Derivative, Integral, Limit)):
            return
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
    resolve = _DefinitionResolver(ordered, frozenset(pending), ambiguous)
    return {left: resolve(right) for left, right in ordered.items()}, tuple(pending), resolve

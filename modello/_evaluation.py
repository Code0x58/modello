"""Shared evaluation of native operators and explicitly finite indexed data."""

from __future__ import annotations

from typing import Mapping

from sympy import Add, Basic, Derivative, Integer, Integral, Limit, Mul, Product, S, Sum, Symbol, simplify

from modello._operators import _bound_coordinates, _ScopedOperator, _SymbolicTemplate
from modello.vectors import _VectorElement, _VectorFormula

_OPERATORS = (Sum, Product, Integral, Derivative, Limit, _VectorElement, _ScopedOperator)
_BINDERS = (Sum, Product, Integral, Derivative, Limit, _VectorFormula, _SymbolicTemplate, _ScopedOperator)


def _evaluate(expression: Basic) -> Basic:
    """Evaluate native operators, directly reducing explicitly finite vector data."""
    if not expression.has(*_OPERATORS):
        return expression
    expression = _expand_finite_indexed(expression)
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


def _finite_indexed_reduction(expression: Basic) -> bool:
    """Finite indexed data and queries should be reduced before generic calculus."""
    return (
        isinstance(expression, (Sum, Product))
        and len(expression.limits[-1]) == 3
        and isinstance(simplify(expression.limits[-1][2] - expression.limits[-1][1]), Integer)
        and expression.function.has(_VectorElement, _ScopedOperator)
    )


def _expand_finite_indexed(expression: Basic) -> Basic:
    """Materialize fixed-width indexed reductions, including moving windows.

    Keep entries symbolic: this is also used to inspect a model law before its
    vector sizes and observations are known.
    """
    return expression.replace(_finite_indexed_reduction, _reduce_indexed)


def _reduce_indexed(operator: Sum | Product) -> Basic:
    variable, first, last = operator.limits[-1]
    width = int(simplify(last - first))
    reversed_limits = width < 0
    if reversed_limits:
        first, width = last + 1, -width - 2
    body = operator.func(operator.function, *operator.limits[:-1]) if len(operator.limits) > 1 else operator.function
    terms = []
    for offset in range(width + 1):
        term = _expand_finite_indexed(_substitute(body, {variable: first + offset}))
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
    if expression.has(*_BINDERS):
        for operator in expression.atoms(*_BINDERS):
            if any(variable in values and values[variable] != variable for variable in _bound_coordinates(operator)):
                return expression.subs(values, simultaneous=True)
    return expression.xreplace(values)

"""Shared evaluation of native operators and explicitly finite indexed data."""

from __future__ import annotations

from typing import Mapping, NamedTuple

from sympy import (
    Add,
    And,
    Basic,
    Derivative,
    Dummy,
    Eq,
    Expr,
    Integer,
    Integral,
    Interval,
    Limit,
    Mul,
    Ne,
    Or,
    Product,
    S,
    Subs,
    Sum,
    Symbol,
    simplify,
    solveset,
)

from modello._operators import (
    _bound_coordinates,
    _conditioned,
    _defined_part,
    _denominator_condition,
    _scalar_atoms,
    _ScopedOperator,
    _SymbolicTemplate,
)
from modello.vectors import _VectorElement, _VectorFormula

_OPERATORS = (Sum, Product, Integral, Derivative, Limit, Subs, _VectorElement, _ScopedOperator)
_BINDERS = (Sum, Product, Integral, Derivative, Limit, Subs, _VectorFormula, _SymbolicTemplate, _ScopedOperator)


class _ConditionalValue(NamedTuple):
    value: Expr
    condition: Basic


def _evaluate_scalar(expression: Expr) -> _ConditionalValue:
    condition = _denominator_condition(expression)
    value = _evaluate(expression)
    return _ConditionalValue(_defined_part(value), And(condition, _denominator_condition(value)))


def _evaluate(expression: Basic) -> Basic:
    """Evaluate supported operations while retaining their validity conditions."""
    if not expression.has(*_OPERATORS):
        return expression
    expression = _expand_finite_indexed(expression)
    if isinstance(expression, _ScopedOperator):
        return expression.doit()
    if isinstance(expression, _VectorElement):
        return expression
    if isinstance(expression, (Limit, Integral, Derivative, Subs)):
        return _calculus(expression)
    if isinstance(expression, (Sum, Product)):
        body = _evaluate(expression.function)
        operator = expression.func(body, *expression.limits)
        if _has_calculus(body):
            return operator
        expanded = _expand_finite_indexed(operator)
        if expanded != operator:
            return _evaluate(expanded)
        expression = operator
        condition = _denominator_condition(body)
        if condition is not S.true:
            variables = set(expression.variables)
            if condition.free_symbols & variables or any(
                bound.free_symbols & variables for limit in expression.limits for bound in limit[1:]
            ):
                # A parameterized summation must not cancel individual poles.
                return expression
            empty = Or(*(Eq(last - first, -1) for _, first, last in expression.limits))
            value = expression.func(_defined_part(expression.function), *expression.limits).doit()
            return _conditioned(value, Or(empty, condition))
        return expression.doit()
    args = tuple(_evaluate(arg) for arg in expression.args)
    if not isinstance(expression, Expr):
        return expression.func(*args) if args != expression.args else expression
    condition = And(
        _denominator_condition(expression),
        *(_denominator_condition(arg) for arg in args if isinstance(arg, Expr)),
    )
    values = tuple(_defined_part(arg) if isinstance(arg, Expr) else arg for arg in args)
    result = _rebuild(expression, values)
    return _conditioned(result, condition)


def _rebuild(expression: Expr, args: tuple[Basic, ...]) -> Expr:
    """Do not cancel an operation whose domain is still unresolved."""
    if isinstance(expression, (Add, Mul)) and any(_pending(arg) for arg in args):
        return expression.func(*args, evaluate=False)
    return expression.func(*args) if args != expression.args else expression


def _pending(expression: Basic) -> bool:
    return (
        _has_calculus(expression)
        or bool(_scalar_atoms(expression, _VectorElement))
        or any(bool(operator.protected_symbols) for operator in _scalar_atoms(expression, _ScopedOperator))
    )


def _has_calculus(expression: Basic) -> bool:
    return any(_scalar_atoms(expression, kind) for kind in (Derivative, Integral, Limit))


def _independent_condition(condition: Basic, variables: set[Basic]) -> Basic:
    return And(*(clause for clause in And.make_args(condition) if not clause.free_symbols & variables))


def _calculus(operator: Limit | Integral | Derivative | Subs) -> Basic:
    if isinstance(operator, Subs):
        body = _evaluate_scalar(operator.expr)
        points = tuple(_evaluate_scalar(point) for point in operator.point)
        held = Subs(_conditioned(*body), operator.variables, tuple(point.value for point in points))
        value = held if _pending(body.value) else held.doit(deep=False)
        return _conditioned(value, And(*(point.condition for point in points)))
    body = _evaluate(operator.args[0])
    variables = set(_bound_coordinates(operator))
    condition = _denominator_condition(body)
    if isinstance(operator, Limit):
        _, variable, point, direction = operator.args
        held = Limit(body, variable, point, dir=str(direction))
        if held.free_symbols:
            return held
        value = Limit(_defined_part(body), variable, point, dir=str(direction)).doit()
        return _conditioned(value, _independent_condition(condition, variables))
    if isinstance(operator, Integral):
        held = Integral(body, *operator.limits)
        if held.free_symbols and not _regular_integral(held, condition):
            return held
        if _has_calculus(body):
            return held
        value = Integral(_defined_part(body), *operator.limits).doit()
        return _conditioned(value, _independent_condition(condition, variables))
    held = Derivative(body, *operator.variable_count, evaluate=False)
    if _has_calculus(body):
        return held
    value = Derivative(_defined_part(body), *operator.variable_count, evaluate=False).doit()
    return _conditioned(value, condition)


def _regular_integral(operator: Integral, condition: Basic) -> bool:
    """Ask SymPy to prove explicit poles stay outside a parameterized interval."""
    variables = set(operator.variables)
    local = [clause for clause in And.make_args(condition) if clause.free_symbols & variables]
    if not local:
        return True
    if len(operator.limits) != 1 or len(operator.limits[0]) != 3:
        return False
    variable, first, last = operator.limits[0]
    if first.is_extended_real is not True or last.is_extended_real is not True:
        return False
    try:
        difference = last - first
        if difference.is_nonnegative is True:
            interval = Interval(first, last)
        elif difference.is_nonpositive is True:
            interval = Interval(last, first)
        else:
            return False
        for clause in local:
            if not isinstance(clause, Ne) or clause.rhs != 0:
                return False
            if solveset(clause.lhs, variable, domain=interval) is not S.EmptySet:
                return False
    except (NotImplementedError, ValueError, TypeError):
        return False
    return True


def _finite_indexed_reduction(expression: Basic) -> bool:
    """Finite indexed data and queries should be reduced before generic calculus."""
    return (
        isinstance(expression, (Sum, Product))
        and len(expression.limits[-1]) == 3
        and isinstance(simplify(expression.limits[-1][2] - expression.limits[-1][1]), Integer)
        and not _has_calculus(expression.function)
        and (
            expression.function.has(_VectorElement, _ScopedOperator)
            or bool(_denominator_condition(expression.function).free_symbols & set(expression.variables))
        )
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
    conditions = []
    for offset in range(width + 1):
        bindings = {variable: first + offset}
        condition = _substitute(_denominator_condition(body), bindings) if isinstance(body, Expr) else S.true
        term = _expand_finite_indexed(_substitute(body, bindings))
        if term is S.NaN:
            return S.NaN
        conditions.append(And(condition, _denominator_condition(term)))
        terms.append(_defined_part(term))
    result = Add(*terms) if isinstance(operator, Sum) else Mul(*terms)
    if reversed_limits:
        result = -result if isinstance(operator, Sum) else 1 / result
    return _conditioned(result, And(*conditions))


def _substitute(expression: Basic, values: Mapping[Symbol, Basic]) -> Basic:
    """Preserve remaining binders when observations replace their variables.

    SymPy uses Subs for an unresolved derivative evaluated at a numeric point;
    xreplace alone would attempt to differentiate with respect to that number.
    """
    if expression.has(Limit) and any(limit.args[1] in values for limit in _scalar_atoms(expression, Limit)):
        expression = _protect_limits(expression, values)
    if expression.has(*_BINDERS):
        for operator in expression.atoms(*_BINDERS):
            if any(variable in values and values[variable] != variable for variable in _bound_coordinates(operator)):
                return expression.subs(values, simultaneous=True)
    return expression.xreplace(values)


def _protect_limits(expression: Basic, values: Mapping[Symbol, Basic]) -> Basic:
    """Alpha-rename Limit binders before native subs can turn them into numbers."""
    if isinstance(expression, _SymbolicTemplate) or not expression.has(Limit):
        return expression
    args = tuple(_protect_limits(arg, values) for arg in expression.args)
    if isinstance(expression, Limit) and expression.args[1] in values:
        body, variable, point, direction = args
        local = Dummy(variable.name, **variable.assumptions0)
        return Limit(body.subs(variable, local, simultaneous=True), local, point, dir=str(direction))
    return expression.func(*args) if args != expression.args else expression

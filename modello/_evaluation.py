"""Shared evaluation of native operators and explicitly finite indexed data."""

from __future__ import annotations

from functools import cache, wraps
from typing import Callable, Iterable, Mapping, NamedTuple

from sympy import (
    Add,
    And,
    Basic,
    Contains,
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
    Piecewise,
    Product,
    S,
    Subs,
    Sum,
    Symbol,
    preorder_traversal,
    simplify,
    solveset,
)
from sympy.core.relational import Relational

from modello._operators import (
    _bound_coordinates,
    _conditioned,
    _defined_part,
    _denominator_condition,
    _held_reduction,
    _scalar_atoms,
    _ScopedExpression,
    _ScopedOperator,
    _SymbolicTemplate,
)
from modello._symbols import Index
from modello.errors import ModelInputError
from modello.vectors import _VectorElement, _VectorFormula

_NATIVE_OPERATORS = (Sum, Product, Integral, Derivative, Limit, Subs)
_OPERATORS = (*_NATIVE_OPERATORS, _VectorElement, _ScopedOperator)
_BINDERS = (*_NATIVE_OPERATORS, _VectorFormula, _SymbolicTemplate, _ScopedOperator)


def _expression_key(value):
    """Retain substitution scopes that SymPy's mathematical equality ignores."""
    if isinstance(value, tuple):
        return tuple(_expression_key(item) for item in value)
    scopes = _substitution_scopes(value)
    return (value, scopes) if scopes else value


def _substitution_scopes(value):
    if isinstance(value, tuple):
        return tuple(scope for item in value for scope in _substitution_scopes(item))
    if not isinstance(value, Basic) or not value.has(Subs):
        return ()
    return tuple(part.args for part in preorder_traversal(value) if isinstance(part, Subs))


def _cache_by_expression(function: Callable[[Basic], Basic]) -> Callable[[Basic], Basic]:
    """Memoize tree rewrites without merging native substitution scopes."""

    @cache
    def cached(key: tuple) -> Basic:
        return function(key[0])

    @wraps(function)
    def resolved(value: Basic) -> Basic:
        return cached((value, _expression_key(value)))

    return resolved


def _unique_expressions(values: Iterable[Basic]) -> tuple[Basic, ...]:
    return tuple({_expression_key(value): value for value in values}.values())


def _native_keys(value):
    """Operands that must survive reconstruction until their premises are checked."""
    if isinstance(value, tuple):
        return set().union(*(_native_keys(item) for item in value))
    if not isinstance(value, Basic) or not value.has(*_NATIVE_OPERATORS):
        return set()
    return {_expression_key(part) for part in preorder_traversal(value) if isinstance(part, _NATIVE_OPERATORS)}


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
    return _rebuild(expression, args, _denominator_condition(expression))


def _construct(expression: Basic, args: tuple[Basic, ...]) -> Basic:
    """Reconstruct a node while retaining operations that still need checking."""
    pending = _native_keys(args)
    if isinstance(expression, (Add, Mul)) and any(_pending(arg) for arg in args):
        # Cached constructors can merge equal Subs with different coordinates.
        value = expression.func._from_args(args)
    elif isinstance(expression, (Sum, Product, Integral)) and pending:
        value = _held_reduction(expression, args)
    elif isinstance(expression, Relational) and pending:
        value = expression.func(*args, evaluate=False)
    else:
        value = expression.func(*args) if _expression_key(args) != _expression_key(expression.args) else expression
    if pending and not pending.issubset(_native_keys(value)):
        raise ModelInputError(
            "Native operation scopes could not be preserved during rebinding; "
            "declare the expression with its new coordinates"
        )
    return value


def _rebuild(expression: Expr, args: tuple[Basic, ...], condition: Basic) -> Expr:
    """Rebuild resolved arguments without dropping domains or pending operations."""
    condition = And(condition, *(_denominator_condition(arg) for arg in args if isinstance(arg, Expr)))
    if condition is S.false:
        return S.NaN
    args = tuple(_defined_part(arg) if isinstance(arg, Expr) else arg for arg in args)
    return _conditioned(_construct(expression, args), condition)


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
        value = held if _has_calculus(body.value) else held.doit(deep=False)
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


def _relative_expression(expression: Expr, index: Symbol) -> Expr:
    """Guard free relative selections, including locally bound moving windows."""
    position = Index(index.name)
    body = _substitute(expression, {index: position})
    return _substitute(_guard_relative_indices(body, frozenset({position})), {position: index})


def _guard_relative_indices(expression: Basic, relative: frozenset[Basic]) -> Basic:
    if isinstance(expression, (Sum, Product, Integral, _VectorFormula)):
        limits = (expression.limits,) if isinstance(expression, _VectorFormula) else expression.limits
        body_relative = relative
        resolved_limits = []
        for variable, *bounds in reversed(limits):
            resolved_limits.append((variable, *(_guard_relative_indices(bound, body_relative) for bound in bounds)))
            dependent = any(bound.free_symbols & body_relative for bound in bounds)
            body_relative = body_relative - {variable}
            if dependent:
                body_relative = body_relative | {variable}
        body = _guard_relative_indices(expression.function, body_relative)
        if isinstance(expression, _VectorFormula):
            return expression.func(body, resolved_limits[0], expression.args[2])
        return expression.func(body, *reversed(resolved_limits))
    if isinstance(expression, Subs):
        bindings = list(zip(expression.variables, expression.point))
        if not _has_calculus(expression.expr):
            # Delegate ordered and compound substitution without evaluating its body.
            body = expression.expr.subs(bindings)
            condition = And(
                _denominator_condition(expression.expr).subs(bindings),
                *(_denominator_condition(point) for point in expression.point),
            )
            if body != expression:
                return _guard_relative_indices(_conditioned(body, condition), relative)
        local = relative
        for variable, point in reversed(bindings):
            dependent = bool(point.free_symbols & local)
            local = local - {variable}
            if dependent:
                local = local | {variable}
        body = _guard_relative_indices(expression.expr, local)
        return Subs(
            body, expression.variables, tuple(_guard_relative_indices(point, relative) for point in expression.point)
        )
    if isinstance(expression, _ScopedExpression):
        scoped = relative - set(expression.protected_symbols)
        args = tuple(
            _guard_relative_indices(arg, scoped if i in expression.scoped_arguments else relative)
            for i, arg in enumerate(expression.args)
        )
    elif isinstance(expression, Derivative):
        local = relative - set(_bound_coordinates(expression))
        return Derivative(_guard_relative_indices(expression.expr, local), *expression.variable_count, evaluate=False)
    elif isinstance(expression, Limit):
        local = relative - set(_bound_coordinates(expression))
        args = (
            _guard_relative_indices(expression.args[0], local),
            *(_guard_relative_indices(arg, relative) for arg in expression.args[1:]),
        )
    else:
        args = tuple(_guard_relative_indices(arg, relative) for arg in expression.args)
    value = expression.func(*args) if args != expression.args else expression
    if isinstance(value, _VectorElement) and value.index.free_symbols & relative:
        valid = And(Contains(value.index, S.Integers), value.index >= 0, value.index < value.vector.shape[0])
        return Piecewise((value, valid), (S.NaN, True))
    return value


def _indexed_value(expression: Basic, index: Symbol, point: Basic) -> Basic:
    """Instantiate one position without capturing native or declared binders."""
    return _expand_finite_indexed(_substitute(expression, {index: point}))


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
            if any(
                (variable in values and values[variable] != variable) or variable.free_symbols & values.keys()
                for variable in _bound_coordinates(operator)
            ):
                return expression.subs(values, simultaneous=True)
    return expression.xreplace(values)


def _rebind(expression: Basic, values: Mapping[Basic, Basic]) -> Basic:
    """Rename a declaration without losing exclusions when arithmetic collapses."""
    if not values:
        return expression

    @_cache_by_expression
    def replace(value: Basic) -> Basic:
        if value in values:
            return values[value]
        args = tuple(replace(arg) for arg in value.args)
        if _expression_key(args) == _expression_key(value.args):
            return value
        result = _construct(value, args)
        condition = _denominator_condition(value) if isinstance(value, Expr) else S.true
        if condition is not S.true and isinstance(result, Expr):
            condition = condition.xreplace(values)
            if _denominator_condition(result) != condition:
                result = _conditioned(result, condition)
        return result

    return replace(expression)


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

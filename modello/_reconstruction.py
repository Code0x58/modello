"""Preserve expression domains and local coordinates during tree reconstruction.

This layer renames and rebuilds declared expressions. Evaluation and property
verification use the same structural guarantees without sharing their policies.
"""

from __future__ import annotations

from functools import cache, wraps
from typing import Callable, Mapping, NamedTuple

from sympy import (
    Add,
    And,
    AtomicExpr,
    Basic,
    Complement,
    Contains,
    Derivative,
    Expr,
    Function,
    Integral,
    Intersection,
    Limit,
    Max,
    Min,
    Mul,
    Piecewise,
    Pow,
    Product,
    S,
    Set,
    Subs,
    Sum,
    Tuple,
    Union,
    preorder_traversal,
)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean, BooleanFunction

from modello._operators import (
    _conditioned,
    _defined_part,
    _denominator_condition,
    _held_reduction,
    _scalar_atoms,
    _ScopedOperator,
    _SymbolicTemplate,
)
from modello.errors import ModelInputError
from modello.vectors import _VectorElement, _VectorFormula

_NATIVE_OPERATORS = (Sum, Product, Integral, Derivative, Limit, Subs)
_OPERATORS = (*_NATIVE_OPERATORS, _VectorElement, _ScopedOperator)
_BINDERS = (*_NATIVE_OPERATORS, _VectorFormula, _SymbolicTemplate, _ScopedOperator)


def _expression_key(value):
    """Distinguish native substitutions with different bound coordinates."""
    if isinstance(value, tuple):
        return tuple(_expression_key(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_expression_key(item) for item in value)
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


def _operation_keys(value, operators, *, exclude=()):
    """Retained operations whose local scopes must survive a tree rewrite."""
    if isinstance(value, tuple):
        return set().union(*(_operation_keys(item, operators, exclude=exclude) for item in value))
    if not isinstance(value, Basic) or not value.has(*operators):
        return set()
    return {
        _expression_key(part)
        for part in preorder_traversal(value)
        if isinstance(part, operators) and not isinstance(part, exclude)
    }


def _source_keys(value):
    """Potential partial sources, including symbols whose definitions are still unknown."""
    if isinstance(value, tuple):
        return set().union(*(_source_keys(item) for item in value))
    if not isinstance(value, Basic):
        return set()
    return (
        set(value.free_symbols)
        | _operation_keys(value, (Expr,), exclude=(Add, Mul, AtomicExpr))
        | {part for part in (S.NaN, S.ComplexInfinity) if value.has(part)}
    )


class _ConditionalValue(NamedTuple):
    value: Basic
    condition: Basic


def _predicate_parts(expression: Basic, resolve: Callable[[Basic], Basic] | None = None) -> _ConditionalValue:
    """Separate Boolean truth from the domains of its whole scalar operands."""
    if isinstance(expression, Expr):
        expression = resolve(expression) if resolve is not None else expression
        return _ConditionalValue(_defined_part(expression), _denominator_condition(expression))
    if not expression.args or expression.is_Atom:
        return _ConditionalValue(resolve(expression) if resolve is not None else expression, S.true)
    parts = tuple(_predicate_parts(arg, resolve) for arg in expression.args)
    condition = And(*(part.condition for part in parts))
    if condition is S.false:
        return _ConditionalValue(expression, condition)
    args = tuple(part.value for part in parts)
    value = _construct(expression, args, domains_separated=True)
    return _ConditionalValue(value, condition)


def _resolve_piecewise(expression: Piecewise, resolve: Callable[[Basic], Basic]) -> Basic:
    """Resolve selectors in order, keeping scalar domains outside Boolean truth."""
    branches = []
    validity = S.true
    for value, condition in expression.args:
        selector = _predicate_parts(condition, resolve)
        # Every selector reached so far must be defined before this branch
        # can be selected, even when an earlier selector is false.
        validity = And(validity, selector.condition)
        if validity is S.false:
            break
        if selector.value is S.false:
            continue
        branches.append((resolve(value), And(validity, selector.value)))
        if selector.value is S.true:
            break
    if validity is not S.true:
        branches.append((S.NaN, S.true))
    return Piecewise(*branches, evaluate=False)


def _is_compound_boolean(expression: Basic) -> bool:
    """A relation or logical expression, rather than a scalar or truth atom.

    SymPy symbols also inherit Boolean; their scalar domains are handled as
    expressions, not by reconstructing logical truth.
    """
    return isinstance(expression, Boolean) and not isinstance(expression, Expr) and not expression.is_Atom


def _tracks_operand_sources(expression: Basic) -> bool:
    """Ordinary expression nodes retain their operands' unevaluated sources.

    Binders handle source domains in their local scopes. Boolean atoms have
    no scalar operands whose domains need retaining.
    """
    return (
        isinstance(expression, (Expr, Boolean, Set, Tuple))
        and not isinstance(expression, (*_BINDERS, _VectorElement))
        and not (isinstance(expression, Boolean) and expression.is_Atom)
    )


def _construct(expression: Basic, args: tuple[Basic, ...], *, domains_separated: bool = False) -> Basic:
    """Rebuild a node without discarding operand domains or local scopes."""
    pending_operations = _operation_keys(args, _NATIVE_OPERATORS)
    operand_sources = _source_keys(args) if not domains_separated and _tracks_operand_sources(expression) else set()
    operand_domain = (
        And(*(_predicate_parts(arg).condition for arg in args))
        if not domains_separated and _is_compound_boolean(expression)
        else S.true
    )
    try:
        if isinstance(expression, (Add, Mul)) and any(_pending(arg) for arg in args):
            # Cached constructors can merge equal Subs with different coordinates.
            value = expression.func._from_args(args)
        elif isinstance(expression, (Sum, Product, Integral)) and pending_operations:
            value = _held_reduction(expression, args)
        elif isinstance(expression, Piecewise):
            value = Piecewise(*args, evaluate=False)
        elif not domains_separated and isinstance(expression, (Add, Mul, Pow, Function, Min, Max)):
            # Renaming alone must not erase retained sources such as 0*log(a).
            value = expression.func(*args, evaluate=False)
            if isinstance(expression, (Add, Mul)) and not pending_operations:
                canonical = expression.func(*args)
                if operand_sources.issubset(_source_keys(canonical)):
                    value = canonical
        elif isinstance(expression, (Relational, BooleanFunction, Contains)) and (
            pending_operations or not domains_separated
        ):
            # Symbols can still expand to partial functions after rebinding.
            value = expression.func(*args, evaluate=False)
        elif not domains_separated and isinstance(expression, (Complement, Intersection, Union)):
            # Set simplification can discard operands whose definedness is
            # still open, just as Boolean simplification can.
            value = expression.func(*args, evaluate=False)
        else:
            # Once operand domains are retained separately, Boolean truth
            # may simplify even when the operands themselves did not change.
            simplify_truth = domains_separated and isinstance(expression, Boolean) and bool(args)
            rebuild_node = simplify_truth or _expression_key(args) != _expression_key(expression.args)
            value = expression.func(*args) if rebuild_node else expression
    except (TypeError, ValueError) as error:
        raise ModelInputError("The resolved operands cannot reconstruct this symbolic expression") from error
    if pending_operations and not pending_operations.issubset(_operation_keys(value, _NATIVE_OPERATORS)):
        raise ModelInputError(
            "Native operation scopes could not be preserved during rebinding; "
            "declare the expression with its new coordinates"
        )
    if operand_sources and not operand_sources.issubset(_source_keys(value)):
        raise ModelInputError("Scalar operand sources could not be preserved during rebinding")
    if operand_domain is not S.true and _expression_key(_predicate_parts(value).condition) != _expression_key(
        operand_domain
    ):
        raise ModelInputError("Boolean reconstruction could not preserve its scalar operand domains")
    return value


def _rebuild(expression: Expr, args: tuple[Basic, ...], condition: Basic) -> Expr:
    """Rebuild resolved arguments without dropping domains or pending operations."""
    condition = And(condition, *(_denominator_condition(arg) for arg in args if isinstance(arg, Expr)))
    if condition is S.false:
        return S.NaN
    args = tuple(_defined_part(arg) if isinstance(arg, Expr) else arg for arg in args)
    return _conditioned(_construct(expression, args, domains_separated=True), condition)


def _pending(expression: Basic) -> bool:
    return (
        _has_calculus(expression)
        or bool(_scalar_atoms(expression, _VectorElement))
        or any(bool(operator.protected_symbols) for operator in _scalar_atoms(expression, _ScopedOperator))
    )


def _has_calculus(expression: Basic) -> bool:
    return any(_scalar_atoms(expression, kind) for kind in (Derivative, Integral, Limit))


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

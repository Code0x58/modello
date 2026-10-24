"""Scoped symbolic declarations and scalar operators, independent of models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sympy import (
    And,
    Basic,
    Derivative,
    Dummy,
    Eq,
    Expr,
    Integral,
    Limit,
    Piecewise,
    Product,
    Q,
    S,
    Subs,
    Sum,
    Symbol,
    diff,
    simplify,
)

from modello.errors import ModelInputError


class _ScopedExpression:
    """Describe which arguments contain local coordinates during expansion."""

    args: tuple[Basic, ...]
    func: Any

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return ()

    @property
    def protected_symbols(self) -> tuple[Basic, ...]:
        """Coordinates whose declarations survive observation substitution."""
        return self.bound_symbols

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset()

    @property
    def calculus_expression(self) -> Basic | None:
        """Body whose definitions must be explicit before differentiation."""
        return None


class _SymbolicTemplate(_ScopedExpression, Basic):
    """A declaration whose identity is retained in its canonical arguments."""

    @property
    def equations(self) -> tuple[Basic, ...]:
        return ()

    @property
    def _binding_signature(self) -> tuple[type, tuple[bool, ...]]:
        return type(self), tuple(getattr(value, "is_integer", None) is True for value in self.bound_symbols)

    def _same_declaration(self, other: _SymbolicTemplate) -> bool:
        """Compare local coordinate names while preserving free field identities."""
        return self == other or (type(self) is type(other) and self.as_dummy() == other.as_dummy())

    def _eval_subs(self, old: Basic, new: Basic) -> Basic | None:
        return self if old in self.protected_symbols else None


class _ScopedOperator(_ScopedExpression, Expr):
    """An unevaluated scalar operation that can expand model definitions."""

    def _eval_subs(self, old: Basic, new: Basic) -> Basic | None:
        if old in self.protected_symbols and self.scoped_arguments:
            evaluated = self.doit()
            return evaluated.subs(old, new) if evaluated != self else Subs(self, old, new)
        return None


class _FunctionSymbol(Dummy):
    """Marker for identities of functions, separate from scalar quantities."""

    def bound(self, name: str) -> _FunctionSymbol:
        return _FunctionSymbol(self.name, **self.assumptions0)


class _DeclarationSymbol(Dummy):
    """Identity retained when a law's local coordinates are alpha-renamed."""


def _bound_coordinates(expression: Basic) -> tuple[Basic, ...]:
    """Coordinates kept local by a native operator or symbolic declaration."""
    if isinstance(expression, _ScopedExpression):
        return expression.protected_symbols
    if isinstance(expression, Limit):
        return (expression.args[1],)
    if isinstance(expression, (Derivative, Integral)):
        return tuple(expression.variables)
    return tuple(getattr(expression, "bound_symbols", ()))


def _affine_coefficients(expression: Expr, variable: Symbol) -> tuple[Expr, Expr] | None:
    """Return the slope and intercept when both are independent of ``variable``."""
    slope = simplify(diff(expression, variable))
    intercept = simplify(expression - slope * variable)
    return None if variable in slope.free_symbols | intercept.free_symbols else (slope, intercept)


def _definitely_undefined(expression: Basic) -> bool:
    """Conditional invalid branches do not make every possible value undefined."""
    if expression is S.NaN or expression is S.ComplexInfinity:
        return True
    if isinstance(expression, Piecewise):
        return all(_definitely_undefined(value) for value, _ in expression.args)
    if isinstance(expression, (_SymbolicTemplate, _ScopedOperator, Derivative, Integral, Limit, Sum, Product)):
        return False
    return any(_definitely_undefined(arg) for arg in expression.args)


def _value_condition(requirements: Mapping[str, bool], value: Expr) -> Basic:
    """Retain declared domain requirements on an evaluated scalar value."""
    if _definitely_undefined(value):
        return S.false
    conditions = []
    for predicate, expected in requirements.items():
        known = getattr(value, f"is_{predicate}", None)
        if known is not None:
            if known != expected:
                return S.false
            continue
        if not hasattr(Q, predicate):
            raise ModelInputError(f"Symbolic laws do not support the {predicate!r} assumption")
        if value.is_real is True and predicate in {
            "positive",
            "nonnegative",
            "negative",
            "nonpositive",
            "zero",
            "nonzero",
        }:
            condition = {
                "positive": value > 0,
                "nonnegative": value >= 0,
                "negative": value < 0,
                "nonpositive": value <= 0,
                "zero": Eq(value, 0),
                "nonzero": ~Eq(value, 0),
            }[predicate]
        else:
            condition = getattr(Q, predicate)(value)
        conditions.append(condition if expected else ~condition)
    return And(*conditions)

"""Scoped symbolic declarations and scalar operators, independent of models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sympy import And, Basic, Derivative, Dummy, Eq, Expr, Integral, Limit, Piecewise, Product, Q, S, Subs, Sum

from modello.errors import ModelInputError


class _ScopedExpression:
    """Describe which arguments contain local coordinates during expansion."""

    args: tuple[Basic, ...]
    func: Any

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return ()

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset()


class _SymbolicTemplate(_ScopedExpression, Basic):
    """A model-local mathematical declaration, rather than a scalar quantity."""

    @property
    def equations(self) -> tuple[Basic, ...]:
        return ()

    @property
    def _binding_signature(self) -> tuple[type, tuple[bool, ...]]:
        return type(self), tuple(getattr(value, "is_integer", None) is True for value in self.bound_symbols)

    def _eval_subs(self, old: Basic, new: Basic) -> Basic | None:
        return self if old in self.bound_symbols else None


class _ScopedOperator(_ScopedExpression, Expr):
    """An unevaluated scalar operation that can expand model definitions."""

    def _eval_subs(self, old: Basic, new: Basic) -> Basic | None:
        if old in self.bound_symbols and self.scoped_arguments:
            evaluated = self.doit()
            return evaluated.subs(old, new) if evaluated != self else Subs(self, old, new)
        return None


class _FunctionSymbol(Dummy):
    """Marker for identities of functions, separate from scalar quantities."""

    def bound(self, name: str) -> _FunctionSymbol:
        return _FunctionSymbol(self.name, **self.assumptions0)


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

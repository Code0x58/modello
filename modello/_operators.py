"""Scoped symbolic declarations and scalar operators, independent of models."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import TYPE_CHECKING, Any, TypeVar, cast

from sympy import (
    And,
    Basic,
    Derivative,
    Dummy,
    Eq,
    Expr,
    Integral,
    Lambda,
    Limit,
    Mul,
    Ne,
    Or,
    Piecewise,
    Product,
    Q,
    Rational,
    S,
    Subs,
    Sum,
    Symbol,
    SympifyError,
    diff,
    piecewise_fold,
    refine,
    simplify,
    sympify,
)

from modello.errors import ModelInputError

if TYPE_CHECKING:
    from modello._expressions import _DefinitionResolver

_Atom = TypeVar("_Atom", bound=Basic)


def _scalar_expression(
    value: object, message: str = "Expected a SymPy-compatible scalar expression", *, strict: bool = False
) -> Expr:
    """Coerce scalars without cancelling poles.

    Strict mode rejects text and preserves finite Decimal values exactly.
    """
    try:
        if strict and isinstance(value, Decimal) and value.is_finite():
            value = Rational(*value.as_integer_ratio())
        expression = sympify(value, strict=strict)
    except (SympifyError, TypeError, ValueError, AttributeError) as error:
        raise ModelInputError(message) from error
    if not isinstance(expression, Expr) or expression.is_Matrix or isinstance(expression, Lambda):
        raise ModelInputError(message)
    return expression


def _scalar_atoms(expression: Basic, kind: type[_Atom]) -> set[_Atom]:
    """Find scalar terms without entering the local bodies of declarations."""
    found: set[_Atom] = set()
    pending = [expression]
    visited: set[Basic] = set()
    while pending:
        value = pending.pop()
        if value in visited or isinstance(value, _SymbolicTemplate):
            continue
        visited.add(value)
        if isinstance(value, kind):
            found.add(value)
        pending.extend(value.args)
    return found


def _denominator_condition(expression: Expr) -> Basic:
    """Retain explicit poles before substitution or algebra can cancel them."""
    if _definitely_undefined(expression):
        return S.false
    if isinstance(expression, Piecewise):
        remaining = S.true
        cases = []
        for value, condition in expression.args:
            cases.append(And(remaining, condition, _denominator_condition(value)))
            remaining = And(remaining, ~condition)
        return Or(*cases)
    if isinstance(expression, (_SymbolicTemplate, _ScopedOperator, Sum, Product, Integral, Derivative, Limit, Subs)):
        return S.true
    return And(
        *(Ne(factor, 0) for factor in Mul.make_args(expression.as_numer_denom()[1])),
        *(_denominator_condition(arg) for arg in expression.args if isinstance(arg, Expr)),
    )


def _defined_part(expression: Expr) -> Expr:
    """Use a guarded expression only under its separately retained domain."""
    if isinstance(expression, Piecewise) and expression.args[-1] == (S.NaN, S.true) and len(expression.args) > 1:
        return Piecewise(*expression.args[:-2], (expression.args[-2][0], True))
    return expression


def _conditioned(expression: Expr, condition: Basic) -> Expr:
    if condition is not S.true:
        try:
            condition = refine(condition)
        except (NotImplementedError, ValueError):
            # An unsupported assumption proof must leave the domain intact.
            pass
    if condition is S.true:
        return expression
    return piecewise_fold(Piecewise((expression, condition), (S.NaN, True)))


class _ScopedExpression:
    """Describe which arguments contain local coordinates during expansion."""

    args: tuple[Basic, ...]
    func: Any

    def _in_context(self, resolver: _DefinitionResolver, protected: frozenset[Basic]) -> Basic:
        """Let an extension retain or validate its declaration context."""
        return cast(Basic, self)

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
    def _has_context(self) -> bool:
        return False

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
        return type(self)(self.name, **self.assumptions0)


class _DeclarationSymbol(_FunctionSymbol):
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

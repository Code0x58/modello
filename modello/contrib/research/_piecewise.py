"""Conservative C0/C1/C2 certificates for one constant piecewise join.

This module imports no proof-engine implementation. The caller supplies its
branch-regularity checker, which remains responsible for branch source guards.
Continuity is relative to the supplied real interval. A join at an included
endpoint can establish C0 from its active side and assigned value, but cannot
establish ordinary derivatives from a one-sided neighborhood.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from sympy import (
    AccumBounds,
    And,
    Basic,
    Eq,
    Expr,
    FiniteSet,
    Interval,
    Limit,
    Piecewise,
    Q,
    S,
    Set,
    Symbol,
    ask,
    diff,
    limit,
    simplify,
)
from sympy.core.relational import Relational

from modello.solutions import _field_domain


class _Regularity(Protocol):
    @property
    def condition(self) -> Basic | None: ...

    @property
    def steps(self) -> tuple[str, ...]: ...

    @property
    def witness(self) -> tuple[tuple[Symbol, Expr], ...]: ...


_Check = Callable[[Expr, Symbol | None, Set, int], _Regularity]


@dataclass(frozen=True)
class PiecewiseCertificate:
    """A supported join's sufficient condition or checked regularity failure.

    ``failure_order`` is set only for a checked finite mismatch, with all
    needed source guards established and lower derivative matches checked.
    A failed Cn branch certificate, a divergent limit, or a missing proof does
    not supply that metadata. ``guards`` retains conditional branch and limit
    guards even when a guarded mismatch must remain unresolved.
    """

    condition: Basic | None
    steps: tuple[str, ...]
    witness: tuple[tuple[Symbol, Expr], ...] = ()
    failure_order: int | None = None
    guards: tuple[Basic, ...] = ()


def _join(expression: Piecewise, variable: Symbol):
    if len(expression.args) != 2 or expression.args[1].cond is not S.true:
        return None
    first, final = expression.args
    condition = first.cond
    if not isinstance(condition, Relational) or condition.rel_op not in {"<", "<=", ">", ">="}:
        return None
    operation = condition.rel_op
    if condition.lhs == variable:
        point = condition.rhs
    elif condition.rhs == variable:
        point = condition.lhs
        operation = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[operation]
    else:
        return None
    if point.free_symbols or point.is_real is not True or point.is_finite is not True:
        return None
    if first.expr.has(Piecewise) or final.expr.has(Piecewise):
        return None
    left, right = (first.expr, final.expr) if operation.startswith("<") else (final.expr, first.expr)
    assigned = first.expr if "=" in operation else final.expr
    return point, left, right, assigned


def _conjunction(conditions: list[Basic | None]) -> Basic | None:
    if any(item is S.false for item in conditions):
        return S.false
    if any(item is None for item in conditions):
        return None
    return And(*conditions)


def _finite(value: Expr, check: _Check, guards: list[Basic | None], steps: list[str], label: str) -> bool | None:
    """Check actual finite real values; retain parameter guards explicitly."""
    if value in (S.NaN, S.ComplexInfinity, S.Infinity, S.NegativeInfinity) or value.is_finite is False:
        steps.append(f"{label} is not finite: {value}.")
        return False
    if value.has(Limit, AccumBounds, Piecewise):
        steps.append(f"{label} has no established single finite limit: {value}.")
        return None
    result = check(value, None, S.Reals, 0)
    guards.append(result.condition)
    steps.extend(f"{label}: {item}" for item in result.steps)
    if result.condition is not None and result.condition is not S.true:
        steps.append(f"{label} retains the condition {result.condition}.")
    if result.condition is None or result.condition is S.false:
        steps.append(f"Definedness of {label} remains unestablished.")
        return None
    assumptions = And(*(item for item in guards if item is not None))
    try:
        real = value.is_real is True or ask(Q.real(value), assumptions) is True
        finite = value.is_finite is True or ask(Q.finite(value), assumptions) is True
    except (ValueError, TypeError, NotImplementedError, AssertionError):
        real = finite = False
    if not real or not finite:
        steps.append(f"Finite real-valuedness of {label} remains unestablished.")
        return None
    return True


def _equal(left: Expr, right: Expr) -> Basic:
    difference = simplify(left - right)
    return Eq(difference, 0)


def piecewise_regular(
    expression: Expr, variable: Symbol | None, domain: Set, order: int, check: _Check
) -> PiecewiseCertificate | None:
    """Check a genuine two-branch Piecewise on a real interval, for orders 0--2.

    ``check(expr, variable, domain, order)`` must certify regularity and retain
    source-domain guards. Constant-value checks pass ``variable=None`` and
    ``order=0`` to the same callback. Unsupported joins return ``None``;
    supported but unproved cases return a certificate with ``condition=None``.

    A singleton domain is supported only for C0: a finite assigned value is
    continuous on that singleton, but it supplies no derivative neighborhood.
    Likewise, an included join endpoint supports only a positive C0 result;
    orders one and two retain the C0 checks and leave differentiability open.
    """
    if type(order) is not int or order not in (0, 1, 2) or variable is None or variable.is_real is not True:
        return None
    declared = _field_domain(variable).intersect(S.Reals)
    if domain.is_subset(declared) is not True:
        return None
    if not isinstance(expression, Piecewise):
        return None
    joined = _join(expression, variable)
    if joined is None:
        return None
    point, left, right, assigned = joined
    if isinstance(domain, FiniteSet):
        if len(domain) != 1 or order != 0:
            return None
        singleton = next(iter(domain))
        if singleton.is_real is not True or singleton.is_finite is not True or singleton.free_symbols:
            return None
        if Eq(singleton, point) is S.true:
            selected = assigned
        elif (singleton < point) is S.true:
            selected = left
        elif (singleton > point) is S.true:
            selected = right
        else:
            return None
        value = selected.subs(variable, singleton)
        steps: list[str] = []
        guards: list[Basic | None] = []
        finite = _finite(value, check, guards, steps, "The assigned singleton value")
        condition = S.false if finite is False else _conjunction(guards) if finite is True else None
        return PiecewiseCertificate(
            condition,
            tuple(steps),
            ((variable, singleton),) if finite is False else (),
            guards=tuple(item for item in guards if item is not None),
        )
    if domain == S.Reals:
        domain = Interval(S.NegativeInfinity, S.Infinity)
    if not isinstance(domain, Interval) or domain.is_subset(S.Reals) is not True:
        return None
    left_domain = domain.intersect(Interval.open(S.NegativeInfinity, point))
    right_domain = domain.intersect(Interval.open(point, S.Infinity))
    included = domain.contains(point)
    if included not in (S.true, S.false) or any(part.is_empty is None for part in (left_domain, right_domain)):
        return None
    active = tuple(
        (side, branch, part)
        for side, branch, part in (("left", left, left_domain), ("right", right, right_domain))
        if part.is_empty is False
    )
    if not active:
        return None
    endpoint_only = included is S.true and len(active) == 1 and order > 0
    checked_order = 0 if endpoint_only else order
    steps = [f"Check the active sides of the constant join {variable}={point}, relative to {domain}."]
    guards = []
    witness: tuple[tuple[Symbol, Expr], ...] = ()
    for side, branch, part in active:
        result = check(branch, variable, part, checked_order)
        guards.append(result.condition)
        steps.extend(f"The {side} branch: {item}" for item in result.steps)
        if result.condition is not None and result.condition is not S.true:
            steps.append(f"The {side} branch retains the condition {result.condition}.")
        if result.condition is S.false and not witness:
            witness = result.witness

    def retained() -> tuple[Basic, ...]:
        return tuple(dict.fromkeys(item for item in guards if item is not None and item is not S.true))

    if _conjunction(guards) is S.false:
        return PiecewiseCertificate(S.false, tuple(steps), witness, guards=retained())
    if included is S.false:
        steps.append("The join is outside the domain; inactive branches and the assigned join value are irrelevant.")
        return PiecewiseCertificate(_conjunction(guards), tuple(steps), guards=retained())
    point_value = assigned.subs(variable, point)
    finite_point = _finite(point_value, check, guards, steps, "The actual assigned join value")
    if finite_point is False:
        return PiecewiseCertificate(S.false, tuple(steps), ((variable, point),), guards=retained())
    if finite_point is None:
        return PiecewiseCertificate(None, tuple(steps), guards=retained())
    equalities: list[Basic] = []
    for degree in range(checked_order + 1):
        limits = []
        for side, branch, _ in active:
            try:
                value = limit(diff(branch, variable, degree), variable, point, dir="-" if side == "left" else "+")
            except (NotImplementedError, ValueError, TypeError):
                steps.append(f"The {side} derivative-order-{degree} limit is unresolved.")
                return PiecewiseCertificate(None, tuple(steps), guards=retained())
            label = f"The {side} derivative-order-{degree} limit"
            finite = _finite(value, check, guards, steps, label)
            if finite is not True:
                condition = S.false if finite is False and _conjunction(guards) is S.true else None
                return PiecewiseCertificate(
                    condition, tuple(steps), ((variable, point),) if condition is S.false else (), guards=retained()
                )
            limits.append(value)
            steps.append(f"{label} is {value}.")
        comparisons = [_equal(limits[0], limits[1])] if len(limits) == 2 else []
        if degree == 0:
            comparisons.append(_equal(point_value, limits[0]))
        for equality in comparisons:
            if equality is S.false:
                source_checked = _conjunction(guards) is S.true
                lower_checked = all(item is S.true for item in equalities)
                steps.append(f"Checked finite derivative-order-{degree} values do not match at the join.")
                return PiecewiseCertificate(
                    S.false if source_checked else None,
                    tuple(steps),
                    ((variable, point),) if source_checked else (),
                    degree if source_checked and lower_checked else None,
                    retained(),
                )
            equalities.append(equality)
    condition = _conjunction([*guards, *equalities])
    if endpoint_only:
        steps.append(
            f"The endpoint C0 condition is {condition}; a one-sided domain does not establish ordinary derivatives "
            "at its included join endpoint."
        )
        return PiecewiseCertificate(None, tuple(steps), guards=retained())
    steps.append(
        f"Finite one-sided derivatives through order {order} match, including the actual assigned value, "
        "under the retained conditions."
    )
    return PiecewiseCertificate(condition, tuple(steps), guards=retained())

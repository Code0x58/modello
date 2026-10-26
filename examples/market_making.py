"""Explore Guéant's continuous-time CARA market maker and its component requirements.

Run ``python -m examples.market_making``. The requirement checks are deliberately
small example functions, not a general proof system or a core Modello API.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

from sympy import (
    And,
    Derivative,
    Eq,
    Expr,
    Limit,
    Mul,
    Piecewise,
    S,
    Subs,
    Symbol,
    diff,
    exp,
    factor,
    log,
    preorder_traversal,
    reduce_inequalities,
    simplify,
)
from sympy.calculus.util import continuous_domain
from sympy.core.function import AppliedUndef
from sympy.logic.boolalg import Boolean

from examples.portfolio.optimal_execution import InventoryRisk
from modello import Annotation, Coordinate, InstanceDummy, Modello, annotate_model, component

Condition: TypeAlias = Boolean | None
Verdict = Literal["strictly true", "conditionally true", "strictly false", "unresolved"]


@dataclass(frozen=True)
class Requirement:
    """A universal check; a conditional result contains only free parameters.

    ``None`` means that this example could not discharge the obligation. It is
    neither a proof of failure nor a known sufficient parameter condition.
    """

    name: str
    condition: Condition

    @property
    def verdict(self) -> Verdict:
        if self.condition is S.true:
            return "strictly true"
        if self.condition is S.false:
            return "strictly false"
        if self.condition is None:
            return "unresolved"
        return "conditionally true"


@dataclass(frozen=True)
class Requirements(Annotation):
    """Definition-time evidence, carried into the parent's component paths."""

    checks: tuple[Requirement, ...]


Curve = TypeVar("Curve", bound=Modello)


def checked(checks: Callable[[Curve], tuple[Requirement, ...]]) -> Callable[[type[Curve]], type[Curve]]:
    """Record checks immediately after a class body, without rejecting experiments.

    Reapply this decorator when a subclass changes a checked declaration.
    Conditional results are obligations, not assumptions added to the solver.
    """

    def decorate(model: type[Curve]) -> type[Curve]:
        return annotate_model(model, Requirements(checks(model("definition"))))

    return decorate


def smooth(expression: Expr, variable: Symbol, order: int) -> Condition:
    """Certify C1/C2 on all of R by continuity of the explicit derivatives.

    Unsupported domains (including arbitrary functions and some piecewise
    formulas) remain unresolved. Merely constructing Derivative is no proof.
    """
    if order not in (1, 2):
        raise ValueError("This example checks C1 or C2")
    # continuous_domain quantifies the coordinate, not parameters. For example,
    # it returns R for x/a without checking the admissible parameter value a=0.
    if any(variable not in part.free_symbols and part.is_finite is not True for part in preorder_traversal(expression)):
        return None
    for degree in range(order + 1):
        derivative = diff(expression, variable, degree)
        try:
            domain = continuous_domain(derivative, variable, S.Reals)
        except (NotImplementedError, ValueError):
            return None
        if domain != S.Reals:
            return S.false if (S.Reals - domain).is_empty is False else None
    return S.true


def _all(conditions: tuple[Condition, ...]) -> Condition:
    if S.false in conditions:
        return S.false
    if None in conditions:
        return None
    return And(*conditions)


def _sign(expression: Expr, variable: Symbol, *, strict: bool) -> Condition:
    """Remove proven positive factors, then check a parameter or real line.

    Parameter-dependent, coordinate-dependent inequalities outside this small
    reduction remain unresolved. No sample grid stands in for a universal test.
    """
    expression = factor(expression)
    if expression.is_Mul:
        expression = Mul(*(term for term in expression.args if term.is_positive is not True))
    if expression.has(AppliedUndef, Derivative):
        return None
    condition = expression > 0 if strict else expression >= 0
    if condition in (S.true, S.false) or variable not in expression.free_symbols:
        return condition
    if expression.free_symbols != {variable}:
        return None
    try:
        allowed = reduce_inequalities([condition], variable).as_set()
    except (NotImplementedError, ValueError):
        return None
    return S.true if allowed == S.Reals else S.false


def positive_slope(expression: Expr, variable: Symbol) -> Condition:
    """A C1, everywhere positive derivative certifies strict increase on R."""
    return _all((smooth(expression, variable, 1), _sign(diff(expression, variable), variable, strict=True)))


def negative_slope(expression: Expr, variable: Symbol) -> Condition:
    """Guéant requires an everywhere negative intensity derivative on R."""
    return positive_slope(-expression, variable)


def convex(expression: Expr, variable: Symbol) -> Condition:
    """Certify convexity through an everywhere nonnegative C2 curvature."""
    return _all((smooth(expression, variable, 2), _sign(diff(expression, variable, 2), variable, strict=False)))


def vanishes_at_infinity(expression: Expr, variable: Symbol) -> Condition:
    """Check the full tail limit; an unevaluated limit remains unresolved."""
    try:
        value = Limit(expression, variable, S.Infinity).doit()
    except (NotImplementedError, ValueError):
        return None
    return None if value.has(Limit) else Eq(value, 0)


def uniform_curvature_bound(expression: Expr, variable: Symbol) -> Condition:
    """Check sup(Lambda Lambda'' / Lambda'^2) < 2 for constant ratios only.

    A pointwise inequality alone would not prove this uniform bound. The
    denominator must be nonzero everywhere before any cancellation is valid.
    """
    slope = diff(expression, variable)
    nonzero = _sign(slope**2, variable, strict=True)
    if nonzero is S.false:
        return S.false
    ratio = simplify(expression * diff(expression, variable, 2) / slope**2)
    if variable in ratio.free_symbols or ratio.has(AppliedUndef, Derivative, S.NaN, S.ComplexInfinity):
        return None
    return _all((nonzero, ratio < 2))


def utility_requirements(utility: CARAUtility) -> tuple[Requirement, ...]:
    """Requirements consumed by the CARA reduction, before an HJB is solved."""
    return (
        Requirement("positive risk aversion", utility.gamma > 0),
        Requirement("utility C2 on R", smooth(utility.value, utility.wealth, 2)),
        Requirement("utility positive slope on R", positive_slope(utility.value, utility.wealth)),
        Requirement("utility concave on R (C2 certificate)", convex(-utility.value, utility.wealth)),
    )


def intensity_requirements(intensity: ExponentialIntensity) -> tuple[Requirement, ...]:
    """The intensity hypotheses from Guéant's general market-making framework."""
    rate, distance = intensity.rate, intensity.distance
    return (
        Requirement("positive intensity on R", _sign(rate, distance, strict=True)),
        Requirement("intensity C2 on R", smooth(rate, distance, 2)),
        Requirement("intensity negative slope on R", negative_slope(rate, distance)),
        Requirement("intensity vanishes at +infinity", vanishes_at_infinity(rate, distance)),
        Requirement("uniform intensity curvature ratio < 2", uniform_curvature_bound(rate, distance)),
    )


def risk_requirements(risk: InventoryRisk) -> tuple[Requirement, ...]:
    """The diffusion contribution is a smooth convex cost of signed inventory."""
    return (
        Requirement("nonnegative inventory risk", _sign(risk.cost, risk.inventory, strict=False)),
        Requirement("inventory risk convex on R (C2 certificate)", convex(risk.cost, risk.inventory)),
    )


def terminal_requirements(penalty: QuadraticTerminalPenalty) -> tuple[Requirement, ...]:
    """A quadratic specialization of the terminal penalty ell(|q|)."""
    return (
        Requirement("nonnegative terminal penalty", _sign(penalty.cost, penalty.inventory, strict=False)),
        Requirement("terminal penalty convex on R (C2 certificate)", convex(penalty.cost, penalty.inventory)),
    )


@checked(utility_requirements)
class CARAUtility(Modello):
    """Increasing, concave utility of wealth; gamma is inverse currency."""

    wealth = Coordinate("w", real=True)
    gamma = InstanceDummy("gamma", positive=True)
    value = -exp(-gamma * wealth)


@checked(intensity_requirements)
class ExponentialIntensity(Modello):
    """Trade arrival rate A exp(-k delta), for any real quote distance."""

    distance = Coordinate("delta", real=True)
    scale = InstanceDummy("A", positive=True)
    decay = InstanceDummy("k", positive=True)
    rate = scale * exp(-decay * distance)


@checked(risk_requirements)
class QuadraticInventoryRisk(InventoryRisk):
    """Reuse the execution example's gamma sigma² q² / 2 diffusion cost."""


@checked(terminal_requirements)
class QuadraticTerminalPenalty(Modello):
    """A smooth convex terminal cost of signed inventory."""

    inventory = Coordinate("q", real=True)
    eta = InstanceDummy("eta", nonnegative=True)
    cost = eta * inventory**2


class ContinuationValue(Modello):
    """One state of theta(t, q) on the inventory lattice, with time continuous."""

    step = InstanceDummy("Delta", positive=True)
    current = InstanceDummy("theta", real=True)
    after_bid = InstanceDummy("theta_plus", real=True)
    after_ask = InstanceDummy("theta_minus", real=True)
    time_derivative = InstanceDummy("theta_t", real=True)
    bid_cost = (current - after_bid) / step
    ask_cost = (current - after_ask) / step


class ExponentialQuote(Modello):
    """Exact CARA Hamiltonian and maximizing quote for exponential intensity.

    These formulas require gamma > 0 and step > 0. They allow negative quote
    distances; constraining the controls would change the maximization.
    """

    intensity = component(ExponentialIntensity)
    distance = ExponentialIntensity.distance
    gamma = InstanceDummy("gamma", positive=True)
    step = InstanceDummy("Delta", positive=True)
    cost = InstanceDummy("p", real=True)
    gain = intensity.rate * (1 - exp(-gamma * step * (distance - cost))) / gamma
    optimal_distance = cost + log(1 + gamma * step / intensity.decay) / (gamma * step)
    value = Subs(gain, distance, optimal_distance)


class MarketMaker(Modello):
    """A single-asset slice of Guéant's reduced CARA HJB (Model A).

    Observe residual=0 to infer theta_t from neighboring continuation values.
    The inventory belongs to {-Q, -Q+Delta, ..., Q}. This slice does not solve
    the coupled terminal-value ODE system or validate the lattice membership.
    """

    utility = component(CARAUtility)
    risk = component(QuadraticInventoryRisk)
    terminal = component(QuadraticTerminalPenalty)
    continuation = component(ContinuationValue)
    bid = component(ExponentialQuote)
    ask = component(ExponentialQuote)

    gamma = utility.gamma
    gamma = risk.gamma
    gamma = bid.gamma
    gamma = ask.gamma
    step = continuation.step
    step = bid.step
    step = ask.step
    bid_cost = continuation.bid_cost
    bid_cost = bid.cost
    ask_cost = continuation.ask_cost
    ask_cost = ask.cost

    capacity = InstanceDummy("Q", positive=True)
    inventory = risk.inventory
    bid_term = Piecewise((bid.value, inventory < capacity), (0, True))
    ask_term = Piecewise((ask.value, inventory > -capacity), (0, True))
    residual = -continuation.time_derivative + risk.cost - bid_term - ask_term
    terminal_value = -Subs(terminal.cost, QuadraticTerminalPenalty.inventory, inventory)


def main() -> None:
    """Print the definition checks and one exact interior HJB calculation."""
    for path, report in Requirements.for_model(MarketMaker).items():
        for check in report.checks:
            print(f"{path}: {check.name}: {check.verdict}")
    quote = ExponentialQuote(
        "quote", intensity=ExponentialIntensity("liquidity", scale=2, decay=1), gamma=1, step=1, cost=0
    )
    print(f"optimal distance: {quote.optimal_distance}; Hamiltonian: {simplify(quote.value)}")
    maker = MarketMaker(
        "interior",
        gamma=1,
        step=1,
        capacity=1,
        inventory=0,
        residual=0,
        risk=QuadraticInventoryRisk("risk", sigma=1),
        continuation=ContinuationValue("theta", current=0, after_bid=0, after_ask=0),
        bid=quote,
        ask=quote,
    )
    print(f"HJB time derivative at q=0: {maker.continuation.time_derivative}")


if __name__ == "__main__":
    main()

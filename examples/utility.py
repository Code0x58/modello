"""Cash utility definitions and symbolic Arrow–Pratt risk-aversion reports."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite

from sympy import Expr, Rational, Symbol, diff, exp, log, simplify, sympify

from examples.compute.named_map import NamedMap
from examples.compute.scalar_plan import ScalarPlan
from examples.compute.series import Series, Time


@dataclass(frozen=True)
class CashUtility:
    """An increasing utility expression with wealth as its sole free symbol.

    Wealth is measured in reporting-currency units. Utility levels depend on
    normalization; the derivative ratios survive positive affine transformations.
    """

    name: str
    wealth: Symbol
    expression: Expr

    def __post_init__(self) -> None:
        expression = sympify(self.expression)
        if not self.name or not isinstance(self.wealth, Symbol) or expression.free_symbols - {self.wealth}:
            raise ValueError("Name a utility expression with wealth as its only free symbol")
        if diff(expression, self.wealth) == 0:
            raise ValueError("Utility must vary with wealth")
        object.__setattr__(self, "expression", expression)

    @cached_property
    def expressions(self) -> NamedMap[Expr]:
        """Keep the utility, its derivatives, and their ratios inspectable."""
        first = diff(self.expression, self.wealth)
        second = diff(first, self.wealth)
        absolute = simplify(-second / first)
        return NamedMap(
            {
                "utility": self.expression,
                "marginal_utility": first,
                "curvature": second,
                "absolute_risk_aversion": absolute,
                "relative_risk_aversion": simplify(self.wealth * absolute),
            }
        )

    @cached_property
    def plan(self) -> ScalarPlan:
        return ScalarPlan(NamedMap({"wealth": self.wealth}), self.expressions)

    def over(self, wealth: Series[Time]) -> NamedMap[Series[Time]]:
        """Evaluate utility and local risk preferences on a positive wealth path."""
        if any(value <= 0 for value in wealth.values):
            raise ValueError("Utility reporting requires positive wealth")
        try:
            result = self.plan.over(wealth.times, NamedMap({"wealth": wealth}))
        except (ArithmeticError, ValueError) as error:
            raise ValueError("Utility and its derivative ratios must be finite on the wealth path") from error
        if any(value <= 0 for value in result["marginal_utility"].values):
            raise ValueError("Utility must be increasing on the wealth path")
        return result


def coefficient(value: float) -> Expr:
    if not isfinite(value) or value < 0:
        raise ValueError("Risk-aversion coefficients must be finite and nonnegative")
    return Rational(str(value))


def cara(absolute_risk_aversion: float) -> CashUtility:
    """Constant absolute risk aversion, with a linear risk-neutral limit."""
    alpha = coefficient(absolute_risk_aversion)
    wealth = Symbol("wealth", positive=True)
    expression = (1 - exp(-alpha * wealth)) / alpha if alpha else wealth
    return CashUtility(f"CARA (alpha={absolute_risk_aversion:g} per USD)", wealth, expression)


def crra(relative_risk_aversion: float) -> CashUtility:
    """Constant relative risk aversion, including logarithmic utility at gamma=1."""
    gamma = coefficient(relative_risk_aversion)
    wealth = Symbol("wealth", positive=True)
    expression = log(wealth) if gamma == 1 else (wealth ** (1 - gamma) - 1) / (1 - gamma)
    return CashUtility(f"CRRA (gamma={relative_risk_aversion:g})", wealth, expression)

"""Small, inspectable least-squares problems derived from model residuals."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from sympy import Expr, S, Symbol, diff, linear_eq_to_matrix, simplify
from sympy.solvers.solveset import NonlinearError

from examples.compute.named_map import NamedMap
from examples.portfolio.sources import Fill


@dataclass(frozen=True)
class ImpactObservation:
    """An aggregate trade bucket and an independently estimated temporary cost.

    Quantity is absolute shares; duration hours; volume market shares/hour;
    cost USD after removing separately measured fees, spread, and price drift.
    """

    quantity: float
    duration: float
    volume: float
    cost: float

    def __post_init__(self) -> None:
        if not all(isfinite(value) for value in (self.quantity, self.duration, self.volume, self.cost)):
            raise ValueError("Impact observations must be finite")
        if min(self.quantity, self.duration, self.volume) <= 0:
            raise ValueError("Impact observations need positive quantity, duration, and market volume")

    @classmethod
    def from_fills(
        cls, fills: tuple[Fill, ...], *, reference_price: float, duration: float, volume: float
    ) -> ImpactObservation:
        """Aggregate one side of one holding against a supplied unaffected reference.

        Fees are excluded. The caller must remove spread and exogenous drift
        from this reference before treating the residual as temporary impact.
        """
        if not fills or not isfinite(reference_price) or reference_price <= 0:
            raise ValueError("Supply fills and a positive independent reference price")
        first = fills[0]
        if any(
            (fill.account, fill.member) != (first.account, first.member) or fill.quantity * first.quantity <= 0
            for fill in fills
        ):
            raise ValueError("An impact bucket must contain one account, holding, and trade side")
        return cls(
            sum(abs(fill.quantity) for fill in fills),
            duration,
            volume,
            sum(fill.quantity * (fill.price - reference_price) for fill in fills),
        )


@dataclass(frozen=True)
class Calibration:
    """Unconstrained least squares, affine in the named unknown parameters."""

    parameters: NamedMap[Symbol]
    residuals: tuple[Expr, ...]
    estimates: NamedMap[Expr]
    normal_equations: tuple[Expr, ...]

    @property
    def loss(self) -> Expr:
        replacements = dict(zip(self.parameters.values(), self.estimates.values(), strict=True))
        return simplify(sum((residual**2 for residual in self.residuals), S.Zero).subs(replacements))

    @classmethod
    def fit(cls, residuals: tuple[Expr, ...], parameters: NamedMap[Symbol]) -> Calibration:
        """Generate normal equations and reject unidentifiable coefficient sets."""
        symbols = tuple(parameters.values())
        if not residuals or not symbols or len(set(symbols)) != len(symbols):
            raise ValueError("Calibration needs observations and independent named parameters")
        if set().union(*(residual.free_symbols for residual in residuals)) - set(symbols):
            raise ValueError("Bind other model parameters before fitting")
        try:
            design, _ = linear_eq_to_matrix(residuals, symbols)
        except NonlinearError:
            raise ValueError("This small calibrator requires residuals affine in fitted parameters") from None
        if design.rank() < len(symbols):
            raise ValueError("Observations do not identify these parameters separately")
        loss = sum((residual**2 for residual in residuals), S.Zero)
        equations = tuple(diff(loss, parameter) for parameter in symbols)
        matrix, rhs = linear_eq_to_matrix(equations, symbols)
        values = tuple(map(simplify, matrix.LUsolve(rhs)))
        if any(value.is_finite is not True or value.is_real is not True for value in values):
            raise ValueError("Calibration must produce finite real coefficients")
        if any(
            (parameter.is_positive and value.is_positive is not True)
            or (parameter.is_nonnegative and value.is_nonnegative is not True)
            for parameter, value in zip(symbols, values, strict=True)
        ):
            raise ValueError("Unconstrained fit is outside the parameter domain")
        return cls(parameters, residuals, NamedMap(zip(parameters, values, strict=True)), equations)

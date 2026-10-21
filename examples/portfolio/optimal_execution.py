"""Derive execution, calibration, and liquidity views from one continuous model."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import isfinite

from sympy import (
    Eq,
    Expr,
    Function,
    Matrix,
    Rational,
    S,
    Symbol,
    diff,
    dsolve,
    euler_equations,
    integrate,
    linear_eq_to_matrix,
    nsolve,
    simplify,
    sympify,
)
from sympy.solvers.solveset import NonlinearError

from examples.data import NamedMap
from examples.portfolio.sources import Fill, timestamp
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, Modello, ScalarValue, component


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


class QuadraticImpact(Modello):
    """Temporary execution cost rate, quadratic in trading speed."""

    speed = InstanceDummy("speed", real=True)
    volume = InstanceDummy("volume", positive=True)
    eta = InstanceDummy("eta", positive=True)
    cost = eta * speed**2 / volume


class InventoryRisk(Modello):
    """CARA risk penalty on the variance rate of remaining inventory."""

    inventory = InstanceDummy("inventory", real=True)
    sigma = InstanceDummy("sigma", positive=True)
    gamma = InstanceDummy("gamma", nonnegative=True)
    variance = sigma**2 * inventory**2
    cost = gamma * variance / 2


class ExecutionDensity(SymbolicModel):
    """USD/hour impact and CARA risk penalty for a deterministic liquidation.

    Inventory is shares remaining; speed is shares sold per hour. Volume is
    market shares/hour; sigma is USD/share/sqrt(hour); eta is USD/share and
    gamma is absolute risk aversion in 1/USD. There is no permanent impact,
    spread, alpha, or participation constraint in this particular model.
    """

    impact_model = component(QuadraticImpact)
    risk_model = component(InventoryRisk)

    inventory = risk_model.inventory
    speed = impact_model.speed
    volume = impact_model.volume
    eta = impact_model.eta
    sigma = risk_model.sigma
    gamma = risk_model.gamma
    impact = impact_model.cost
    variance = risk_model.variance
    risk = risk_model.cost
    objective = impact + risk

    def lagrangian(self, time: Symbol, remaining: Expr) -> Expr:
        """Lift the resolved scalar model to q(t), with selling speed -q'(t)."""
        self._require_state_symbols()
        return self.objective.subs({self.inventory: remaining, self.speed: -diff(remaining, time)}, simultaneous=True)

    def _require_state_symbols(self) -> None:
        if not isinstance(self.inventory, Symbol) or not isinstance(self.speed, Symbol) or self.inventory == self.speed:
            raise ValueError("Keep inventory and speed unbound when transforming the density")

    def euler_equation(self, time: Symbol, remaining: Expr) -> Eq:
        """Differentiate the functional, rather than maintain a separate ODE."""
        return euler_equations(self.lagrangian(time, remaining), remaining, time)[0]

    def continuous_curve(self, initial: ScalarValue, horizon: ScalarValue) -> Eq:
        """Ask SymPy for the boundary-value solution where dsolve supports it."""
        end = sympify(horizon)
        if end.is_positive is not True or end.is_finite is False:
            raise ValueError("The liquidation horizon must be positive and finite")
        opening = sympify(initial)
        if opening.is_nonnegative is not True or opening.is_finite is False:
            raise ValueError("Opening liquidation inventory must be nonnegative")
        time = Symbol("t", real=True)
        remaining = Function("q")(time)
        return dsolve(
            self.euler_equation(time, remaining),
            remaining,
            ics={remaining.subs(time, 0): opening, remaining.subs(time, end): S.Zero},
        )

    def interval(
        self, opening: ScalarValue, closing: ScalarValue, duration: ScalarValue, *, volume: ScalarValue | None = None
    ) -> IntervalCost:
        """Integrate the same rates exactly along a linear inventory segment.

        Coefficients and market volume are constant within this interval. The
        piecewise-linear path is an approximation; its integral is exact.
        """
        self._require_state_symbols()
        left, right, width = map(sympify, (opening, closing, duration))
        if width.is_positive is not True or width.is_finite is False:
            raise ValueError("Interval duration must be positive and finite")
        elapsed = Symbol("s", nonnegative=True)
        path = left + (right - left) * elapsed / width
        substitutions = {
            self.inventory: path,
            self.speed: (left - right) / width,
        }
        if volume is not None:
            market_volume = sympify(volume)
            if market_volume.is_positive is not True or market_volume.is_finite is False:
                raise ValueError("Market volume must be positive and finite")
            if isinstance(self.volume, Symbol):
                substitutions[self.volume] = market_volume
            elif simplify(self.volume - market_volume) != 0:
                raise ValueError("Keep volume unbound to supply a different volume curve")

        def integral(expression: Expr) -> Expr:
            expression = expression.subs(substitutions, simultaneous=True)
            return simplify(integrate(expression, (elapsed, 0, width)))

        return IntervalCost(
            integral(self.impact), integral(self.variance), integral(self.risk), integral(self.objective)
        )

    def discretize(
        self,
        initial: ScalarValue,
        durations: tuple[ScalarValue, ...],
        *,
        volumes: tuple[ScalarValue, ...] | None = None,
    ) -> ExecutionGrid:
        """Restrict the functional to a mesh with fixed endpoints Q and zero."""
        return ExecutionGrid(
            self,
            sympify(initial),
            tuple(map(sympify, durations)),
            tuple(map(sympify, volumes)) if volumes is not None else (self.volume,) * len(durations),
        )

    def fit_impact(
        self, observations: tuple[ImpactObservation, ...], *, parameters: tuple[str, ...] = ("eta",)
    ) -> Calibration:
        """Fit interval impact costs, excluding price drift, spread, and fees."""
        residuals = tuple(
            self.interval(
                Rational(str(row.quantity)), S.Zero, Rational(str(row.duration)), volume=Rational(str(row.volume))
            ).impact
            - Rational(str(row.cost))
            for row in observations
        )
        return Calibration.fit(residuals, self.fit_parameters(parameters))

    def fit_parameters(self, names: tuple[str, ...]) -> NamedMap[Symbol]:
        """Require named, unbound parameters of this case, not derived fields."""
        fields = NamedMap((name, getattr(self, name)) for name in names)
        if not names or not all(isinstance(field, Symbol) for field in fields.values()):
            raise ValueError("Fitting needs named independent unknown parameters")
        return fields


class CrowdedImpactCost(QuadraticImpact):
    """Extend the shared impact law with a quartic participation cost."""

    quadratic = component(QuadraticImpact)
    speed = quadratic.speed
    volume = quadratic.volume
    eta = quadratic.eta
    beta = InstanceDummy("beta", nonnegative=True)
    cost = quadratic.cost + beta * quadratic.speed**4 / quadratic.volume**3


class CrowdedImpact(ExecutionDensity):
    """Replace the impact component; all continuous and discrete views follow."""

    impact_model = component(CrowdedImpactCost)
    beta = impact_model.beta


@dataclass(frozen=True)
class IntervalCost:
    """Expected impact in USD, price-risk variance in USD², and its USD penalty."""

    impact: Expr
    variance: Expr
    risk: Expr
    objective: Expr


@dataclass(frozen=True)
class ExecutionGrid:
    """A finite-dimensional restriction of the continuous functional."""

    model: ExecutionDensity
    initial: Expr
    durations: tuple[Expr, ...]
    volumes: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if not self.durations or len(self.durations) != len(self.volumes):
            raise ValueError("Supply one positive market volume per nonempty interval")
        if self.initial.is_nonnegative is not True or self.initial.is_finite is False:
            raise ValueError("Opening liquidation inventory must be nonnegative")
        if any(value.is_positive is not True or value.is_finite is False for value in (*self.durations, *self.volumes)):
            raise ValueError("Durations and market volume must be positive and finite")

    @cached_property
    def unknowns(self) -> tuple[Symbol, ...]:
        # Fresh symbols avoid coupling independently constructed execution grids.
        return tuple(InstanceDummy(f"q{i}", real=True) for i in range(1, len(self.durations)))

    @property
    def inventory(self) -> tuple[Expr, ...]:
        return (self.initial, *self.unknowns, S.Zero)

    @cached_property
    def intervals(self) -> tuple[IntervalCost, ...]:
        return tuple(
            self.model.interval(opening, closing, duration, volume=volume)
            for opening, closing, duration, volume in zip(
                self.inventory[:-1], self.inventory[1:], self.durations, self.volumes, strict=True
            )
        )

    @cached_property
    def objective(self) -> Expr:
        return sum((interval.objective for interval in self.intervals), S.Zero)

    @cached_property
    def gradient(self) -> tuple[Expr, ...]:
        return tuple(diff(self.objective, quantity) for quantity in self.unknowns)

    @cached_property
    def hessian(self) -> Matrix:
        return Matrix(self.gradient).jacobian(self.unknowns) if self.unknowns else Matrix.zeros(0, 0)

    def stationary(
        self, *, at: NamedMap[float] | None = None, guess: tuple[float, ...] | None = None
    ) -> ExecutionSolution:
        """Solve generated first-order conditions; this is not a constrained optimiser.

        Quadratic models use exact linear algebra. Nonlinear models require an
        explicit starting point for nsolve. Inspect residuals, curvature, and
        liquidation feasibility before interpreting a stationary point.
        """
        parameters = self.model.fit_parameters(tuple(at)) if at else NamedMap[Symbol]({})
        values_at = () if at is None else tuple((parameters[name], sympify(value)) for name, value in at.items())
        for parameter, value in values_at:
            if (
                value.is_finite is not True
                or value.is_real is not True
                or (parameter.is_positive and value.is_positive is not True)
                or (parameter.is_nonnegative and value.is_nonnegative is not True)
            ):
                raise ValueError("Parameter values must be finite and within their declared domains")
        equations = tuple(equation.subs(dict(values_at)) for equation in self.gradient)
        if not self.unknowns:
            return ExecutionSolution(self, (), values_at)
        try:
            matrix, rhs = linear_eq_to_matrix(equations, self.unknowns)
        except NonlinearError:
            if guess is None or len(guess) != len(self.unknowns):
                raise ValueError("Nonlinear stationarity needs one initial guess per interior inventory") from None
            values = tuple(nsolve(equations, self.unknowns, guess))
        else:
            values = tuple(map(simplify, matrix.LUsolve(rhs)))
        return ExecutionSolution(self, values, values_at)

    def fit_policy(
        self, inventory: tuple[ScalarValue, ...], *, parameters: tuple[str, ...] = ("gamma",)
    ) -> Calibration:
        """Fit first-order residuals assuming this mesh and an interior optimal policy."""
        observed = tuple(map(sympify, inventory))
        if (
            len(observed) != len(self.inventory)
            or simplify(observed[0] - self.initial).is_zero is not True
            or observed[-1].is_zero is not True
        ):
            raise ValueError("Observed policy must have the same mesh and fixed endpoints")
        if any((a - b).is_nonnegative is not True for a, b in zip(observed[:-1], observed[1:], strict=True)):
            raise ValueError("Observed liquidation inventories must be nonnegative and decreasing")
        replacements = dict(zip(self.unknowns, observed[1:-1], strict=True))
        return Calibration.fit(
            tuple(equation.subs(replacements) for equation in self.gradient), self.model.fit_parameters(parameters)
        )


@dataclass(frozen=True)
class ExecutionSolution:
    """Keep the objective and its derivation available after solving a grid."""

    grid: ExecutionGrid
    interior: tuple[Expr, ...]
    parameter_values: tuple[tuple[Symbol, Expr], ...] = ()

    @property
    def substitutions(self) -> dict[Symbol, Expr]:
        return dict((*self.parameter_values, *zip(self.grid.unknowns, self.interior, strict=True)))

    @property
    def inventory(self) -> tuple[Expr, ...]:
        return (self.grid.initial.subs(dict(self.parameter_values)), *self.interior, S.Zero)

    @property
    def trades(self) -> tuple[Expr, ...]:
        return tuple(simplify(a - b) for a, b in zip(self.inventory[:-1], self.inventory[1:], strict=True))

    @property
    def costs(self) -> IntervalCost:
        return IntervalCost(
            *(
                simplify(
                    sum((getattr(interval, field) for interval in self.grid.intervals), S.Zero).subs(self.substitutions)
                )
                for field in ("impact", "variance", "risk", "objective")
            )
        )

    @property
    def residuals(self) -> tuple[Expr, ...]:
        return tuple(simplify(equation.subs(self.substitutions)) for equation in self.grid.gradient)

    def sensitivity(self, parameter: Symbol) -> tuple[Expr, ...]:
        """Implicit differentiation of F(q, parameter)=0 at the stationary point."""
        if not self.grid.unknowns:
            return ()
        curvature = self.grid.hessian.subs(self.substitutions)
        forcing = Matrix([diff(equation, parameter).subs(self.substitutions) for equation in self.grid.gradient])
        return tuple(map(simplify, curvature.LUsolve(-forcing)))


def main() -> None:
    """Use synthetic trades with an explicitly unaffected reference price."""
    at = timestamp("2026-01-07T09:00:00Z")
    observations = tuple(
        ImpactObservation.from_fills(
            (Fill(f"training-{i}", "training", at, at, "msft", -quantity, price, 0),),
            reference_price=100,
            duration=1,
            volume=64,
        )
        for i, (quantity, price) in enumerate(((8, 99.75), (16, 99.5), (4, 99.875)))
    )
    fit = ExecutionDensity("impact").fit_impact(observations)
    model = ExecutionDensity("schedule", eta=fit.estimates["eta"], volume=100, sigma=3, gamma=Rational(1, 1000))
    schedule = model.discretize(10, (1, 1, 1, 1)).stationary()
    print("Fitted temporary impact eta:", fit.estimates["eta"])
    print("Remaining shares:", tuple(round(float(value), 4) for value in schedule.inventory))
    print(
        "Expected impact / CARA risk penalty (USD):",
        round(float(schedule.costs.impact), 4),
        round(float(schedule.costs.risk), 4),
    )
    preference = ExecutionDensity("preference", eta=fit.estimates["eta"], volume=100, sigma=3)
    recovered = preference.discretize(10, (1, 1, 1, 1)).fit_policy(schedule.inventory)
    print("Implied gamma under the optimal-policy assumption:", recovered.estimates["gamma"])


if __name__ == "__main__":
    main()

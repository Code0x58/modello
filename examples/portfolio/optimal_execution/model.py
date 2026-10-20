"""A small variational execution example, separate from account replay and policy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

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

from examples.compute.named_map import NamedMap
from examples.portfolio.optimal_execution.calibration import Calibration, ImpactObservation
from examples.symbolic import SymbolicModel
from modello import InstanceDummy, ScalarValue


class ExecutionDensity(SymbolicModel):
    """USD/hour impact and CARA risk penalty for a deterministic liquidation.

    Inventory is shares remaining; speed is shares sold per hour. Volume is
    market shares/hour; sigma is USD/share/sqrt(hour); eta is USD/share and
    gamma is absolute risk aversion in 1/USD. There is no permanent impact,
    spread, alpha, or participation constraint in this particular model.
    """

    inventory = InstanceDummy("inventory", real=True)
    speed = InstanceDummy("speed", real=True)
    volume = InstanceDummy("volume", positive=True)
    eta = InstanceDummy("eta", positive=True)
    sigma = InstanceDummy("sigma", positive=True)
    gamma = InstanceDummy("gamma", nonnegative=True)
    impact = eta * speed**2 / volume
    variance = sigma**2 * inventory**2
    risk = gamma * variance / 2
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


class CrowdedImpact(ExecutionDensity):
    """A smooth convex extension, quadratic plus quartic participation costs."""

    beta = InstanceDummy("beta", nonnegative=True)
    impact = (
        ExecutionDensity.eta * ExecutionDensity.speed**2 / ExecutionDensity.volume
        + beta * ExecutionDensity.speed**4 / ExecutionDensity.volume**3
    )


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

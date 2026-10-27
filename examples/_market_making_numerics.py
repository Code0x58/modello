"""Optional numerical methods; each result retains its research premises."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import exp, expm1, isfinite, log, log1p

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.special import expit
from sympy import Symbol, diff, exp as symbolic_exp, lambdify, simplify

from examples._market_making_models import exponential_requirement, solver_requirements
from examples._research_checks import require_parameter_assumptions
from modello import ConstraintKind
from modello.contrib.research import Method, ResearchCase, ResearchResult


def _number(case, path):
    value = case.expression(path).doit()
    if value.free_symbols or value.is_finite is not True or value.is_real is not True:
        raise ValueError(f"Bind {path} to a finite real value before numerical evaluation")
    return float(value)


@dataclass
class QuoteEvaluator:
    """Stable Hamiltonian evaluation after its uniqueness premises are checked."""

    family: str
    scale: float
    decay: float
    gamma: float
    step: float
    rate_function: Callable[[float], float] | None = None
    j_function: Callable[[float, float], float] | None = None
    evaluations: int = 0
    root_iterations: int = 0
    max_root_residual: float = 0

    @classmethod
    def from_case(cls, case, side):
        distance = case._reference(f"{side}.intensity.distance")
        rate = case.expression(f"{side}.intensity.rate", wrt=distance).doit()
        if rate.free_symbols - {distance}:
            raise ValueError("Bind all intensity coefficients before quote optimization")
        scale, decay = (_number(case, f"{side}.intensity.{name}") for name in ("scale", "decay"))
        gamma, step = _number(case, "gamma"), _number(case, "step")
        if simplify(rate - scale * symbolic_exp(-decay * distance)) == 0:
            family = "exponential"
        elif simplify(rate - scale / (1 + symbolic_exp(decay * distance))) == 0:
            family = "logistic"
        else:
            # Match using the exact coefficients, rather than rounded floats.
            a = case.expression(f"{side}.intensity.scale")
            k = case.expression(f"{side}.intensity.decay")
            if simplify(rate - a * symbolic_exp(-k * distance)) == 0:
                family = "exponential"
            elif simplify(rate - a / (1 + symbolic_exp(k * distance))) == 0:
                family = "logistic"
            else:
                family = "general"
        if min(scale, decay, gamma, step) <= 0:
            raise ValueError("Intensity coefficients, risk aversion, and lot size must be positive")
        evaluator = cls(family, scale, decay, gamma, step)
        if family == "general":
            from sympy import log as symbolic_log

            p = Symbol("p", real=True)
            j = distance - symbolic_log(1 - gamma * step * rate / diff(rate, distance)) / (gamma * step) - p
            evaluator.rate_function = lambdify(distance, rate, modules="math")
            evaluator.j_function = lambdify((distance, p), j, modules="math")
        return evaluator

    def root_equation(self, distance, cost):
        a = self.gamma * self.step
        if self.family == "exponential":
            return (distance - cost) - log1p(a / self.decay) / a
        if self.family == "logistic":
            log_gap = np.logaddexp(log1p(a / self.decay), log(a / self.decay) - self.decay * distance)
            return (distance - cost) - float(log_gap) / a
        assert self.j_function is not None
        return self.j_function(distance, cost)

    def evaluate(self, cost: float, *, force_root=False):
        if not isfinite(cost):
            raise ValueError("Continuation differences must be finite")
        if self.family == "exponential" and not force_root:
            distance = cost + log1p(self.gamma * self.step / self.decay) / (self.gamma * self.step)
        else:
            width = max(1 / self.decay, 1.0)
            for _ in range(64):
                upper = cost + width
                if self.root_equation(upper, cost) > 0:
                    break
                width *= 2
            else:
                raise ValueError(f"Could not bracket the quote for cost={cost}")
            root = root_scalar(
                lambda value: self.root_equation(value, cost),
                bracket=(cost, upper),
                method="brentq",
                xtol=1e-12,
                rtol=1e-12,
                maxiter=128,
            )
            if not root.converged:
                raise ValueError(f"Quote optimization failed for cost={cost}: {root.flag}")
            distance = float(root.root)
            self.root_iterations += root.iterations
        if distance <= cost:
            raise ValueError("Numerical precision cannot resolve a positive quote margin")
        if self.family == "exponential":
            rate = self.scale * exp(-self.decay * distance)
        elif self.family == "logistic":
            rate = self.scale * float(expit(-self.decay * distance))
        else:
            assert self.rate_function is not None
            rate = self.rate_function(distance)
        if not isfinite(rate) or rate <= 0:
            raise ValueError("The intensity overflowed, underflowed, or left its positive domain")
        a = self.gamma * self.step
        margin = distance - cost
        value = rate * (-expm1(-a * margin)) / self.gamma
        derivative = -self.step * rate * exp(-a * margin)
        residual = abs(self.root_equation(distance, cost))
        if not all(isfinite(item) for item in (distance, value, derivative, residual)):
            raise ValueError("Nonfinite Hamiltonian evaluation")
        self.evaluations += 1
        self.max_root_residual = max(self.max_root_residual, residual)
        return distance, value, derivative


@dataclass(frozen=True)
class QuoteSolution:
    """Physical-time outputs; None denotes an unavailable trade side."""

    times: tuple[float, ...]
    inventory: tuple[float, ...]
    theta: tuple[tuple[float, ...], ...]
    bid: tuple[tuple[float | None, ...], ...]
    ask: tuple[tuple[float | None, ...], ...]
    diagnostics: dict[str, object]


def _state_roles(case):
    names = (
        "inventory",
        "continuation.current",
        "continuation.after_bid",
        "continuation.after_ask",
        "continuation.time_derivative",
        "bid.value",
        "ask.value",
    )
    symbols = {case._reference(name) for name in names}
    if len(symbols) != len(names):
        raise ValueError("The ODE adapter requires independent state and Hamiltonian coordinates")
    varying = symbols | {
        case._reference(name) for name in ("terminal.inventory", "bid.intensity.distance", "ask.intensity.distance")
    }
    for constraint in case.problem.constraints:
        if constraint.left in symbols and constraint.kind is ConstraintKind.DEFINITION:
            raise ValueError("The ODE adapter requires independent state and Hamiltonian coordinates")
        if (
            constraint.kind is ConstraintKind.INPUT
            and (case._law(constraint.left).free_symbols | case._law(constraint.right).free_symbols) & varying
        ):
            raise ValueError("The full ODE study accepts parameter observations, not fixed state observations")
    require_parameter_assumptions(
        case,
        *varying,
        message="The full ODE study requires parameter assumptions; state coordinates vary on the grid "
        "and quote distances remain unrestricted",
    )


def solve_quotes(case: ResearchCase, *, rtol=1e-8, atol=1e-10, max_step=None, points=101) -> QuoteSolution:
    """Integrate in remaining time, using the declared HJB's RHS and gates."""
    _state_roles(case)
    horizon, capacity, step = (_number(case, name) for name in ("horizon", "capacity", "step"))
    ratio = case.expression("capacity") / case.expression("step")
    if ratio.is_integer is not True or ratio.is_positive is not True:
        raise ValueError("Capacity must be a positive integral number of lots")
    if type(points) is not int or points < 3 or rtol <= 0 or atol <= 0:
        raise ValueError("Use at least three output times and positive tolerances")
    max_step = horizon / 100 if max_step is None else max_step
    if not isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive and finite")
    count = int(ratio)
    grid = np.arange(-count, count + 1, dtype=float) * step
    grid[0], grid[-1] = -capacity, capacity
    q = case._reference("inventory")
    hb, ha = case._reference("bid.value"), case._reference("ask.value")
    theta_t = case._reference("continuation.time_derivative")
    residual = case.expression("residual", wrt=q).doit()
    if diff(residual, theta_t) != -1:
        raise ValueError("The declared HJB must be affine in theta_t with coefficient -1")
    rhs_expression = -residual.subs(theta_t, 0)
    terminal = case.expression("terminal_value", wrt=q).doit()
    if rhs_expression.free_symbols - {q, hb, ha} or terminal.free_symbols - {q}:
        raise ValueError("Bind all coefficients before integrating the HJB")
    rhs = lambdify((q, hb, ha), rhs_expression, modules="math")
    bid_gate = lambdify(q, diff(rhs_expression, hb), modules="math")
    ask_gate = lambdify(q, diff(rhs_expression, ha), modules="math")
    initial = np.array([float(terminal.subs(q, position)) for position in grid])
    bid, ask = QuoteEvaluator.from_case(case, "bid"), QuoteEvaluator.from_case(case, "ask")

    def evaluated(values):
        bid_values, ask_values = np.zeros(len(grid)), np.zeros(len(grid))
        bid_derivatives, ask_derivatives = np.zeros(len(grid)), np.zeros(len(grid))
        bid_quotes, ask_quotes = [None] * len(grid), [None] * len(grid)
        for i, position in enumerate(grid):
            if bid_gate(position):
                if i == len(grid) - 1:
                    raise ValueError("The declared bid gate enables a trade beyond the inventory boundary")
                bid_quotes[i], bid_values[i], bid_derivatives[i] = bid.evaluate((values[i] - values[i + 1]) / step)
            if ask_gate(position):
                if i == 0:
                    raise ValueError("The declared ask gate enables a trade beyond the inventory boundary")
                ask_quotes[i], ask_values[i], ask_derivatives[i] = ask.evaluate((values[i] - values[i - 1]) / step)
        return bid_values, ask_values, bid_derivatives, ask_derivatives, tuple(bid_quotes), tuple(ask_quotes)

    def f(time, values):
        b, a, _, _, _, _ = evaluated(values)
        return np.array([rhs(position, b[i], a[i]) for i, position in enumerate(grid)], dtype=float)

    def jac(time, values):
        _, _, b, a, _, _ = evaluated(values)
        return diags((-a[1:] / step, (b + a) / step, -b[:-1] / step), (-1, 0, 1), format="csc")

    solution = solve_ivp(
        f, (0, horizon), initial, method="Radau", jac=jac, rtol=rtol, atol=atol, max_step=max_step, dense_output=True
    )
    if not solution.success or solution.t[-1] != horizon:
        raise RuntimeError(f"HJB integration failed: {solution.message}")
    times = np.linspace(0, horizon, points)
    values = solution.sol(horizon - times).T
    bids, asks = [], []
    for row in values:
        _, _, _, _, b, a = evaluated(row)
        bids.append(b)
        asks.append(a)
    if not np.isfinite(values).all():
        raise ValueError("The value function contains nonfinite values")
    defects = []
    for divisor in (50, 100):
        h = horizon / divisor
        worst = 0.0
        for physical_time in np.linspace(horizon / 10, 9 * horizon / 10, 19):
            remaining = horizon - physical_time
            derivative = (solution.sol(remaining - h) - solution.sol(remaining + h)) / (2 * h)
            expected = -f(remaining, solution.sol(remaining))
            worst = max(worst, float(np.max(np.abs(derivative - expected) / (1 + np.abs(expected)))))
        defects.append(worst)
    diagnostics = {
        "method": "Radau",
        "rtol": rtol,
        "atol": atol,
        "max_step": max_step,
        "nfev": solution.nfev,
        "njev": solution.njev,
        "nlu": solution.nlu,
        "bid_family": bid.family,
        "ask_family": ask.family,
        "quote_evaluations": bid.evaluations + ask.evaluations,
        "root_iterations": bid.root_iterations + ask.root_iterations,
        "max_root_residual": max(bid.max_root_residual, ask.max_root_residual),
        "time_difference_defects": defects,
        "terminal_error": float(np.max(np.abs(values[-1] - initial))),
        "rhs": str(rhs_expression),
    }
    return QuoteSolution(
        tuple(map(float, times)),
        tuple(map(float, grid)),
        tuple(tuple(map(float, row)) for row in values),
        tuple(bids),
        tuple(asks),
        diagnostics,
    )


def run_quotes(case: ResearchCase, **settings) -> ResearchResult[QuoteSolution]:
    method = Method(
        "CARA inventory ODE",
        solve_quotes,
        solver_requirements(case),
        ("residual", "terminal_value", "bid.gain", "ask.gain"),
    )
    return case.run(method, **settings)


def exponential_reference(case: ResearchCase, times) -> np.ndarray:
    """Independent linear transformation, equation (3.13) in Guéant's paper."""

    def compute(current):
        _state_roles(current)
        capacity, step, horizon = (_number(current, name) for name in ("capacity", "step", "horizon"))
        ratio = current.expression("capacity") / current.expression("step")
        if ratio.is_integer is not True or ratio.is_positive is not True:
            raise ValueError("Capacity must be a positive integral number of lots")
        requested_times = tuple(float(time) for time in times)
        if not requested_times or any(not isfinite(time) or not 0 <= time <= horizon for time in requested_times):
            raise ValueError("Reference times must be finite and within the model horizon")
        k = _number(current, "bid.intensity.decay")
        if k != _number(current, "ask.intensity.decay"):
            raise ValueError("The linear reference requires equal exponential decay coefficients")
        n = int(ratio)
        grid = np.arange(-n, n + 1) * step
        grid[0], grid[-1] = -capacity, capacity
        q = current._reference("inventory")
        risk = current.expression("risk.cost", wrt=q).doit()
        terminal = current.expression("terminal_value", wrt=q).doit()
        if (risk.free_symbols | terminal.free_symbols) - {q}:
            raise ValueError("Bind the reference's risk and terminal coefficients")
        a = _number(current, "gamma") * step
        coefficient = exp((-1 - k / a) * log1p(a / k))
        matrix = diags(
            (
                np.full(2 * n, _number(current, "ask.intensity.scale") * coefficient),
                np.array([-k * float(risk.subs(q, position)) / step for position in grid]),
                np.full(2 * n, _number(current, "bid.intensity.scale") * coefficient),
            ),
            (-1, 0, 1),
            format="csc",
        )
        initial = np.array([exp(k * float(terminal.subs(q, position)) / step) for position in grid])
        rows = []
        for time in requested_times:
            value = expm_multiply((horizon - time) * matrix, initial)
            if np.any(value <= 0) or not np.isfinite(value).all():
                raise ValueError("The linear reference lost positivity or finiteness")
            rows.append(step / k * np.log(value))
        return np.array(rows)

    requirements = (*solver_requirements(case), exponential_requirement("bid"), exponential_requirement("ask"))
    return case.run(Method("Exponential linear reference", compute, requirements)).data

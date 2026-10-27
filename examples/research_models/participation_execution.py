"""Guéant's constant-volume, free-horizon participation-rate execution problem.

Run ``python -m examples.research_models.participation_execution`` for exact
solutions and checked certificates. Add ``--numerical`` for independent SciPy
minimization and quadrature. The symbolic example needs only SymPy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from sympy import (
    And,
    Derivative,
    Eq,
    Expr,
    Interval,
    Max,
    Or,
    S,
    Subs,
    diff,
    exp,
    integrate,
    limit,
    simplify,
)

from modello import ConstraintKind, Coordinate, InstanceDummy, ScalarValue, component
from modello.contrib.research import (
    Method,
    Requirement,
    ResearchCase,
    ResearchError,
    ResearchModel,
    ResearchReport,
    ResearchResult,
    Theorem,
    convex,
    exact_form,
    increasing,
    nonnegative,
    positive,
    predicate,
    requires,
    smooth,
)

SOURCE = "https://arxiv.org/html/1210.7608"
POSITIVE_RATES = Interval.open(0, S.Infinity)
_UNIQUE = Requirement("at_most_one_pov_minimum", "log_objective", "theorem")


def impact_requirements() -> tuple[Requirement, ...]:
    """Live execution-function hypotheses from Section 1 of the paper."""
    return tuple(
        replace(claim, consumer="pov_execution", reason=reason, source=SOURCE + "#S1")
        for claim, reason in (
            (positive("value", wrt="rate", domain=POSITIVE_RATES), "Execution costs are positive at positive rates."),
            (increasing("value", wrt="rate", domain=POSITIVE_RATES), "The execution function L is increasing."),
            (convex("value", wrt="rate", domain=POSITIVE_RATES), "The execution function L is convex."),
            (smooth("value", wrt="rate", domain=POSITIVE_RATES), "The scalar methods differentiate L on rho > 0."),
        )
    )


class PowerLawImpact(ResearchModel):
    """Instantaneous execution function L(rho) = eta rho**(1 + phi).

    ``eta`` is currency/share and ``phi`` is positive. ``rate`` is relative to
    market volume excluding the trader's own trades, and need not be below 1.
    """

    rate = Coordinate("rho", positive=True)
    eta = InstanceDummy("eta", positive=True)
    phi = InstanceDummy("phi", positive=True)
    value = requires(eta * rate ** (1 + phi), *impact_requirements())


class SurchargedPowerImpact(PowerLawImpact):
    """Add a quadratic execution-function term without retaining its shortcut."""

    surcharge = InstanceDummy("c", nonnegative=True)
    value = PowerLawImpact.eta * PowerLawImpact.rate ** (1 + PowerLawImpact.phi) + surcharge * PowerLawImpact.rate**2


class ConstantVolume(ResearchModel):
    """Constant, deterministic exogenous market volume in shares/time."""

    rate = InstanceDummy("V", positive=True)


class BrownianPriceRisk(ResearchModel):
    """Volatility in currency/share/sqrt(time); CARA gamma in 1/currency.

    ``gamma=0`` is the risk-neutral mean-cost extension. Expected exponential
    utility at gamma=0 is constant, so it is not the original utility problem.
    """

    sigma = InstanceDummy("sigma", positive=True)
    gamma = InstanceDummy("gamma", nonnegative=True)


class FreeParticipation(ResearchModel):
    """An open lower endpoint at zero and no participation cap or deadline.

    Inactive ``cap`` and ``deadline`` inputs remain available for replacements;
    they place no restriction on the feasible rates until enabled.
    """

    quantity = InstanceDummy("Q", positive=True)
    volume = InstanceDummy("V", positive=True)
    cap = InstanceDummy("rho_max", positive=True)
    deadline = InstanceDummy("T_max", positive=True)
    cap_active = S.Zero
    deadline_active = S.Zero
    lower = S.Zero


class CappedParticipation(FreeParticipation):
    """Add the closed upper bound rho <= cap."""

    cap_active = S.One


class DeadlineParticipation(FreeParticipation):
    """Add completion by T_max, equivalently rho >= Q/(V T_max)."""

    deadline_active = S.One
    lower = FreeParticipation.quantity / (FreeParticipation.volume * FreeParticipation.deadline)


class CappedDeadlineParticipation(DeadlineParticipation):
    """Apply both the participation cap and the completion deadline."""

    cap_active = S.One


class ParticipationExecution(ResearchModel):
    """A scalar search over constant participation, rather than arbitrary paths.

    Permanent impact and fixed per-share spread costs are omitted. The paper's
    linear permanent impact and fixed spread contribute schedule-independent
    constants; they would change total proceeds but not this minimizing rate.
    """

    impact = component(PowerLawImpact)
    market = component(ConstantVolume)
    risk = component(BrownianPriceRisk)
    limits = component(FreeParticipation)

    rate = PowerLawImpact.rate
    quantity = limits.quantity
    volume = market.rate
    volume = limits.volume
    eta = impact.eta
    phi = impact.phi
    gamma = risk.gamma
    sigma = risk.sigma
    time = Coordinate("t", real=True)
    log_rate = Coordinate("x", real=True)
    duration = quantity / (rate * volume)
    inventory = quantity - rate * volume * time
    temporary_cost = quantity * impact.value / rate
    risk_cost = gamma * sigma**2 * quantity**3 / (6 * volume * rate)
    objective = temporary_cost + risk_cost
    log_objective = Subs(objective, rate, exp(log_rate))
    log_curvature = Derivative(log_objective, log_rate, 2)


def uniqueness_theorem() -> Theorem:
    """Strict convexity in log-rate gives at most one attained scalar minimum.

    Existence is checked separately. The theorem does not claim convexity in
    rho, nor optimality among nonconstant participation schedules.
    """
    return Theorem(
        "strict_log_rate_convexity",
        _UNIQUE,
        (
            exact_form(
                "log_objective",
                lambda get: Subs(get("objective"), get("rate"), exp(get("log_rate"))),
                wrt="log_rate",
                name="actual_log_coordinate",
                consumer="pov_uniqueness",
                reason="The log-rate curve is the declared POV objective under rho=exp(x).",
            ),
            exact_form(
                ParticipationExecution.log_curvature
                - Derivative(ParticipationExecution.log_objective, ParticipationExecution.log_rate, 2, evaluate=False),
                S.Zero,
                wrt="log_rate",
                name="actual_log_curvature",
                consumer="pov_uniqueness",
                reason="The claimed curvature is the actual second derivative of the transformed objective.",
            ),
            smooth(
                "log_objective",
                wrt="log_rate",
                consumer="pov_uniqueness",
                reason="The transformed objective is C2 on the real line.",
            ),
            positive(
                "log_curvature",
                wrt="log_rate",
                consumer="pov_uniqueness",
                reason="An everywhere positive second derivative gives strict convexity in log-rate.",
            ),
        ),
        SOURCE + "#S2.SS1",
    )


def with_limits(
    case: ResearchCase[ParticipationExecution], *, cap: ScalarValue | None = None, deadline: ScalarValue | None = None
) -> ResearchCase[ParticipationExecution]:
    """Branch by adding or removing limits while retaining earlier observations.

    Passing ``None`` removes that constraint through component replacement.
    To change an already observed numeric bound, branch from the earlier case:
    observations accumulate rather than overwrite one another.
    """
    model: type[FreeParticipation]
    if cap is not None and deadline is not None:
        model = CappedDeadlineParticipation
    elif cap is not None:
        model = CappedParticipation
    elif deadline is not None:
        model = DeadlineParticipation
    else:
        model = FreeParticipation
    changed = case.replace("limits", model)
    values = {}
    if cap is not None:
        values["cap"] = cap
    if deadline is not None:
        values["deadline"] = deadline
    return changed.observe(limits=model("limits", **values)) if values else changed


def symbolic_case() -> ResearchCase[ParticipationExecution]:
    """Return an unobserved case for parameter assumptions and observations."""
    return ResearchCase.for_model(ParticipationExecution, "pov_execution").with_theorem(uniqueness_theorem())


def participation_case(
    *,
    quantity: ScalarValue = 3,
    volume: ScalarValue = 4,
    eta: ScalarValue = 1,
    phi: ScalarValue = S.Half,
    gamma: ScalarValue = S.One / 6,
    sigma: ScalarValue = 1,
    cap: ScalarValue | None = None,
    deadline: ScalarValue | None = None,
) -> ResearchCase[ParticipationExecution]:
    """Observe exact defaults with rho*=1/4, completion time 3, and cost 9/4."""
    case = symbolic_case()
    case = with_limits(case, cap=cap, deadline=deadline)
    return case.observe(quantity=quantity, volume=volume, eta=eta, phi=phi, gamma=gamma, sigma=sigma)


def _risk_integral(get: Callable[[str], Expr]) -> Expr:
    return (
        get("gamma")
        * get("sigma") ** 2
        * integrate(
            (get("quantity") - get("rate") * get("volume") * get("time")) ** 2,
            (get("time"), 0, get("quantity") / (get("rate") * get("volume"))),
        )
        / 2
    )


def _requirements(case: ResearchCase[ParticipationExecution], *, power_law: bool) -> tuple[Requirement, ...]:
    symbol = case.symbols
    requirements = [
        exact_form(
            "temporary_cost",
            lambda get: get("quantity") * get("impact.value") / get("rate"),
            wrt="rate",
            domain=POSITIVE_RATES,
            name="temporary_cost_identity",
            consumer="pov_execution",
            reason="The optional quadrature comparison uses the actual temporary execution cost.",
        ),
        exact_form(
            "duration",
            lambda get: get("quantity") / (get("rate") * get("volume")),
            wrt="rate",
            domain=POSITIVE_RATES,
            name="completion_time_identity",
            consumer="pov_execution",
            reason="The completion time is Q/(rho V).",
        ),
        exact_form(
            "inventory",
            lambda get: get("quantity") - get("rate") * get("volume") * get("time"),
            wrt="time",
            name="constant_pov_inventory_identity",
            consumer="pov_execution",
            reason="Inventory is Q-rho V t up to the completion time.",
        ),
        exact_form(
            "limits.lower",
            lambda get: get("limits.deadline_active") * get("quantity") / (get("volume") * get("limits.deadline")),
            name="deadline_lower_identity",
            consumer="pov_execution",
            reason="An enabled deadline gives precisely the corresponding lower participation bound.",
        ),
        exact_form(
            "objective",
            lambda get: (
                get("quantity") * get("impact.value") / get("rate")
                + get("gamma") * get("sigma") ** 2 * get("quantity") ** 3 / (6 * get("volume") * get("rate"))
            ),
            wrt="rate",
            domain=POSITIVE_RATES,
            name="pov_objective_identity",
            consumer="pov_execution",
            reason="Use the declared execution function and constant-volume Brownian inventory risk.",
            source=SOURCE + "#S2.SS1",
        ),
        exact_form(
            "risk_cost",
            _risk_integral,
            wrt="rate",
            domain=POSITIVE_RATES,
            name="inventory_risk_integral",
            consumer="pov_execution",
            reason="The risk term equals gamma sigma²/2 times the integrated squared inventory.",
            source=SOURCE + "#S2.SS1",
        ),
        exact_form(
            "log_objective",
            lambda get: Subs(get("objective"), get("rate"), exp(get("log_rate"))),
            wrt="log_rate",
            name="log_coordinate_identity",
            consumer="pov_execution",
            reason="The strictly convex curve must be the actual objective under rho=exp(x).",
        ),
        replace(
            _UNIQUE,
            consumer="pov_execution",
            reason="Strict log-rate convexity permits at most one attained POV minimum.",
        ),
        predicate(
            And(
                Or(Eq(symbol.limits.cap_active, 0), Eq(symbol.limits.cap_active, 1)),
                Or(Eq(symbol.limits.deadline_active, 0), Eq(symbol.limits.deadline_active, 1)),
                Or(Eq(symbol.limits.cap_active, 0), symbol.limits.lower <= symbol.limits.cap),
            ),
            name="feasible_limits",
            consumer="pov_execution",
            reason="Enabled bounds must form a nonempty positive-rate interval.",
        ),
    ]
    for path in ("quantity", "volume", "eta", "phi", "sigma"):
        requirements.append(
            positive(path, name=f"positive_{path}", consumer="pov_execution", reason=f"{path} must be positive.")
        )
    requirements.append(
        nonnegative(
            "gamma", name="nonnegative_risk_aversion", consumer="pov_execution", reason="Risk aversion is nonnegative."
        )
    )
    if power_law:
        requirements.append(
            exact_form(
                "impact.value",
                lambda get: get("eta") * get("rate") ** (1 + get("phi")),
                wrt="rate",
                domain=POSITIVE_RATES,
                name="pure_power_law",
                consumer="analytic_pov",
                reason="Guéant's Proposition 4 shortcut needs this exact power law.",
                source=SOURCE + "#S2.SS1",
            )
        )
    return tuple(requirements)


@dataclass(frozen=True)
class ParticipationSolution:
    """Exact scalar solution and a separate report checking its certificate."""

    status: Literal["minimum", "infimum_not_attained"]
    rate: Expr | None
    duration: Expr | None
    cost: Expr
    unconstrained_rate: Expr | None
    active_bounds: tuple[str, ...]
    lower_multiplier: Expr | None
    upper_multiplier: Expr | None
    certificate: ResearchReport


def _bound_data(case: ResearchCase[ParticipationExecution]) -> tuple[Expr, Expr | None]:
    cap_active, deadline_active = (case.expression(f"limits.{name}") for name in ("cap_active", "deadline_active"))
    if cap_active not in (S.Zero, S.One) or deadline_active not in (S.Zero, S.One):
        raise ValueError("The supported limit switches must be exactly zero or one")
    lower = case.expression("limits.lower")
    if deadline_active == 0 and lower != 0 or deadline_active == 1 and lower.is_positive is not True:
        raise ValueError("The active deadline must give a positive lower rate; an inactive one must give zero")
    return lower, case.expression("limits.cap") if cap_active == 1 else None


def _checked_report(
    case: ResearchCase[ParticipationExecution], requirements: tuple[Requirement, ...]
) -> ResearchReport:
    report = case.check(max_steps=1000, requirements=requirements)
    failed = [item.name for item in requirements if report.explain(item.name).condition is not S.true]
    if report.consistent is False or failed:
        raise ResearchError("Unestablished POV certificate: " + ", ".join(failed), case, report)
    return report


def _independent_rate(case: ResearchCase[ParticipationExecution]) -> None:
    rate = case.expression("rate", wrt="rate")
    if any(rate in case.expression(condition).free_symbols for condition in case.assumptions):
        raise ValueError("Optimize over the positive rate interval; encode caps and deadlines with with_limits")
    for constraint in case.problem.constraints:
        if constraint.kind is not ConstraintKind.INPUT:
            continue
        expressions = (case.expression(value, wrt="rate") for value in (constraint.left, constraint.right))
        if any(rate in value.free_symbols for value in expressions):
            raise ValueError("Optimize over an independent rate; observe parameters rather than its derived outputs")


def _risk_regime(case: ResearchCase[ParticipationExecution]) -> Literal["positive", "zero"]:
    gamma = case.expression("gamma")
    if gamma.is_positive is True:
        return "positive"
    if gamma.is_zero is True:
        return "zero"
    requirements = (
        predicate(
            gamma > 0, name="risk_positive_branch", consumer="risk_regime", reason="Check the positive-risk branch."
        ),
        predicate(
            Eq(gamma, 0), name="risk_zero_branch", consumer="risk_regime", reason="Check the risk-neutral branch."
        ),
    )
    report = case.check(max_steps=1000, requirements=requirements)
    if report.explain("risk_positive_branch").condition is S.true:
        return "positive"
    if report.explain("risk_zero_branch").condition is S.true:
        return "zero"
    raise ValueError("Resolve whether gamma is positive or zero before selecting an optimizer")


def _analytic(case: ResearchCase[ParticipationExecution]) -> ParticipationSolution:
    _independent_rate(case)
    lower, upper = _bound_data(case)
    rate = case.expression("rate", wrt="rate")
    objective = case.expression("objective", wrt="rate")
    quantity, volume, eta, phi, gamma, sigma = (
        case.expression(path) for path in ("quantity", "volume", "eta", "phi", "gamma", "sigma")
    )
    coefficient = gamma * sigma**2 * quantity**3 / (6 * volume)
    regime = _risk_regime(case)
    scaled_slope = simplify(rate**2 * diff(objective, rate))
    common = (
        exact_form(
            scaled_slope,
            eta * quantity * phi * rate ** (phi + 1) - coefficient,
            wrt="rate",
            domain=POSITIVE_RATES,
            name="scaled_first_order_identity",
            consumer="analytic_certificate",
            reason="rho² J'(rho) has one strictly increasing power term and a constant risk term.",
        ),
        positive(
            diff(scaled_slope, rate),
            wrt="rate",
            domain=POSITIVE_RATES,
            name="scaled_slope_strictly_increasing",
            consumer="analytic_certificate",
            reason="The scaled slope crosses zero at most once even when J is not convex in rho.",
        ),
    )
    if regime == "zero" and lower == 0:
        requirements: tuple[Requirement, ...] = (
            *common,
            positive(
                objective,
                wrt="rate",
                domain=POSITIVE_RATES,
                name="cost_above_infimum",
                consumer="analytic_certificate",
                reason="Every positive rate has strictly positive cost.",
            ),
            exact_form(
                limit(objective, rate, 0, dir="+"),
                S.Zero,
                name="unattained_zero_limit",
                consumer="analytic_certificate",
                reason="The infimum is zero as rho decreases to the excluded endpoint.",
            ),
        )
        return ParticipationSolution(
            "infimum_not_attained", None, None, S.Zero, None, (), None, None, _checked_report(case, requirements)
        )
    if regime == "positive":
        free_rate = simplify((gamma * sigma**2 * quantity**2 / (6 * eta * phi * volume)) ** (1 / (1 + phi)))
        candidate = free_rate if lower == 0 else Max(lower, free_rate)
        if upper is not None:
            candidate = -Max(-candidate, -upper)
        candidate = simplify(candidate)
    else:
        free_rate, candidate = None, lower
    slope = simplify(diff(objective, rate).subs(rate, candidate))
    at_lower, at_upper = candidate == lower, upper is not None and candidate == upper
    lower_multiplier = Max(slope, 0) if at_lower else S.Zero
    upper_multiplier = Max(-slope, 0) if at_upper else S.Zero
    constraints = [
        candidate > 0,
        candidate >= lower,
        lower_multiplier >= 0,
        upper_multiplier >= 0,
        Eq(slope - lower_multiplier + upper_multiplier, 0),
        Eq(lower_multiplier * (candidate - lower), 0),
    ]
    if upper is not None:
        constraints.extend((candidate <= upper, Eq(upper_multiplier * (upper - candidate), 0)))
    duration = simplify(quantity / (volume * candidate))
    requirements = (
        *common,
        predicate(
            And(*constraints),
            name="pov_kkt_certificate",
            consumer="analytic_certificate",
            reason="Check primal feasibility, nonnegative multipliers, stationarity, and complementary slackness.",
        ),
        exact_form(
            case.expression("inventory", wrt="time").subs(
                {rate: candidate, case.expression("time", wrt="time"): duration}, simultaneous=True
            ),
            S.Zero,
            name="completion_boundary",
            consumer="analytic_certificate",
            reason="The chosen rate liquidates all shares at its completion time.",
        ),
        replace(
            _UNIQUE,
            consumer="analytic_certificate",
            reason="Strict log-rate convexity makes the certified scalar minimum unique.",
        ),
    )
    active = tuple(name for name, enabled in (("deadline", at_lower), ("cap", at_upper)) if enabled)
    return ParticipationSolution(
        "minimum",
        candidate,
        duration,
        simplify(objective.subs(rate, candidate)),
        free_rate,
        active,
        lower_multiplier,
        upper_multiplier,
        _checked_report(case, requirements),
    )


def analytic_method(case: ResearchCase[ParticipationExecution]) -> Method[ParticipationSolution]:
    """The pure-power shortcut, guarded by live identities and shape evidence."""
    return Method(
        "Guéant analytic POV",
        _analytic,
        _requirements(case, power_law=True),
        uses=("objective", "duration", "inventory", "log_objective", "log_curvature", "limits.lower"),
    )


@dataclass(frozen=True)
class NumericalParticipation:
    """Independent numerical observations, without a proof of an error bound."""

    status: Literal["minimum", "infimum_not_attained"]
    rate: float | None
    duration: float | None
    cost: float
    log_bracket: tuple[float, float] | None
    diagnostics: dict[str, float | int | str]


def _numerical(case: ResearchCase[ParticipationExecution], *, xatol: float = 1e-12) -> NumericalParticipation:
    import math

    from scipy.integrate import quad
    from scipy.optimize import minimize_scalar
    from sympy import lambdify

    _independent_rate(case)
    if not math.isfinite(xatol) or xatol <= 0:
        raise ValueError("xatol must be finite and positive")
    lower, upper = _bound_data(case)
    if lower.free_symbols or upper is not None and upper.free_symbols:
        raise ValueError("Observe the active participation cap and deadline before numerical minimization")
    x = case.expression("log_rate", wrt="log_rate")
    expression = case.expression("log_objective", wrt="log_rate").doit()
    if expression.free_symbols - {x}:
        raise ValueError("Observe all parameters before numerical minimization")
    value, derivative = (lambdify(x, formula, "math") for formula in (expression, diff(expression, x)))
    if case.expression("gamma") == 0 and lower == 0:
        _checked_report(
            case,
            (
                positive(
                    "objective",
                    wrt="rate",
                    domain=POSITIVE_RATES,
                    name="numerical_positive_cost",
                    consumer="numerical_certificate",
                    reason="A zero limit is unattained at every positive rate.",
                ),
                exact_form(
                    limit(expression, x, S.NegativeInfinity),
                    S.Zero,
                    name="numerical_zero_limit",
                    consumer="numerical_certificate",
                    reason="Check the open-endpoint limit before reporting nonattainment.",
                ),
            ),
        )
        return NumericalParticipation(
            "infimum_not_attained",
            None,
            None,
            0.0,
            None,
            {"kind": "symbolic endpoint classification; no numerical optimizer"},
        )
    left = math.log(float(lower)) if lower > 0 else -1.0
    right = math.log(float(upper)) if upper is not None else max(1.0, left + 1.0)
    if lower == 0:
        left = min(left, right - 1)
    if lower == 0:
        for _ in range(128):
            if derivative(left) < 0:
                break
            left -= 1.0
        else:
            raise ValueError("No lower sign bracket was found")
    if upper is None:
        for _ in range(128):
            if derivative(right) > 0:
                break
            right += 1.0
        else:
            raise ValueError("No upper sign bracket was found")
    iterations = 0
    if left == right or lower > 0 and derivative(left) >= 0:
        optimum = left
    elif upper is not None and derivative(right) <= 0:
        optimum = right
    else:
        result = minimize_scalar(value, bounds=(left, right), method="bounded", options={"xatol": xatol})
        if not result.success:
            raise ValueError(f"Scalar minimization failed: {result.message}")
        optimum, iterations = float(result.x), int(result.nfev)
    rho = (
        float(lower)
        if optimum == left and lower > 0
        else (float(upper) if optimum == right and upper is not None else math.exp(optimum))
    )
    cost = float(value(optimum))
    quantity, volume, gamma, sigma = (float(case.expression(path)) for path in ("quantity", "volume", "gamma", "sigma"))
    duration = quantity / (volume * rho)
    variance_integral, quad_error = quad(
        lambda time: (quantity - rho * volume * time) ** 2, 0, duration, epsabs=1e-11, epsrel=1e-11
    )
    temporary = float(case.expression("temporary_cost", wrt="rate").subs(case.expression("rate", wrt="rate"), rho))
    integral_cost = temporary + gamma * sigma**2 * variance_integral / 2
    log_slope = float(derivative(optimum))
    violation = (
        0.0
        if left == right
        else max(-log_slope, 0)
        if optimum == left and lower > 0
        else (max(log_slope, 0) if optimum == right and upper is not None else abs(log_slope))
    )
    diagnostics: dict[str, float | int | str] = {
        "kind": "numerical observation",
        "objective_evaluations": iterations,
        "log_kkt_violation": violation,
        "quadrature_objective_difference": abs(integral_cost - cost),
        "quadrature_reported_error": quad_error,
        "xatol": xatol,
    }
    if not all(math.isfinite(item) for item in (rho, cost, duration, violation, integral_cost)):
        raise ValueError("The numerical optimizer produced nonfinite outputs")
    return NumericalParticipation("minimum", rho, duration, cost, (left, right), diagnostics)


def numerical_method(case: ResearchCase[ParticipationExecution]) -> Method[NumericalParticipation]:
    """Minimize the actual log-rate objective; no pure-power formula is used."""
    return Method(
        "independent scalar POV minimization",
        _numerical,
        _requirements(case, power_law=False),
        uses=("objective", "duration", "log_objective", "log_curvature", "limits.lower"),
    )


def compare(*, numerical: bool = False) -> dict[str, dict[str, object]]:
    """Return reproducible branches, results, reports, and blocked-method evidence."""
    free = participation_case()
    both = with_limits(free, cap=S.Half, deadline=2)
    surcharge = free.replace("impact", SurchargedPowerImpact).observe(
        impact=SurchargedPowerImpact("surcharge", surcharge=S.One / 4)
    )
    cases = {
        "free": free,
        "capped": with_limits(free, cap=S.One / 8),
        "deadline": with_limits(free, deadline=2),
        "both": both,
        "cap_removed": with_limits(both, deadline=2),
        "deadline_removed": with_limits(both, cap=S.Half),
        "risk_neutral_free": participation_case(gamma=0),
        "risk_neutral_deadline": participation_case(gamma=0, deadline=2),
        "infeasible": with_limits(free, cap=S.One / 8, deadline=2),
        "surcharge_replacement": surcharge,
    }
    results: dict[str, dict[str, object]] = {}
    for name, case in cases.items():
        row: dict[str, object] = {"case": case}
        for key, method in (("analytic", analytic_method), ("numerical", numerical_method)):
            if key == "numerical" and not numerical:
                continue
            try:
                result: ResearchResult = case.run(method)
                row[key] = result
            except ResearchError as error:
                row[f"{key}_blocked"] = str(error)
                row[f"{key}_report"] = error.report
        results[name] = row
    return results


def main() -> None:
    """Print exact examples and optionally their numerical comparisons."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical", action="store_true", help="Run independent SciPy minimization and quadrature")
    arguments = parser.parse_args()
    for name, row in compare(numerical=arguments.numerical).items():
        print(name)
        for key in ("analytic", "numerical"):
            result = row.get(key)
            if isinstance(result, ResearchResult):
                data = result.data
                print(f"  {key}: {data.status}; rate={data.rate}; duration={data.duration}; cost={data.cost}")
            elif f"{key}_blocked" in row:
                print(f"  {key}: {row[f'{key}_blocked']}")


if __name__ == "__main__":
    main()

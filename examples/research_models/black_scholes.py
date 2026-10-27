"""European options with live Black--Scholes--Merton research contracts.

Run ``python -m examples.research_models.black_scholes`` to replay the study.
Only the numerical reference imports SciPy, and it does so inside its method.
The CAS identities do not prove a stochastic replication or uniqueness theorem.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from sympy import Abs, Expr, Interval, Rational, S, Subs, Symbol, diff, erf, exp, limit, log, pi, simplify, sqrt

from modello import Coordinate, InstanceDummy, component
from modello.contrib.research import (
    Method,
    Requirement,
    ResearchCase,
    ResearchError,
    ResearchModel,
    ResearchResult,
    Theorem,
    exact_form,
    nonnegative,
    positive,
    predicate,
    requires,
    smooth,
)

BSM_SOURCE = "https://www.columbia.edu/~mh2078/FoundationsFE/BlackScholes.pdf"
CDF_SOURCE = "https://dlmf.nist.gov/7.2#i"
SPOT_DOMAIN = Interval.open(0, S.Infinity)
TIME_DOMAIN = Interval.Ropen(0, 1)
SCHEDULE_DOMAIN = Interval(0, 1)
Get = Callable[[str], Expr]
_TAIL_FIELDS = (("rate_total", "rate"), ("yield_total", "yield_rate"), ("variance_total", "variance_rate"))
_PRICING_USES = (
    "pricing.call",
    "pricing.put",
    "pricing.d1",
    "pricing.d2",
    "pricing.discounted_spot",
    "pricing.discounted_strike",
    "pricing.call_minus_put",
    "pricing.call_delta",
    "pricing.put_delta",
    "pricing.gamma",
)


def normal_cdf(value: Expr) -> Expr:
    """The standard Gaussian CDF, represented exactly through ``erf``."""
    return (1 + erf(value / sqrt(2))) / 2


def normal_density(value: Expr) -> Expr:
    return exp(-(value**2) / 2) / sqrt(2 * pi)


def _d1(get: Get) -> Expr:
    return (
        log(get("spot") / get("strike"))
        + get("carry.rate_total")
        - get("carry.yield_total")
        + get("carry.variance_total") / 2
    ) / sqrt(get("carry.variance_total"))


def _d2(get: Get) -> Expr:
    return _d1(get) - sqrt(get("carry.variance_total"))


def _discounted_spot(get: Get) -> Expr:
    return get("spot") * exp(-get("carry.yield_total"))


def _discounted_strike(get: Get) -> Expr:
    return get("strike") * exp(-get("carry.rate_total"))


def _call_put_difference(get: Get) -> Expr:
    return get("call") - get("put")


def _helper_contract(name: str, expected: Callable[[Get], Expr]) -> Requirement:
    return exact_form(
        name,
        expected,
        wrt="spot",
        domain=SPOT_DOMAIN,
        name=f"{name}_source",
        consumer="interior_valuation",
        reason="The exposed helper agrees with its primitive inputs or the actual declared call and put.",
        source=BSM_SOURCE,
    )


def _helper_contracts() -> tuple[Requirement, ...]:
    return tuple(
        _helper_contract(name, formula)
        for name, formula in (
            ("d1", _d1),
            ("d2", _d2),
            ("discounted_spot", _discounted_spot),
            ("discounted_strike", _discounted_strike),
        )
    )


def _option_formula(get: Get, *, put: bool = False) -> Expr:
    first = _d1(get)
    second = first - sqrt(get("carry.variance_total"))
    stock = get("spot") * exp(-get("carry.yield_total"))
    bond = get("strike") * exp(-get("carry.rate_total"))
    return (
        bond * normal_cdf(-second) - stock * normal_cdf(-first)
        if put
        else (stock * normal_cdf(first) - bond * normal_cdf(second))
    )


def _call_formula(get: Get) -> Expr:
    return _option_formula(get)


def _put_formula(get: Get) -> Expr:
    return _option_formula(get, put=True)


def _call_delta_formula(get: Get) -> Expr:
    return exp(-get("carry.yield_total")) * normal_cdf(_d1(get))


def _put_delta_formula(get: Get) -> Expr:
    return _call_delta_formula(get) - exp(-get("carry.yield_total"))


def _gamma_formula(get: Get) -> Expr:
    return exp(-get("carry.yield_total")) * normal_density(_d1(get)) / (get("spot") * sqrt(get("carry.variance_total")))


def _greek_contract(name: str, expected: Callable[[Get], Expr]) -> Requirement:
    return exact_form(
        name,
        expected,
        wrt="spot",
        domain=SPOT_DOMAIN,
        name=f"{name}_form",
        consumer="interior_valuation",
        reason=(
            "The numerical adapter uses this exact Gaussian Greek; the symbolic method checks its derivative identity."
        ),
        source=BSM_SOURCE,
    )


def _formula_contract(option: str) -> Requirement:
    return exact_form(
        option,
        _put_formula if option == "put" else _call_formula,
        wrt="spot",
        domain=SPOT_DOMAIN,
        name=f"{option}_form",
        consumer="interior_valuation",
        reason="The calculation consumes the declared integrated Gaussian option formula.",
        source=BSM_SOURCE,
    )


def _smooth_contract(option: str) -> Requirement:
    return smooth(
        option,
        wrt="spot",
        domain=SPOT_DOMAIN,
        name=f"{option}_c2",
        consumer="interior_valuation",
        reason="Spot differentiation applies only inside the positive-variance, positive-spot domain.",
        source=CDF_SOURCE,
    )


def _interior_contracts(option: str) -> tuple[Requirement, ...]:
    return (
        positive("strike", consumer="interior_valuation", reason="A strictly positive strike keeps log(S/K) real."),
        positive(
            "carry.variance_total",
            name="positive_variance",
            consumer="interior_valuation",
            reason="The interior formula divides by the square root of strictly positive total variance.",
        ),
        _formula_contract(option),
        _smooth_contract(option),
    )


class IntegratedCarry(ResearchModel):
    """Scalar integrals R=integral(r), Q=integral(q), W=integral(sigma squared)."""

    rate_total = InstanceDummy("R", real=True)
    yield_total = InstanceDummy("Q", real=True)
    variance_total = InstanceDummy("W", real=True)


class EuropeanOptions(ResearchModel):
    """A European call and put sharing strike and deterministic integrated carry.

    Inputs are real at definition time; positive strike and variance are live
    requirements. A claim about spot quantifies S in (0, infinity), even when
    ``spot`` has been observed at a particular value.
    """

    spot = InstanceDummy("S", real=True)
    strike = InstanceDummy("K", real=True)
    carry = component(IntegratedCarry)
    d1 = requires(
        (log(spot / strike) + carry.rate_total - carry.yield_total + carry.variance_total / 2)
        / sqrt(carry.variance_total),
        _helper_contract("d1", _d1),
    )
    d2 = requires(d1 - sqrt(carry.variance_total), _helper_contract("d2", _d2))
    discounted_spot = requires(spot * exp(-carry.yield_total), _helper_contract("discounted_spot", _discounted_spot))
    discounted_strike = requires(
        strike * exp(-carry.rate_total), _helper_contract("discounted_strike", _discounted_strike)
    )
    call = requires(discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2), *_interior_contracts("call"))
    put = requires(discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1), *_interior_contracts("put"))
    call_delta = requires(exp(-carry.yield_total) * normal_cdf(d1), _greek_contract("call_delta", _call_delta_formula))
    put_delta = requires(
        exp(-carry.yield_total) * (normal_cdf(d1) - 1), _greek_contract("put_delta", _put_delta_formula)
    )
    gamma = requires(
        exp(-carry.yield_total) * normal_density(d1) / (spot * sqrt(carry.variance_total)),
        _greek_contract("gamma", _gamma_formula),
    )
    call_minus_put = requires(
        call - put,
        _helper_contract("call_minus_put", _call_put_difference),
        exact_form(
            "call_minus_put",
            lambda get: _discounted_spot(get) - _discounted_strike(get),
            wrt="spot",
            domain=SPOT_DOMAIN,
            name="put_call_parity",
            consumer="interior_valuation",
            reason="Parity discounts stock by cumulative dividend yield and strike by cumulative interest.",
            source=BSM_SOURCE,
        ),
    )


class ConstantSchedule(ResearchModel):
    """Deterministic coefficients on the normalized year 0 <= t <= T=1.

    Slopes are zero definitions in this model. Replacing it with
    ``AffineSchedule`` turns them into independent parameters and refreshes all
    dependent expressions and requirements.
    """

    time = Coordinate("t", real=True)
    horizon = S.One
    rate_intercept = InstanceDummy("r0", real=True)
    yield_intercept = InstanceDummy("q0", real=True)
    variance_intercept = InstanceDummy("v0", real=True)
    rate_slope = S.Zero
    yield_slope = S.Zero
    variance_slope = S.Zero
    rate = rate_intercept + rate_slope * time
    yield_rate = yield_intercept + yield_slope * time
    variance_rate = variance_intercept + variance_slope * time
    rate_total = rate_intercept * (horizon - time) + rate_slope * (horizon**2 - time**2) / 2
    yield_total = yield_intercept * (horizon - time) + yield_slope * (horizon**2 - time**2) / 2
    variance_total = requires(
        variance_intercept * (horizon - time) + variance_slope * (horizon**2 - time**2) / 2,
        positive(
            "variance_total",
            wrt="time",
            domain=TIME_DOMAIN,
            name="interior_variance",
            consumer="deterministic_schedule",
            reason="Total variance must be positive before maturity; t=1 is deliberately excluded.",
        ),
    )


class AffineSchedule(ConstantSchedule):
    """The exact integrals of r0+r1*t, q0+q1*t, and v0+v1*t."""

    rate_slope = InstanceDummy("r1", real=True)
    yield_slope = InstanceDummy("q1", real=True)
    variance_slope = InstanceDummy("v1", real=True)


class OptionStudy(ResearchModel):
    """A scalar pricing snapshot and an independently checked time schedule.

    The snapshot is not silently joined to a time coordinate. Guarded methods
    check that its three totals equal the schedule integrals at ``now``.
    """

    pricing = component(EuropeanOptions)
    schedule = component(ConstantSchedule)
    now = InstanceDummy("t0", real=True)
    # Scalar aliases let research capture even singular exploratory inputs
    # without first constructing (and solving) an invalid option instance.
    spot = pricing.spot
    strike = pricing.strike
    rate_total = pricing.carry.rate_total
    yield_total = pricing.carry.yield_total
    variance_total = pricing.carry.variance_total


class PayoffBoundary(ResearchModel):
    """The positive-part payoff expressed in discounted forward moneyness x."""

    moneyness = Coordinate("x", real=True)
    kink = Abs(moneyness)
    call = (moneyness + kink) / 2
    put = (-moneyness + kink) / 2


def _cdf_theorems() -> tuple[Theorem, ...]:
    """Trust only erf composition after checking the exact finite formula.

    This bridges a specific ``continuous_domain(erf(...))`` limitation. It is
    not a theorem about arbitrary unknown functions or an option pricing rule.
    """
    elementary = tuple(
        smooth(
            path,
            wrt="spot",
            domain=SPOT_DOMAIN,
            name=f"smooth_{path}",
            consumer="Gaussian_composition",
            reason="The Gaussian argument and its elementary multiplicative factors are C2.",
        )
        for path in ("d1", "d2", "discounted_spot", "discounted_strike")
    )
    return tuple(
        Theorem(
            f"{option}_Gaussian_composition",
            _smooth_contract(option),
            (*_interior_contracts(option)[:2], _formula_contract(option), *_helper_contracts(), *elementary),
            CDF_SOURCE,
            "1",
        )
        for option in ("call", "put")
    )


def symbolic_case() -> ResearchCase[EuropeanOptions]:
    """An unobserved interior branch, keeping all scalar parameters symbolic."""
    case = ResearchCase.for_model(EuropeanOptions, "integrated_bsm")
    case = case.assume(case.symbols.spot > 0, case.symbols.strike > 0, case.symbols.carry.variance_total > 0)
    for theorem in _cdf_theorems():
        case = case.with_theorem(theorem)
    return case


def study_case(
    *,
    schedule: bool = False,
    dividends: bool = True,
    now: Any = 0,
    spot: Any = 100,
    strike: Any = 100,
    variance: Any = Rational(1, 25),
) -> ResearchCase[OptionStudy]:
    """Construct exact data at S=K=100, with constant or affine coefficients.

    The changing schedule is r=.04+.01*t, q=.01+.005*t,
    sigma_squared=.04+.02*t. ``dividends=False`` sets both yield terms to zero.
    ``variance`` sets the instantaneous variance intercept. Defaults use exact
    rationals, including scalar integrated totals; use rational overrides for
    an exact symbolic study.
    """
    case = ResearchCase.for_model(OptionStudy, "bsm_study")
    if schedule:
        case = case.replace("schedule", AffineSchedule)
    when = S(now)
    r0, q0, v0 = Rational(1, 25), Rational(1, 100) if dividends else S.Zero, S(variance)
    r1 = Rational(1, 100) if schedule else S.Zero
    q1 = Rational(1, 200) if schedule and dividends else S.Zero
    v1 = Rational(1, 50) if schedule else S.Zero
    remaining = 1 - when
    tail = (1 - when**2) / 2
    schedule_type = AffineSchedule if schedule else ConstantSchedule
    observations = dict(rate_intercept=r0, yield_intercept=q0, variance_intercept=v0)
    if schedule:
        observations.update(rate_slope=r1, yield_slope=q1, variance_slope=v1)
    case = case.observe(
        now=when,
        spot=spot,
        strike=strike,
        rate_total=r0 * remaining + r1 * tail,
        yield_total=q0 * remaining + q1 * tail,
        variance_total=v0 * remaining + v1 * tail,
        schedule=schedule_type("schedule", **observations),
    )
    for theorem in _cdf_theorems():
        case = case.with_theorem(theorem)
    return case


def _snapshot_requirements(case: ResearchCase[OptionStudy]) -> tuple[Requirement, ...]:
    time = case.expression("schedule.time")
    requirements = [
        positive(
            "pricing.spot",
            name="positive_spot",
            consumer="BSM_snapshot",
            reason="The observed stock price is positive.",
        ),
        predicate(
            OptionStudy.now >= 0,
            name="nonnegative_time",
            consumer="BSM_snapshot",
            reason="The snapshot is inside the declared calendar interval.",
        ),
        predicate(
            OptionStudy.now < 1,
            name="before_maturity",
            consumer="BSM_snapshot",
            reason="The positive-variance formula excludes maturity.",
        ),
        nonnegative(
            "schedule.variance_rate",
            wrt="schedule.time",
            domain=SCHEDULE_DOMAIN,
            name="variance_schedule",
            consumer="BSM_snapshot",
            reason="An instantaneous variance schedule cannot be negative.",
        ),
    ]
    for name in ("rate_total", "yield_total", "variance_total"):
        # Evaluate the live schedule at this case's state. The method's identity
        # guard prevents these exact snapshot expectations crossing branches.
        expected = case.expression(f"schedule.{name}", wrt=time).subs(time, case.expression("now"))
        requirements.append(
            exact_form(
                f"pricing.carry.{name}",
                expected,
                name=f"snapshot_{name}",
                consumer="BSM_snapshot",
                reason=(
                    "The scalar pricing input equals the independently declared "
                    "schedule integral at the observation time."
                ),
            )
        )
    return tuple(requirements)


def _schedule_requirements(case: ResearchCase[OptionStudy]) -> tuple[Requirement, ...]:
    """Identify genuine C1 tail integrals, rather than arbitrary antiderivatives.

    C1 regularity on the closed year is a sufficient FTC scope. The derivative
    relations on the pricing interior and the zero terminal values then fix
    the integration constants for all three deterministic coefficient curves.
    """
    time = case.expression("schedule.time")
    requirements = [
        exact_form(
            "schedule.horizon",
            S.One,
            name="normalized_horizon",
            consumer="deterministic_tail_integrals",
            reason="This adapter quantifies schedules on the normalized year ending at T=1.",
        )
    ]
    for total, local in _TAIL_FIELDS:
        for path in (total, local):
            requirements.append(
                smooth(
                    f"schedule.{path}",
                    wrt="schedule.time",
                    order=1,
                    domain=SCHEDULE_DOMAIN,
                    name=f"schedule_{path}_c1",
                    consumer="deterministic_tail_integrals",
                    reason="C1 regularity on the closed calendar interval supplies a sufficient FTC scope.",
                )
            )
        requirements.extend(
            (
                exact_form(
                    diff(case.expression(f"schedule.{total}", wrt=time), time),
                    -case.expression(f"schedule.{local}", wrt=time),
                    wrt=time,
                    domain=TIME_DOMAIN,
                    name=f"chain_{total}",
                    consumer="deterministic_tail_integrals",
                    reason=(
                        "Differentiate the live deterministic tail integral "
                        "to obtain minus its instantaneous coefficient."
                    ),
                ),
                exact_form(
                    Subs(getattr(case.symbols.schedule, total), time, S.One),
                    S.Zero,
                    name=f"terminal_{total}",
                    consumer="deterministic_tail_integrals",
                    reason=(
                        "A tail integral vanishes at maturity; "
                        "its derivative alone does not fix the integration constant."
                    ),
                ),
            )
        )
    return tuple(requirements)


@dataclass(frozen=True)
class SymbolicChecks:
    """Exact zero residuals, with separate spot and time quantifiers."""

    residuals: Mapping[str, Expr]
    gamma_positive: bool
    spot_domain: Interval = SPOT_DOMAIN
    time_domain: Interval = TIME_DOMAIN
    interpretation: str = "Exact CAS identities and the deterministic chain rule; no stochastic replication proof."


@lru_cache(maxsize=1)
def _integrated_identities() -> tuple[tuple[str, Expr], ...]:
    """Differentiate before reducing; do not expand a huge scheduled tree."""
    spot, strike, variance = (Symbol(name, positive=True) for name in ("S", "K", "W"))
    rate, dividend = (Symbol(name, real=True) for name in ("R", "Q"))
    local_rate, local_yield, local_variance = (Symbol(name, real=True) for name in ("r", "q", "v"))
    values = dict(
        spot=spot,
        strike=strike,
        **{"carry.rate_total": rate, "carry.yield_total": dividend, "carry.variance_total": variance},
    )
    get = values.__getitem__
    first, second = _d1(get), _d1(get) - sqrt(variance)
    call, put = _call_formula(get), _put_formula(get)
    density = normal_density(first)
    delta = exp(-dividend) * normal_cdf(first)
    gamma = exp(-dividend) * density / (spot * sqrt(variance))
    residuals = {
        "integrated_density_balance": spot * exp(-dividend) * density - strike * exp(-rate) * normal_density(second),
        "integrated_put_call_parity": call - put - spot * exp(-dividend) + strike * exp(-rate),
        "integrated_call_delta": diff(call, spot) - delta,
        "integrated_put_delta": diff(put, spot) - delta + exp(-dividend),
        "integrated_call_gamma": diff(call, spot, 2) - gamma,
        "integrated_put_gamma": diff(put, spot, 2) - gamma,
    }
    for option, value in (("call", call), ("put", put)):
        residuals[f"integrated_{option}_pde"] = (
            -local_rate * diff(value, rate)
            - local_yield * diff(value, dividend)
            - local_variance * diff(value, variance)
            + (local_rate - local_yield) * spot * diff(value, spot)
            + local_variance * spot**2 * diff(value, spot, 2) / 2
            - local_rate * value
        )
    return tuple((name, simplify(value)) for name, value in residuals.items())


def symbolic_method(case: ResearchCase[OptionStudy]) -> Method[SymbolicChecks]:
    """Check this branch's actual spot derivatives and deterministic schedule.

    Pass the factory to ``case.run`` to build it for the current branch. The
    integrated PDE audit keeps R, Q and W independent until applying the checked
    schedule chain rule. No derivative tree is checked at module import.
    """
    spot = case.symbols.pricing.spot
    requirements = (*_snapshot_requirements(case), *_schedule_requirements(case))
    time = case.expression("schedule.time")

    def execute(current: ResearchCase[OptionStudy]) -> SymbolicChecks:
        residuals = dict(_integrated_identities())
        for total, local in _TAIL_FIELDS:
            residuals[f"schedule_{total}_chain"] = simplify(
                diff(current.expression(f"schedule.{total}", wrt=time), time)
                + current.expression(f"schedule.{local}", wrt=time)
            )
            residuals[f"schedule_{total}_terminal"] = simplify(
                current.expression(f"schedule.{total}", wrt=time).subs(time, 1)
            )
        for option in ("call", "put"):
            # Restrict a real scalar input to the explicitly quantified positive
            # spot domain. This also discharges retained Ne(S, 0) division guards
            # before differentiation; it does not assume anything at S=0.
            interior = Symbol("S_interior", positive=True)
            curve = current.expression(f"pricing.{option}", wrt=spot).subs(spot, interior)
            residuals[f"snapshot_{option}_delta"] = simplify(
                diff(curve, interior) - current.expression(f"pricing.{option}_delta", wrt=spot).subs(spot, interior)
            )
            residuals[f"snapshot_{option}_gamma"] = simplify(
                diff(curve, interior, 2) - current.expression("pricing.gamma", wrt=spot).subs(spot, interior)
            )
        gamma_positive = current.expression("pricing.gamma", wrt=spot).subs(spot, interior).is_positive is True
        if any(value != 0 for value in residuals.values()):
            raise ValueError("An exact symbolic identity did not reduce to zero")
        if not gamma_positive:
            raise ValueError("The exact interior gamma positivity check did not succeed")
        return SymbolicChecks(MappingProxyType(residuals), gamma_positive)

    return Method(
        "BSM_identities",
        execute,
        requirements,
        (*_PRICING_USES, "schedule.variance_total"),
    ).bind(case)


@dataclass(frozen=True)
class NumericalCheck:
    call: float
    put: float
    call_delta: float
    put_delta: float
    gamma: float
    quadrature_call: float
    quadrature_put: float
    call_error: float
    put_error: float
    quadrature_error_estimates: tuple[float, float]
    kind: str = "numerical observation"


def valuation_method(case: ResearchCase[OptionStudy]) -> Method[NumericalCheck]:
    """Compare the declared formula with discounted lognormal payoff quadrature."""

    def execute(current: ResearchCase[OptionStudy]) -> NumericalCheck:
        try:
            from scipy.integrate import quad
        except ImportError as error:
            raise RuntimeError("Install modello[research] to run the independent SciPy payoff quadrature") from error

        def number(path: str) -> float:
            expression = current.expression(path)
            if expression.free_symbols or expression.is_real is not True or expression.is_finite is not True:
                raise ValueError(f"{path} needs a finite real scalar observation for numerical valuation")
            value = float(expression.evalf())
            if not math.isfinite(value):
                raise ValueError(f"{path} must be a finite real scalar")
            return value

        spot, strike = number("pricing.spot"), number("pricing.strike")
        rate, dividend, variance = (number(f"pricing.carry.{name}_total") for name in ("rate", "yield", "variance"))
        deviation = math.sqrt(variance)
        threshold = (math.log(strike / spot) - rate + dividend + variance / 2) / deviation

        def integrand(z: float) -> float:
            # Combine exponentials with the Gaussian density before evaluation,
            # avoiding overflow at the infinite quadrature endpoints.
            return (
                spot * math.exp(rate - dividend - variance / 2 + deviation * z - z * z / 2)
                - strike * math.exp(-z * z / 2)
            ) / math.sqrt(2 * math.pi)

        call_integral, call_estimate = quad(integrand, threshold, math.inf, epsabs=1e-10, epsrel=1e-10)
        put_integral, put_estimate = quad(lambda z: -integrand(z), -math.inf, threshold, epsabs=1e-10, epsrel=1e-10)
        reference_call, reference_put = math.exp(-rate) * call_integral, math.exp(-rate) * put_integral
        call, put = number("pricing.call"), number("pricing.put")
        return NumericalCheck(
            call,
            put,
            number("pricing.call_delta"),
            number("pricing.put_delta"),
            number("pricing.gamma"),
            reference_call,
            reference_put,
            abs(call - reference_call),
            abs(put - reference_put),
            (math.exp(-rate) * call_estimate, math.exp(-rate) * put_estimate),
        )

    return Method(
        "BSM_snapshot",
        execute,
        (*_snapshot_requirements(case), *_schedule_requirements(case)),
        (*_PRICING_USES, "schedule.variance_total"),
    ).bind(case)


def dividend_free_method() -> Method[Expr]:
    """The old shortcut is intentionally blocked once cumulative yield changes."""
    requirement = exact_form(
        "pricing.call_minus_put",
        lambda get: get("pricing.spot") - get("pricing.strike") * exp(-get("pricing.carry.rate_total")),
        wrt="pricing.spot",
        domain=SPOT_DOMAIN,
        name="dividend_free_parity",
        consumer="dividend_free_shortcut",
        reason="Replacing discounted stock by spot requires zero cumulative dividend yield.",
        source=BSM_SOURCE,
    )
    return Method(
        "dividend_free_shortcut",
        lambda current: current.expression("pricing.call_minus_put", wrt="pricing.spot"),
        (
            positive(
                "pricing.spot",
                name="positive_spot",
                consumer="dividend_free_shortcut",
                reason="The observed stock price is positive.",
            ),
            requirement,
        ),
        ("pricing.call_minus_put",),
    )


def boundary_checks() -> dict[str, Expr]:
    """Compute unequal one-sided payoff slopes at the zero-variance kink.

    At maturity, x=S-K. With zero variance before maturity,
    x=S*exp(-Q)-K*exp(-R), the discounted deterministic forward moneyness.
    These boundary payoffs are not obtained by substituting W=0 into d1/d2.
    """
    case = ResearchCase.for_model(PayoffBoundary)
    x = PayoffBoundary.moneyness
    values = {}
    for option in ("call", "put"):
        derivative = diff(case.expression(option, wrt=x), x)
        values[f"{option}_left_slope"] = limit(derivative, x, 0, dir="-")
        values[f"{option}_right_slope"] = limit(derivative, x, 0, dir="+")
    return values


def run_demo(*, numerical: bool = True) -> dict[str, Any]:
    """Replay branches and retain raw cases, reports, and method results."""
    no_dividends = study_case(dividends=False)
    dividends = study_case()
    changing = study_case(schedule=True, now=Rational(1, 4))
    symbolic: dict[str, ResearchResult[SymbolicChecks]] = {}
    numeric: dict[str, ResearchResult[NumericalCheck]] = {}
    for name, case in (("constant", dividends), ("affine", changing)):
        symbolic[name] = case.run(symbolic_method)
        if numerical:
            numeric[name] = case.run(valuation_method)
    shortcut = no_dividends.run(dividend_free_method())
    try:
        dividends.run(dividend_free_method())
    except ResearchError as error:
        rejected_shortcut = error.report
    else:
        raise RuntimeError("The dividend-free shortcut unexpectedly survived a positive dividend yield")
    rejected_interior = {}
    for name, case in (
        ("maturity", study_case(now=1)),
        ("zero_variance", study_case(variance=0)),
        ("negative_variance", study_case(variance=-Rational(1, 25))),
    ):
        try:
            case.run(valuation_method)
        except ResearchError as error:
            rejected_interior[name] = {"case": case, "report": error.report}
        else:
            raise RuntimeError(f"The interior formula unexpectedly accepted {name}")
    return {
        "cases": {"no_dividends": no_dividends, "constant": dividends, "affine": changing},
        "symbolic": symbolic,
        "numerical": numeric,
        "dividend_free_shortcut": shortcut,
        "rejected_dividend_free_shortcut": rejected_shortcut,
        "rejected_interior": rejected_interior,
        "boundary": boundary_checks(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic-only", action="store_true", help="Skip the optional SciPy payoff quadrature.")
    arguments = parser.parse_args(argv)
    demo = run_demo(numerical=not arguments.symbolic_only)
    for name, result in demo["symbolic"].items():
        counts: dict[str, int] = {}
        for evidence in result.report.checks:
            counts[evidence.verdict] = counts.get(evidence.verdict, 0) + 1
        print(f"{name}: {len(result.data.residuals)} exact zero residuals; report {counts}")
    for name, result in demo["numerical"].items():
        data = result.data
        print(
            f"{name}: call={data.call:.10f}, put={data.put:.10f}, delta={data.call_delta:.10f}, gamma={data.gamma:.10f}"
        )
        print(f"  discounted payoff quadrature differences: call={data.call_error:.3g}, put={data.put_error:.3g}")
    print(
        "Dividend-free shortcut after adding yield:",
        demo["rejected_dividend_free_shortcut"].explain("dividend_free_parity").verdict,
    )
    print("Boundary one-sided slopes:", demo["boundary"])
    print("Interior method rejected:", ", ".join(demo["rejected_interior"]))
    print("Exact CAS identities and numerical observations do not establish stochastic replication or uniqueness.")


if __name__ == "__main__":
    main()

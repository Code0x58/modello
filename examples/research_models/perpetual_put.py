"""A perpetual American put with exact boundary and obstacle certificates.

Run ``python -m examples.research_models.perpetual_put`` for symbolic checks.
Add ``--numerical`` to optimize discounted hitting payoffs over stopping
barriers independently. Stochastic verification is an explicitly trusted rule,
separate from the scalar algebra and the optional numerical observations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from sympy import Derivative, Expr, Interval, Piecewise, S, Subs, diff, exp, limit, simplify

from examples._research_checks import require_parameter_assumptions
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
    exact_form,
    nonnegative,
    positive,
    predicate,
    smooth,
)

SOURCE = "https://finance.martinsewell.com/option-pricing/Merton1973.pdf#page=35"
POSITIVE = Interval.open(0, S.Infinity)
EXERCISE = Interval.Lopen(0, 1)
CONTINUATION = Interval.open(1, S.Infinity)
Get = Callable[[str], Expr]

TRUSTED_STOPPING_ASSUMPTIONS = (
    "Under a risk-neutral probability Q, W is a standard Brownian motion on a filtered probability space "
    "with the usual conditions, and dS_t=r*S_t*dt+sigma*S_t*dW_t for all t>=0, with S_0>0.",
    "The constants r, sigma and K are strictly positive, dividends are zero, and exercise has no fees or delays.",
    "The objective is sup_tau E_Q[exp(-r*tau)*max(K-S_tau,0)] over all stopping times of this filtration "
    "with values in [0,infinity], including immediate exercise; the discounted payoff at an infinite stopping time "
    "is defined as zero.",
    "The trusted verification rule uses generalized Ito and localized optional sampling for a C1, "
    "piecewise C2 candidate with locally bounded second derivatives. Smooth fit removes its local-time term.",
    "The checked bounds 0<=P<=K give integrable, uniformly bounded stopped discounted candidates. "
    "The checked envelope K*exp(-r*t)->0 supplies transversality on the event of no exercise.",
)
_OBSTACLE = nonnegative("continuation_gap", wrt="moneyness", domain=CONTINUATION, name="continuation_obstacle")
_OPTIMAL = Requirement("optimal_stopping_under_GBM", "value", "theorem")


class ConstantBSMMarket(ResearchModel):
    """The constant-coefficient BSM economic setting, without a stopping theorem.

    Parameters are real so singular branches can be inspected. The methods
    require positive r and sigma and zero dividends before using beta.
    """

    rate = InstanceDummy("r", real=True)
    volatility = InstanceDummy("sigma", real=True)
    dividend = InstanceDummy("q", real=True)
    beta = -2 * rate / volatility**2


class PutContract(ResearchModel):
    """A perpetual right to receive max(K-S, 0) upon exercise."""

    strike = InstanceDummy("K", real=True)


class PerpetualPut(ResearchModel):
    """Exercise and continuation branches on normalized moneyness x=S/B."""

    market = component(ConstantBSMMarket)
    contract = component(PutContract)
    rate = market.rate
    volatility = market.volatility
    dividend = market.dividend
    beta = market.beta
    strike = contract.strike
    initial_spot = InstanceDummy("S0", real=True)
    moneyness = Coordinate("x", positive=True)
    time = Coordinate("t", real=True)
    boundary = 2 * rate * strike / (2 * rate + volatility**2)
    spot = boundary * moneyness
    exercise = strike - spot
    continuation = (strike - boundary) * moneyness**beta
    continuation_gap = continuation - exercise
    continuation_slope = Derivative(continuation, moneyness)
    continuation_curvature = Derivative(continuation, moneyness, 2)
    continuation_residual = (
        volatility**2 * moneyness**2 * Derivative(continuation, moneyness, 2) / 2
        + (rate - dividend) * moneyness * Derivative(continuation, moneyness)
        - rate * continuation
    )
    exercise_residual = (
        volatility**2 * moneyness**2 * Derivative(exercise, moneyness, 2) / 2
        + (rate - dividend) * moneyness * Derivative(exercise, moneyness)
        - rate * exercise
    )
    value = Piecewise((exercise, moneyness <= 1), (continuation, True))
    terminal_envelope = strike * exp(-rate * time)


def _generator_formula(branch: str) -> Callable[[Get], Expr]:
    def formula(get: Get) -> Expr:
        x, value = get("moneyness"), get(branch)
        return (
            get("volatility") ** 2 * x**2 * Derivative(value, x, 2, evaluate=False) / 2
            + (get("rate") - get("dividend")) * x * Derivative(value, x, evaluate=False)
            - get("rate") * value
        )

    return formula


def _source_forms() -> tuple[Requirement, ...]:
    """Protect intermediate formulas as well as the final piecewise value."""
    formulas: tuple[tuple[str, Callable[[Get], Expr], str | None], ...] = (
        ("beta", lambda get: -2 * get("rate") / get("volatility") ** 2, None),
        ("boundary", lambda get: get("beta") * get("strike") / (get("beta") - 1), None),
        ("spot", lambda get: get("boundary") * get("moneyness"), "moneyness"),
        ("exercise", lambda get: get("strike") - get("spot"), "moneyness"),
        ("continuation", lambda get: (get("strike") - get("boundary")) * get("moneyness") ** get("beta"), "moneyness"),
        ("continuation_gap", lambda get: get("continuation") - get("exercise"), "moneyness"),
        (
            "continuation_slope",
            lambda get: Derivative(get("continuation"), get("moneyness"), evaluate=False),
            "moneyness",
        ),
        (
            "continuation_curvature",
            lambda get: Derivative(get("continuation"), get("moneyness"), 2, evaluate=False),
            "moneyness",
        ),
        (
            "value",
            lambda get: Piecewise((get("exercise"), get("moneyness") <= 1), (get("continuation"), True)),
            "moneyness",
        ),
        ("terminal_envelope", lambda get: get("strike") * exp(-get("rate") * get("time")), "time"),
    )
    requirements = [
        exact_form(
            path,
            formula,
            wrt=coordinate,
            domain=POSITIVE if coordinate == "moneyness" else S.Reals,
            name=f"form_{path}",
            consumer="perpetual_put_algebra",
            reason="The computation uses this live declaration, including its intermediate formulas.",
            source=SOURCE,
        )
        for path, formula, coordinate in formulas
    ]
    for branch in ("continuation", "exercise"):
        requirements.append(
            exact_form(
                f"{branch}_residual",
                _generator_formula(branch),
                wrt="moneyness",
                domain=POSITIVE,
                name=f"form_{branch}_generator",
                consumer="perpetual_put_algebra",
                reason="The residual is the risk-neutral GBM generator minus discounting.",
                source=SOURCE,
            )
        )
    return tuple(requirements)


def _boundary_identity(order: int, name: str) -> Requirement:
    difference = PerpetualPut.continuation - PerpetualPut.exercise
    if order:
        difference = Derivative(difference, PerpetualPut.moneyness, order, evaluate=False)
    return exact_form(Subs(difference, PerpetualPut.moneyness, 1), S.Zero, name=name)


def obstacle_theorem() -> Theorem:
    """A scalar convex-tangent lemma, separate from stochastic verification."""
    return Theorem(
        "convex_tangent_obstacle",
        _OBSTACLE,
        (
            smooth(
                "continuation_gap",
                wrt="moneyness",
                domain=Interval(1, S.Infinity),
                consumer="convex_tangent",
                reason="The obstacle gap is C2 up to x=1.",
            ),
            nonnegative(
                "continuation_curvature",
                wrt="moneyness",
                domain=CONTINUATION,
                consumer="convex_tangent",
                reason="A nonnegative second derivative makes the continuation branch convex.",
            ),
            replace(
                _boundary_identity(0, "tangent_value"),
                consumer="convex_tangent",
                reason="The obstacle gap vanishes at x=1.",
            ),
            replace(
                _boundary_identity(1, "tangent_slope"),
                consumer="convex_tangent",
                reason="The obstacle gap has zero first derivative at x=1.",
            ),
            *_source_forms(),
        ),
        "Convex tangent inequality: h(1)=h'(1)=0 and h''>=0 on [1,infinity) imply h>=0 there.",
    )


def _requirements(case: ResearchCase[PerpetualPut]) -> tuple[Requirement, ...]:
    symbol = case.symbols
    requirements = [*_source_forms()]
    for path in ("rate", "volatility", "strike"):
        requirements.append(
            positive(
                path,
                name=f"positive_{path}",
                consumer="perpetual_put_algebra",
                reason=f"The positive-parameter perpetual-put formula requires {path}>0.",
                source=SOURCE,
            )
        )
    requirements.extend(
        (
            exact_form(
                "dividend",
                S.Zero,
                name="zero_dividends",
                consumer="perpetual_put_algebra",
                reason="The exponent and boundary used here assume zero dividends.",
                source=SOURCE,
            ),
            predicate(
                symbol.beta < 0,
                name="negative_exponent",
                consumer="perpetual_put_algebra",
                reason="The decaying solution uses the negative characteristic root.",
                source=SOURCE,
            ),
            positive(
                "boundary",
                name="positive_boundary",
                consumer="perpetual_put_algebra",
                reason="The normalization S/B needs B>0.",
            ),
            positive(
                symbol.strike - symbol.boundary,
                name="boundary_below_strike",
                consumer="perpetual_put_algebra",
                reason="The exercise boundary lies strictly below the strike.",
            ),
            replace(
                _boundary_identity(0, "boundary_value_matching"),
                consumer="perpetual_put_algebra",
                reason="Exercise and continuation values match at x=1.",
            ),
            replace(
                _boundary_identity(1, "boundary_smooth_fit"),
                consumer="perpetual_put_algebra",
                reason="First spot derivatives match because B is positive.",
            ),
            smooth(
                "exercise",
                wrt="moneyness",
                domain=POSITIVE,
                name="exercise_branch_c2",
                consumer="perpetual_put_algebra",
                reason="Each analytic branch is C2 separately.",
            ),
            smooth(
                "continuation",
                wrt="moneyness",
                domain=POSITIVE,
                name="continuation_branch_c2",
                consumer="perpetual_put_algebra",
                reason="The power branch is C2 at positive moneyness.",
            ),
            exact_form(
                "continuation_residual",
                S.Zero,
                wrt="moneyness",
                domain=CONTINUATION,
                name="continuation_pricing_generator",
                consumer="perpetual_put_algebra",
                reason="The discounted pricing-generator identity holds in the continuation region.",
                source=SOURCE,
            ),
            nonnegative(
                -symbol.exercise_residual,
                wrt="moneyness",
                domain=EXERCISE,
                name="exercise_generator_nonpositive",
                consumer="perpetual_put_algebra",
                reason="The discounted exercise-branch generator is -rK<=0.",
            ),
            nonnegative(
                "exercise",
                wrt="moneyness",
                domain=EXERCISE,
                name="exercise_nonnegative",
                consumer="perpetual_put_algebra",
                reason="Exercise equals the nonnegative put payoff in its region.",
            ),
            nonnegative(
                symbol.strike - symbol.exercise,
                wrt="moneyness",
                domain=EXERCISE,
                name="exercise_below_strike",
                consumer="perpetual_put_algebra",
                reason="The exercise value is bounded by K.",
            ),
            positive(
                "continuation",
                wrt="moneyness",
                domain=CONTINUATION,
                name="continuation_positive",
                consumer="perpetual_put_algebra",
                reason="The continuation value dominates the zero part of the payoff.",
            ),
            nonnegative(
                symbol.strike - symbol.continuation,
                wrt="moneyness",
                domain=CONTINUATION,
                name="continuation_below_strike",
                consumer="perpetual_put_algebra",
                reason="The continuation value is bounded by K.",
            ),
            replace(
                _OBSTACLE, consumer="perpetual_put_algebra", reason="Convexity and smooth fit give P>=K-S above B."
            ),
            Requirement(
                "decay_at_infinity",
                "continuation",
                "limit",
                "moneyness",
                POSITIVE,
                expected=S.Zero,
                consumer="perpetual_put_algebra",
                reason="The continuation value vanishes as S tends to infinity.",
                source=SOURCE,
            ),
            Requirement(
                "transversality_envelope",
                "terminal_envelope",
                "limit",
                "time",
                expected=S.Zero,
                consumer="perpetual_put_algebra",
                reason="K exp(-rt) tends to zero; bounded candidate values obey this envelope.",
            ),
        )
    )
    return tuple(requirements)


def symbolic_case() -> ResearchCase[PerpetualPut]:
    """Prepare unobserved scalar formulas, trusting only the convex-tangent lemma."""
    return ResearchCase.for_model(PerpetualPut, "perpetual_put").with_theorem(obstacle_theorem())


def perpetual_put_case(
    *,
    rate: ScalarValue = S.One / 25,
    volatility: ScalarValue = S.One / 5,
    strike: ScalarValue = 100,
    dividend: ScalarValue = 0,
    initial_spot: ScalarValue = 100,
) -> ResearchCase[PerpetualPut]:
    """Exact defaults have beta=-2, boundary 200/3, and value 400/27 at S0=100."""
    return symbolic_case().observe(
        rate=rate, volatility=volatility, strike=strike, dividend=dividend, initial_spot=initial_spot
    )


def trust_stopping_theorem(case: ResearchCase[PerpetualPut]) -> ResearchCase[PerpetualPut]:
    """Explicitly adopt the stated GBM optimal-stopping verification rule.

    The scalar checker checks the rule's analytic premises. It does not prove
    the Brownian model, generalized Ito, or optional sampling. The full trusted
    setting is exported in ``TRUSTED_STOPPING_ASSUMPTIONS`` and the result.
    """
    return case.with_theorem(
        Theorem(
            "trusted_GBM_perpetual_put_verification",
            _OPTIMAL,
            _requirements(case),
            SOURCE,
            "1",
        )
    )


def _independent_curve(case: ResearchCase[PerpetualPut]) -> None:
    coordinates = {case._reference(name) for name in ("moneyness", "time")}
    require_parameter_assumptions(
        case,
        *coordinates,
        message="The verification traverses both regions and an unbounded time horizon; "
        "do not restrict its coordinates",
    )
    for constraint in case.problem.constraints:
        if constraint.kind is ConstraintKind.INPUT:
            if any(
                case.expression(value, wrt="moneyness").free_symbols & coordinates
                for value in (constraint.left, constraint.right)
            ):
                raise ValueError("Observe market parameters and initial_spot rather than derived curve values")


def regularity_report(case: ResearchCase[PerpetualPut]) -> ResearchReport:
    """Keep the generic global C1/C2 attempts, including unresolved outcomes."""
    return case.check(
        max_steps=1000,
        requirements=tuple(
            smooth(
                "value",
                wrt="moneyness",
                domain=POSITIVE,
                order=order,
                name=f"global_c{order}",
                consumer="global_regularity_diagnostic",
                reason="Inspect the actual piecewise curve; no theorem overrides this attempt.",
            )
            for order in (1, 2)
        ),
    )


@dataclass(frozen=True)
class PutCertificate:
    """Exact one-sided spot derivatives and scalar verification evidence."""

    beta: Expr
    boundary: Expr
    value_at_initial_spot: Expr
    one_sided: dict[str, Expr]
    second_derivative_jump: Expr
    regularity: ResearchReport
    boundary_report: ResearchReport
    interpretation: str = "Scalar identities, obstacle bounds, and matched C1 boundary; no stochastic theorem invoked."


def _certificate(case: ResearchCase[PerpetualPut]) -> PutCertificate:
    _independent_curve(case)
    x = case.expression("moneyness", wrt="moneyness")
    boundary = case.expression("boundary")
    branches = {
        side: case.expression(branch, wrt=x).doit()
        for side, branch in (("left", "exercise"), ("right", "continuation"))
    }
    one_sided = {
        f"{label}_{side}": simplify(
            limit(diff(curve, x, order), x, 1, dir="-" if side == "left" else "+") / boundary**order
        )
        for side, curve in branches.items()
        for order, label in ((0, "value"), (1, "delta"), (2, "gamma"))
    }
    if (
        simplify(one_sided["value_left"] - one_sided["value_right"]) != 0
        or simplify(one_sided["delta_left"] - one_sided["delta_right"]) != 0
    ):
        raise ValueError("The actual one-sided value or slope fails to match")
    jump = simplify(one_sided["gamma_right"] - one_sided["gamma_left"])
    report = case.check(
        max_steps=1000,
        requirements=(
            positive(
                jump,
                name="nonzero_second_derivative_jump",
                consumer="boundary_certificate",
                reason="Actual one-sided second derivatives differ, refuting C2 at B.",
            ),
            positive(
                "initial_spot",
                name="positive_initial_spot",
                consumer="boundary_certificate",
                reason="Evaluate at S0>0.",
            ),
        ),
    )
    if any(
        report.explain(name).condition is not S.true
        for name in ("nonzero_second_derivative_jump", "positive_initial_spot")
    ):
        raise ResearchError("The boundary or initial-state certificate is unestablished", case, report)
    value = case.expression("value", wrt=x).subs(x, case.expression("initial_spot") / boundary).doit()
    return PutCertificate(
        case.expression("beta"), boundary, simplify(value), one_sided, jump, regularity_report(case), report
    )


def algebra_method(case: ResearchCase[PerpetualPut]) -> Method[PutCertificate]:
    """Use piecewise smoothness and smooth fit, without requiring global C2."""
    return Method(
        "perpetual put scalar certificate",
        _certificate,
        _requirements(case),
        ("value", "continuation_residual", "exercise_residual", "continuation_curvature"),
    ).bind(case)


def c2_consumer_method() -> Method[Expr]:
    """An overly restrictive consumer, retained to demonstrate its rejection."""
    return Method(
        "global C2 consumer",
        lambda case: case.expression("value", wrt="moneyness"),
        (
            smooth(
                "value",
                wrt="moneyness",
                domain=POSITIVE,
                name="requires_global_c2",
                consumer="global C2 consumer",
                reason="This consumer demands C2 even across the free boundary.",
            ),
        ),
        ("value",),
    )


@dataclass(frozen=True)
class StoppingVerification:
    certificate: PutCertificate
    trusted_assumptions: tuple[str, ...] = TRUSTED_STOPPING_ASSUMPTIONS
    stopping_rule: str = "tau_B=inf{t>=0:S_t<=B}; tau_B=infinity if the barrier is never reached."
    interpretation: str = (
        "Optimal among all admissible stopping times only under the explicitly trusted GBM verification rule."
    )


def verification_method(case: ResearchCase[PerpetualPut]) -> Method[StoppingVerification]:
    """Require the explicitly registered stochastic rule as well as the algebra."""
    return Method(
        "trusted perpetual put stopping verification",
        lambda current: StoppingVerification(_certificate(current)),
        (
            replace(
                _OPTIMAL,
                consumer="stochastic_verification",
                reason="This explicitly trusted conclusion is conditional on: "
                + " ".join(TRUSTED_STOPPING_ASSUMPTIONS),
                source=SOURCE,
            ),
        ),
        ("value", "continuation_residual", "exercise_residual", "continuation_curvature"),
    ).bind(case)


@dataclass(frozen=True)
class BarrierComparison:
    barrier: float
    discounted_payoff: float
    analytic_barrier: float
    barrier_error: float
    value_error: float
    evaluations: int
    kind: str = "numerical observation over constant stopping barriers"


def numerical_method(case: ResearchCase[PerpetualPut]) -> Method[BarrierComparison]:
    """Independently maximize (K-b)*(S0/b)**beta over 0<b<=min(S0,K)."""

    def execute(current: ResearchCase[PerpetualPut], *, xatol: float = 1e-12) -> BarrierComparison:
        import math

        from scipy.optimize import minimize_scalar

        _independent_curve(current)
        if not math.isfinite(xatol) or xatol <= 0:
            raise ValueError("xatol must be finite and positive")
        values = [current.expression(path) for path in ("strike", "initial_spot", "beta", "boundary")]
        if any(value.free_symbols or value.is_finite is not True or value.is_real is not True for value in values):
            raise ValueError("Numerical barrier optimization needs finite real parameter observations")
        strike, spot, beta, boundary = map(float, values)
        if spot <= 0:
            raise ValueError("initial_spot must be positive")
        upper = min(spot / strike, 1.0)

        def payoff(fraction: float) -> float:
            if fraction == 0:
                return 0.0
            return strike * (1 - fraction) * (spot / (strike * fraction)) ** beta

        result = minimize_scalar(
            lambda fraction: -payoff(fraction), bounds=(0, upper), method="bounded", options={"xatol": xatol}
        )
        if not result.success:
            raise ValueError(f"Barrier optimization failed: {result.message}")
        fraction = max((0.0, upper, float(result.x)), key=payoff)
        barrier, present_value = strike * fraction, payoff(fraction)
        analytic = min(boundary, spot)
        x = current.expression("moneyness", wrt="moneyness")
        exact_value = float(
            current.expression("value", wrt=x)
            .subs(x, current.expression("initial_spot") / current.expression("boundary"))
            .doit()
        )
        if not all(math.isfinite(value) for value in (barrier, present_value, exact_value)) or barrier <= 0:
            raise ValueError("Barrier optimization returned a nonfinite or inadmissible output")
        return BarrierComparison(
            barrier,
            present_value,
            analytic,
            abs(barrier - analytic),
            abs(present_value - exact_value),
            int(result.nfev),
        )

    return Method(
        "independent stopping-barrier maximization",
        execute,
        _requirements(case),
        ("value", "continuation_residual", "exercise_residual", "continuation_curvature"),
    ).bind(case)


def run_demo(*, numerical: bool = False) -> dict[str, Any]:
    """Return raw cases/results, blocked consumers, and trusted-rule evidence."""
    case = perpetual_put_case()
    trusted = trust_stopping_theorem(case)
    results: dict[str, Any] = {
        "case": case,
        "algebra": case.run(algebra_method),
        "trusted_case": trusted,
        "verification": trusted.run(verification_method),
    }
    for name, current, method in (
        ("global_c2", case, c2_consumer_method()),
        ("untrusted_verification", case, verification_method),
        ("zero_rate", perpetual_put_case(rate=0), algebra_method),
    ):
        try:
            current.run(method)
        except ResearchError as error:
            results[f"blocked_{name}"] = {"case": current, "report": error.report, "message": str(error)}
        else:
            raise RuntimeError(f"The restricted consumer unexpectedly accepted {name}")
    if numerical:
        results["numerical"] = case.run(numerical_method)
        exercising = perpetual_put_case(initial_spot=50)
        results["exercise_case"] = exercising
        results["numerical_exercise"] = exercising.run(numerical_method)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numerical", action="store_true", help="Optimize expected discounted stopping-barrier payoffs"
    )
    arguments = parser.parse_args()
    result = run_demo(numerical=arguments.numerical)
    certificate = result["algebra"].data
    print(f"beta={certificate.beta}; B={certificate.boundary}; P(S0)={certificate.value_at_initial_spot}")
    print("One-sided spot derivatives:", certificate.one_sided)
    print(
        "Generic global regularity:",
        {f"C{order}": certificate.regularity.explain(f"global_c{order}").verdict for order in (1, 2)},
    )
    print("Stochastic verification: explicitly trusted GBM rule, with all assumptions retained in its result.")
    for name, value in result.items():
        if name.startswith("blocked_"):
            print(f"{name}: {value['message']}")
        elif isinstance(value, ResearchResult) and name.startswith("numerical"):
            print(f"{name}: {value.data}")


if __name__ == "__main__":
    main()

"""CIR bond valuation, checked Riccati identities, and a separate Feller rule.

Run ``python -m examples.research_models.cir_bond`` for exact examples, or add
``--numerical`` for an independent Riccati ODE and quadrature comparison.
Importing this module requires only SymPy and Modello. Stochastic path and
valuation conclusions are explicitly sourced trusted rules, not CAS proofs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from math import isfinite

from sympy import Expr, Interval, Rational, S, cosh, diff, exp, log, simplify, sinh, sqrt, sympify

from modello import Coordinate, InstanceDummy, ScalarValue, component
from modello.contrib.research import (
    Claim,
    Method,
    Requirement,
    ResearchCase,
    ResearchError,
    ResearchModel,
    ResearchResult,
    Theorem,
    convex,
    decreasing,
    exact_form,
    nonnegative,
    positive,
    smooth,
)

SOURCE = "https://roycheng.cn/files/papers/interestRate/paper_Cox%26Ingersoll%26Ross_1985.pdf"
RATE_DOMAIN = Interval(0, S.Infinity)
TIME_DOMAIN = Interval(0, S.Infinity)
Get = Callable[[str], Expr]


def _h(get: Get) -> Expr:
    return sqrt(get("kappa") ** 2 + 2 * get("sigma") ** 2)


def _denominator(get: Get) -> Expr:
    h = _h(get)
    half_time = h * get("time") / 2
    return h * cosh(half_time) + get("kappa") * sinh(half_time)


def _b(get: Get) -> Expr:
    h = _h(get)
    return 2 * sinh(h * get("time") / 2) / _denominator(get)


def _log_a(get: Get) -> Expr:
    h, kappa, theta, sigma = _h(get), get("kappa"), get("theta"), get("sigma")
    return 2 * kappa * theta / sigma**2 * (log(h / _denominator(get)) + kappa * get("time") / 2)


def _at_horizon(formula: Callable[[Get], Expr], get: Get) -> Expr:
    return formula(lambda path: get("horizon") if path == "time" else get(path))


class CIRParameters(ResearchModel):
    """Positive risk-neutral parameters, without a built-in Feller condition."""

    kappa = InstanceDummy("kappa", positive=True)
    theta = InstanceDummy("theta", positive=True)
    sigma = InstanceDummy("sigma", positive=True)


class CIRBond(ResearchModel):
    """Affine valuation under dR=kappa(theta-R)dt+sigma sqrt(R)dW^Q.

    ``rate`` and remaining time ``time`` quantify curves on [0, infinity).
    ``initial_rate`` and ``horizon`` select an evaluation without specializing
    those quantified coordinates. No derivative of sqrt(rate) is consumed.
    """

    parameters = component(CIRParameters)
    kappa = parameters.kappa
    theta = parameters.theta
    sigma = parameters.sigma
    rate = Coordinate("r", nonnegative=True)
    time = Coordinate("tau", nonnegative=True)
    initial_rate = InstanceDummy("r0", nonnegative=True)
    horizon = InstanceDummy("T_minus_t", nonnegative=True)
    drift = kappa * (theta - rate)
    diffusion = sigma * sqrt(rate)
    variance_rate = sigma**2 * rate
    feller_margin = 2 * kappa * theta - sigma**2
    h = sqrt(kappa**2 + 2 * sigma**2)
    denominator = h * cosh(h * time / 2) + kappa * sinh(h * time / 2)
    b = 2 * sinh(h * time / 2) / denominator
    log_a = 2 * kappa * theta / sigma**2 * (log(h / denominator) + kappa * time / 2)
    log_price = log_a - b * rate
    price = exp(log_price)
    denominator_at_horizon = h * cosh(h * horizon / 2) + kappa * sinh(h * horizon / 2)
    b_at_horizon = 2 * sinh(h * horizon / 2) / denominator_at_horizon
    log_a_at_horizon = 2 * kappa * theta / sigma**2 * (log(h / denominator_at_horizon) + kappa * horizon / 2)
    log_curve = log_a_at_horizon - b_at_horizon * rate
    curve = exp(log_curve)
    log_value = log_a_at_horizon - b_at_horizon * initial_rate
    value = exp(log_value)


def _require(claim: Claim, consumer: str, reason: str, *, page: int = 8) -> Requirement:
    return Requirement(claim, consumer, reason, SOURCE + f"#page={page}")


def _parameter_requirements(*, pricing: bool) -> tuple[Requirement, ...]:
    requirements = [
        _require(positive(name, name=f"positive_{name}"), "CIR_parameters", f"The CIR rule requires {name}>0.")
        for name in ("kappa", "theta", "sigma")
    ]
    requirements.append(
        _require(
            nonnegative("initial_rate", name="nonnegative_initial_rate"),
            "CIR_parameters",
            "The nonnegative CIR solution starts at a nonnegative rate, including zero.",
        )
    )
    if pricing:
        requirements.append(
            _require(
                nonnegative("horizon", name="nonnegative_horizon"),
                "bond_pricing",
                "Remaining time is nonnegative; zero is the terminal payment.",
                page=10,
            )
        )
    return tuple(requirements)


def _dynamics_requirements() -> tuple[Requirement, ...]:
    return (
        _require(
            exact_form(
                "drift",
                lambda get: get("kappa") * (get("theta") - get("rate")),
                wrt="rate",
                domain=RATE_DOMAIN,
                name="cir_drift_form",
            ),
            "CIR_dynamics",
            "The trusted path rule applies to the stated risk-neutral CIR drift.",
        ),
        _require(
            exact_form(
                "diffusion",
                lambda get: get("sigma") * sqrt(get("rate")),
                wrt="rate",
                domain=RATE_DOMAIN,
                name="cir_diffusion_form",
            ),
            "CIR_dynamics",
            "The Brownian diffusion is sigma sqrt(r), defined at zero; C2 regularity is not required.",
        ),
        _require(
            exact_form(
                "variance_rate",
                lambda get: get("sigma") ** 2 * get("rate"),
                wrt="rate",
                domain=RATE_DOMAIN,
                name="cir_variance_form",
            ),
            "CIR_dynamics",
            "The pricing generator consumes sigma squared times r.",
        ),
    )


def _coefficient_requirements() -> tuple[Requirement, ...]:
    requirements = [
        _require(
            exact_form(path, formula, wrt="time", domain=TIME_DOMAIN, name=f"{path}_form"),
            "affine_coefficients",
            "The affine shortcut consumes the actual CIR coefficient formula.",
            page=10,
        )
        for path, formula in (("denominator", _denominator), ("b", _b), ("log_a", _log_a))
    ]
    requirements.extend(
        (
            _require(
                exact_form(
                    "b_at_horizon",
                    lambda get: _at_horizon(_b, get),
                    name="horizon_b_form",
                ),
                "affine_coefficients",
                "The reported B is the declared time curve at the selected horizon.",
                page=10,
            ),
            _require(
                exact_form(
                    "log_a_at_horizon",
                    lambda get: _at_horizon(_log_a, get),
                    name="horizon_log_a_form",
                ),
                "affine_coefficients",
                "The reported log A is the declared time curve at the selected horizon.",
                page=10,
            ),
            _require(
                exact_form(
                    "log_price",
                    lambda get: get("log_a") - get("b") * get("rate"),
                    wrt="time",
                    domain=TIME_DOMAIN,
                    name="affine_log_price_form",
                ),
                "affine_coefficients",
                "The generator calculation uses the declared log price log A - B r.",
                page=10,
            ),
            _require(
                exact_form(
                    "price",
                    lambda get: exp(get("log_price")),
                    wrt="time",
                    domain=TIME_DOMAIN,
                    name="affine_price_form",
                ),
                "affine_coefficients",
                "The source formula is exp(log A - B r), rather than an arbitrary function sharing a residual.",
                page=10,
            ),
            _require(
                exact_form(
                    "log_curve",
                    lambda get: get("log_a_at_horizon") - get("b_at_horizon") * get("rate"),
                    wrt="rate",
                    domain=RATE_DOMAIN,
                    name="horizon_log_curve_form",
                ),
                "affine_coefficients",
                "The evaluated curve retains the same coefficient formula at the selected horizon.",
                page=10,
            ),
            _require(
                exact_form(
                    "curve",
                    lambda get: exp(get("log_curve")),
                    wrt="rate",
                    domain=RATE_DOMAIN,
                    name="horizon_curve_form",
                ),
                "affine_coefficients",
                "The selected-horizon rate curve is exp(log A - B r).",
                page=10,
            ),
            _require(
                exact_form(
                    "log_value",
                    lambda get: get("log_a_at_horizon") - get("b_at_horizon") * get("initial_rate"),
                    name="initial_log_value_form",
                ),
                "affine_coefficients",
                "The scalar value evaluates that curve at the initial rate.",
                page=10,
            ),
            _require(
                exact_form("value", lambda get: exp(get("log_value")), name="initial_value_form"),
                "affine_coefficients",
                "The scalar value is the exponential of its initial log price.",
                page=10,
            ),
        )
    )
    return tuple(requirements)


def _path_claim(*, strict: bool) -> Claim:
    return Claim("unattainable_zero" if strict else "nonnegative_paths", "diffusion", "theorem")


def symbolic_case(model: type[CIRBond] = CIRBond) -> ResearchCase[CIRBond]:
    """Register precise source rules without assuming their scalar premises.

    The probabilistic context of each trusted rule is a standard Brownian CIR
    SDE under the risk-neutral measure. Scalar CAS checks establish only the
    listed parameter and formula premises, not stochastic boundary theory.
    """
    case = ResearchCase.for_model(model, "cir_bond")
    path_premises = (*_parameter_requirements(pricing=False), *_dynamics_requirements())
    case = case.with_theorem(
        Theorem("CIR_nonnegative_solution", _path_claim(strict=False), path_premises, SOURCE + "#page=8", "1")
    )
    feller_form = _require(
        exact_form(
            "feller_margin",
            lambda get: 2 * get("kappa") * get("theta") - get("sigma") ** 2,
            name="feller_margin_form",
        ),
        "boundary_classification",
        "The condition uses the same risk-neutral CIR parameters as the dynamics.",
    )
    case = case.with_theorem(
        Theorem(
            "CIR_unattainable_zero_boundary",
            _path_claim(strict=True),
            (
                *path_premises,
                feller_form,
                _require(
                    positive("initial_rate", name="strictly_positive_initial_rate"),
                    "boundary_classification",
                    "The unattainable-boundary statement starts strictly above zero.",
                ),
                _require(
                    nonnegative("feller_margin", name="feller_condition"),
                    "boundary_classification",
                    "2 kappa theta >= sigma squared is sufficient for zero to be unattainable from r0>0.",
                ),
            ),
            SOURCE + "#page=8",
            "1",
        )
    )
    return case.with_theorem(
        Theorem(
            "CIR_affine_discounted_expectation",
            Claim("affine_valuation", "value", "theorem"),
            (
                *_parameter_requirements(pricing=True),
                *_dynamics_requirements(),
                *_coefficient_requirements(),
                _require(
                    _path_claim(strict=False),
                    "bond_pricing",
                    "The standard nonnegative CIR solution is the underlying rate process.",
                ),
            ),
            SOURCE + "#page=10",
            "1",
        )
    )


def _exact(value: ScalarValue) -> Expr:
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("CIR observations must be finite real scalar values")
    exact = Rational(str(value)) if isinstance(value, float) else sympify(value)
    if exact.free_symbols or exact.is_finite is not True or exact.is_real is not True:
        raise ValueError("CIR observations must be finite real scalar values")
    return exact


def cir_case(
    *,
    kappa: ScalarValue = Rational(3, 5),
    theta: ScalarValue | None = Rational(1, 20),
    sigma: ScalarValue = Rational(1, 5),
    initial_rate: ScalarValue = Rational(1, 25),
    horizon: ScalarValue = 5,
) -> ResearchCase[CIRBond]:
    """Observe exact inputs; ``theta=None`` retains the symbolic level.

    Python floats are interpreted through their decimal spelling. The factory
    rejects nonfinite or nonreal scalar observations. Intrinsic positive
    parameter domains can reject inputs during component construction; the
    consuming methods also check their parameter and state-domain premises.
    Zero volatility remains outside this model's domain.
    """
    parameters = {"kappa": _exact(kappa), "sigma": _exact(sigma)}
    if theta is not None:
        parameters["theta"] = _exact(theta)
    return symbolic_case().observe(
        parameters=CIRParameters("risk_neutral", **parameters),
        initial_rate=_exact(initial_rate),
        horizon=_exact(horizon),
    )


def with_feller(case: ResearchCase[CIRBond]) -> ResearchCase[CIRBond]:
    """Add the sufficient boundary condition as an explicitly removable assumption."""
    return case.assume(case.expression("feller_margin") >= 0)


@dataclass(frozen=True)
class BondPrice:
    initial_rate: Expr
    horizon: Expr
    b: Expr
    log_a: Expr
    price: Expr


@dataclass(frozen=True)
class BondVerification:
    """Exact scalar identities, independent of the trusted stochastic rules."""

    riccati_b: Expr
    riccati_log_a: Expr
    generator: Expr
    terminal_price: Expr


@dataclass(frozen=True)
class BoundaryConclusion:
    statement: str
    source: str = SOURCE + "#page=8"
    kind: str = "trusted stochastic boundary rule"


@dataclass(frozen=True)
class NumericalBond:
    price: float
    b: float
    log_a: float
    diagnostics: Mapping[str, float | int | str]


def _verification(case: ResearchCase[CIRBond]) -> BondVerification:
    time = case.expression("time", wrt="time")
    rate = case.expression("rate", wrt="rate")
    kappa, theta, sigma = (case.expression(path) for path in ("kappa", "theta", "sigma"))
    b, log_a = (case.expression(path, wrt="time") for path in ("b", "log_a"))
    log_price = case.expression("log_price", wrt="time")
    drift, variance = (case.expression(path, wrt="rate") for path in ("drift", "variance_rate"))
    first = diff(log_price, rate)
    generator = diff(log_price, time) - drift * first - variance * (diff(log_price, rate, 2) + first**2) / 2 + rate
    return BondVerification(
        simplify(diff(b, time) - (1 - kappa * b - sigma**2 * b**2 / 2)),
        simplify(diff(log_a, time) + kappa * theta * b),
        simplify(generator),
        simplify(case.expression("price", wrt="time").subs(time, 0)),
    )


def _verification_requirements(case: ResearchCase[CIRBond]) -> tuple[Requirement, ...]:
    checked = _verification(case)
    requirements = [
        _require(
            positive("denominator", wrt="time", domain=TIME_DOMAIN, name="coefficient_denominator"),
            "scalar_verification",
            "The original coefficient denominator must be positive before residual cancellation.",
            page=10,
        ),
    ]
    requirements.extend(
        _require(
            smooth(path, wrt="time", domain=TIME_DOMAIN, name=f"{path}_time_c2"),
            "scalar_verification",
            "The actual coefficient is regular on nonnegative remaining time.",
            page=10,
        )
        for path in ("b", "log_a")
    )
    requirements.extend(
        _require(
            exact_form(expression, expected, name=name),
            "scalar_verification",
            reason,
            page=10,
        )
        for name, expression, expected, reason in (
            ("riccati_b", checked.riccati_b, S.Zero, "The actual B satisfies B'=1-kappa B-sigma squared B squared/2."),
            ("riccati_log_a", checked.riccati_log_a, S.Zero, "The actual log A satisfies (log A)'=-kappa theta B."),
            (
                "pricing_generator",
                checked.generator,
                S.Zero,
                "The normalized pricing generator residual is zero after checking the exponential price form.",
            ),
            ("terminal_price", checked.terminal_price, S.One, "The declared bond pays one at remaining time zero."),
        )
    )
    return tuple(requirements)


def verification_method(case: ResearchCase[CIRBond]) -> Method[BondVerification]:
    """Check actual coefficients and identities; no stochastic theorem is inferred."""
    return Method(
        "exact CIR scalar verification",
        _verification,
        (
            *_parameter_requirements(pricing=True),
            *_dynamics_requirements(),
            *_coefficient_requirements(),
            *_verification_requirements(case),
        ),
        uses=("b", "log_a", "log_price", "price", "drift", "variance_rate"),
    ).bind(case)


def _price(case: ResearchCase[CIRBond]) -> BondPrice:
    return BondPrice(
        *(case.expression(path) for path in ("initial_rate", "horizon", "b_at_horizon", "log_a_at_horizon", "value"))
    )


def pricing_method(case: ResearchCase[CIRBond]) -> Method[BondPrice]:
    """Value the source formula with live scalar checks and a sourced valuation rule."""
    requirements = [
        *_parameter_requirements(pricing=True),
        *_dynamics_requirements(),
        *_coefficient_requirements(),
        *_verification_requirements(case),
        _require(
            Claim("affine_valuation", "value", "theorem"),
            "bond_pricing",
            "CIR's source formula identifies the discounted expectation under the stated risk-neutral dynamics.",
            page=10,
        ),
        _require(
            smooth("curve", wrt="rate", domain=RATE_DOMAIN, name="rate_c2"),
            "bond_pricing",
            "The affine bond price, including r=0, is twice differentiable in the current rate.",
            page=10,
        ),
        _require(
            convex("curve", wrt="rate", domain=RATE_DOMAIN, name="rate_convex"),
            "bond_pricing",
            "The bond is convex in the current rate.",
            page=10,
        ),
        _require(
            positive("curve", wrt="rate", domain=RATE_DOMAIN, name="positive_price"),
            "bond_pricing",
            "The affine price is strictly positive at finite current rates.",
            page=10,
        ),
    ]
    if case.expression("horizon").is_positive is True:
        requirements.append(
            _require(
                decreasing("curve", wrt="rate", domain=RATE_DOMAIN, name="rate_decreasing"),
                "bond_pricing",
                "At positive remaining time the bond is strictly decreasing in current rate.",
                page=10,
            )
        )
    else:
        rate = case.expression("rate", wrt="rate")
        slope = -diff(case.expression("curve", wrt="rate"), rate)
        requirements.append(
            _require(
                nonnegative(slope, wrt=rate, domain=RATE_DOMAIN, name="rate_nonincreasing"),
                "bond_pricing",
                "The terminal bond is constant; nonincrease does not assert a strictly negative slope.",
                page=10,
            )
        )
    return Method("CIR affine bond price", _price, tuple(requirements), uses=("value", "curve", "price")).bind(case)


def price(case: ResearchCase[CIRBond]) -> ResearchResult[BondPrice]:
    return case.run(pricing_method)


def verify(case: ResearchCase[CIRBond]) -> ResearchResult[BondVerification]:
    return case.run(verification_method)


def strict_positive_boundary_method(case: ResearchCase[CIRBond]) -> Method[BoundaryConclusion]:
    """Require the sourced unattainable-zero rule, separately from bond valuation."""
    return Method(
        "CIR strictly positive boundary conclusion",
        lambda current: BoundaryConclusion("From r0>0, the CIR solution does not hit zero at any finite time."),
        (
            _require(
                _path_claim(strict=True),
                "boundary_classification",
                "This conclusion needs a positive initial rate and the Feller sufficient condition.",
            ),
        ),
        uses=("drift", "diffusion", "variance_rate", "initial_rate", "feller_margin"),
    ).bind(case)


def _numerical(case: ResearchCase[CIRBond], *, rtol: float = 1e-10, atol: float = 1e-12) -> NumericalBond:
    import scipy
    from scipy.integrate import quad, solve_ivp

    if not isfinite(rtol) or not isfinite(atol) or min(rtol, atol) <= 0:
        raise ValueError("Numerical tolerances must be finite and strictly positive")
    expressions = tuple(case.expression(path) for path in ("kappa", "theta", "sigma", "initial_rate", "horizon"))
    if any(expression.free_symbols for expression in expressions):
        raise ValueError("Numerical CIR verification requires observed parameters")
    kappa, theta, sigma, rate, horizon = map(float, expressions)
    if not all(isfinite(item) for item in (kappa, theta, sigma, rate, horizon)):
        raise ValueError("Numerical CIR verification needs finite observed parameters")
    exact = _price(case)
    if horizon == 0:
        return NumericalBond(
            1.0,
            0.0,
            0.0,
            {
                "kind": "numerical observation",
                "horizon": horizon,
                "ode_nfev": 0,
                "price_difference": abs(float(exact.price) - 1),
                "scipy_version": scipy.__version__,
                "rtol": rtol,
                "atol": atol,
            },
        )

    def rhs(time, state):
        b = state[0]
        return (1 - kappa * b - sigma**2 * b**2 / 2, -kappa * theta * b)

    solution = solve_ivp(rhs, (0, horizon), (0.0, 0.0), method="DOP853", dense_output=True, rtol=rtol, atol=atol)
    if not solution.success or solution.sol is None:
        raise ValueError(f"CIR Riccati integration failed: {solution.message}")
    b, log_a = map(float, solution.y[:, -1])
    integral, quadrature_error = quad(lambda time: float(solution.sol(time)[0]), 0, horizon, epsabs=atol, epsrel=rtol)
    from math import exp as numeric_exp

    value = numeric_exp(log_a - b * rate)
    if not all(isfinite(item) for item in (b, log_a, integral, value, quadrature_error)):
        raise ValueError("CIR Riccati integration produced nonfinite results")
    diagnostics: dict[str, float | int | str] = {
        "kind": "numerical observation",
        "ode_nfev": solution.nfev,
        "b_difference": abs(b - float(exact.b)),
        "log_a_difference": abs(log_a - float(exact.log_a)),
        "price_difference": abs(value - float(exact.price)),
        "quadrature_log_a_difference": abs(-kappa * theta * integral - log_a),
        "quadrature_reported_error": quadrature_error,
        "scipy_version": scipy.__version__,
        "rtol": rtol,
        "atol": atol,
    }
    return NumericalBond(value, b, log_a, diagnostics)


def numerical_method(case: ResearchCase[CIRBond]) -> Method[NumericalBond]:
    """Independently integrate the Riccati ODE; diagnostics never prove a theorem."""
    return Method(
        "independent CIR Riccati integration",
        _numerical,
        (
            *_parameter_requirements(pricing=True),
            *_dynamics_requirements(),
            *_coefficient_requirements(),
            *_verification_requirements(case),
        ),
        uses=("kappa", "theta", "sigma", "initial_rate", "horizon", "value"),
    ).bind(case)


def run_demo(*, numerical: bool = False) -> dict[str, dict[str, object]]:
    """Return raw cases/results, including a blocked boundary conclusion.

    The symbolic pair adds then removes only the Feller assumption. Its
    positive theta premise is retained in both branches. A failed sufficient
    theorem is unresolved, and never means that negative rates occur.
    """
    cases = {
        "feller_satisfied": cir_case(),
        "feller_failed": cir_case(sigma=Rational(2, 5)),
        "zero_initial_rate": cir_case(initial_rate=0),
        "terminal": cir_case(horizon=0),
    }
    rows: dict[str, dict[str, object]] = {}
    for name, case in cases.items():
        row: dict[str, object] = {"case": case, "price": price(case), "verification": verify(case)}
        try:
            row["boundary"] = case.run(strict_positive_boundary_method)
        except ResearchError as error:
            row["boundary_blocked"] = str(error)
            row["boundary_report"] = error.report
        if numerical:
            row["numerical"] = case.run(numerical_method)
        rows[name] = row
    base = cir_case(theta=None)
    bounded = with_feller(base)
    feller_condition = bounded.assumptions[-1]
    relaxed = bounded.without_assumptions(
        feller_condition, reason="Compare valuation without the optional Feller boundary condition."
    )
    for name, case in (("feller_assumption_added", bounded), ("feller_assumption_removed", relaxed)):
        row = {"case": case}
        try:
            row["boundary"] = case.run(strict_positive_boundary_method)
        except ResearchError as error:
            row["boundary_blocked"] = str(error)
            row["boundary_report"] = error.report
        rows[name] = row
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical", action="store_true", help="Compare with Riccati integration and quadrature")
    arguments = parser.parse_args()
    for name, row in run_demo(numerical=arguments.numerical).items():
        print(name)
        result = row.get("price")
        if isinstance(result, ResearchResult):
            print(f"  price: {result.data.price.evalf(12)}")
        boundary = row.get("boundary")
        if isinstance(boundary, ResearchResult):
            print(f"  boundary: {boundary.data.statement} [trusted source rule]")
        elif "boundary_blocked" in row:
            print("  boundary: sufficient premises unestablished; see the retained report")
        reference = row.get("numerical")
        if isinstance(reference, ResearchResult):
            print(f"  numerical price difference: {reference.data.diagnostics['price_difference']:.3g}")


if __name__ == "__main__":
    main()

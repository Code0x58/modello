"""Exact two-asset allocation, changing restrictions, and checked KKT evidence.

Run ``python -m examples.research_models.markowitz``. Inputs are illustrative
moments, not estimates or investment recommendations. Budget is eliminated
exactly with weights ``w`` and ``1-w``; the remaining optimization is scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sympy import Derivative, Eq, Expr, FiniteSet, Float, Ge, Interval, Rational, S, Subs, simplify, sympify

from examples._research_checks import require_parameter_assumptions
from modello import ConstraintKind, Coordinate, InstanceDummy, component
from modello.contrib.research import (
    Method,
    Requirement,
    ResearchCase,
    ResearchModel,
    ResearchResult,
    Theorem,
    convex,
    exact_form,
    nonnegative,
    positive,
    predicate,
    requires,
    smooth,
)

SOURCE = "https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf"


class Covariance(ResearchModel):
    """A symmetric 2x2 covariance specification, including singular cases."""

    first = requires(
        InstanceDummy("v1", real=True),
        nonnegative("first", name="first_variance", consumer="covariance", reason="Variance is nonnegative."),
    )
    second = requires(
        InstanceDummy("v2", real=True),
        nonnegative("second", name="second_variance", consumer="covariance", reason="Variance is nonnegative."),
    )
    cross = InstanceDummy("c", real=True)
    determinant = requires(
        first * second - cross**2,
        nonnegative(
            "determinant",
            name="principal_minor",
            consumer="covariance",
            reason="Together with nonnegative diagonal entries, this certifies 2x2 positive semidefiniteness.",
            source=SOURCE + "#page=656",
        ),
    )
    contrast = first + second - 2 * cross


class ExpectedReturns(ResearchModel):
    first = InstanceDummy("mu1", real=True)
    second = InstanceDummy("mu2", real=True)


class TwoAssetPortfolio(ResearchModel):
    """Variance along the fully invested budget line, with a separate candidate."""

    moments = component(Covariance)
    means = component(ExpectedReturns)
    weight = Coordinate("w", real=True)
    candidate = InstanceDummy("w_star", real=True)
    other_weight = 1 - weight
    expected_return = weight * means.first + other_weight * means.second
    variance = requires(
        moments.first * weight**2 + moments.second * other_weight**2 + 2 * moments.cross * weight * other_weight,
        convex(
            "variance", wrt="weight", consumer="allocation", reason="Variance must be convex along the budget line."
        ),
        smooth(
            "variance", wrt="weight", consumer="allocation", reason="Classical scalar KKT uses this smooth objective."
        ),
    )
    slope = Derivative(variance, weight, evaluate=False)
    candidate_variance = Subs(variance, weight, candidate)
    candidate_return = Subs(expected_return, weight, candidate)
    candidate_slope = Subs(slope, weight, candidate)
    completion_residual = (
        variance
        - candidate_variance
        - candidate_slope * (weight - candidate)
        - moments.contrast * (weight - candidate) ** 2
    )


def _exact(value) -> Expr:
    return Rational(str(value)) if isinstance(value, float) else sympify(value)


@dataclass(frozen=True)
class PortfolioRestrictions:
    """Optional economic restrictions, separate from covariance validity.

    ``without`` branches the restriction set; it does not mutate an old case
    or discard a solver's mathematical premises.
    """

    long_only: bool = True
    target: Expr | None = None
    cap: Expr | None = None
    changes: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("target", "cap"):
            value = getattr(self, name)
            if value is not None:
                value = _exact(value)
                if value.free_symbols or value.is_finite is not True or value.is_real is not True:
                    raise ValueError(f"{name} must be an exact finite real value")
                object.__setattr__(self, name, value)

    def without(self, *names: str) -> PortfolioRestrictions:
        for name in names:
            if name not in {"long_only", "target", "cap"}:
                raise ValueError(f"Unknown restriction: {name}")
        return replace(
            self,
            long_only=False if "long_only" in names else self.long_only,
            target=None if "target" in names else self.target,
            cap=None if "cap" in names else self.cap,
            changes=(*self.changes, *(f"Removed {name}." for name in names)),
        )


@dataclass(frozen=True)
class LinearRestriction:
    name: str
    coefficient: Expr
    bound: Expr


@dataclass(frozen=True)
class Allocation:
    """An exact feasible minimizer and its scalar optimality certificate."""

    weight: Expr
    other_weight: Expr
    variance: Expr
    expected_return: Expr
    feasible: FiniteSet | Interval
    minimizers: FiniteSet | Interval
    active: tuple[str, ...]
    multipliers: tuple[tuple[str, Expr], ...]
    stationarity_residual: Expr
    completion_residual: Expr
    restrictions: PortfolioRestrictions


def portfolio_case(
    *, first_variance=4, second_variance=1, covariance=0, first_return=Rational(8, 100), second_return=Rational(4, 100)
):
    """Use exact illustrative moments; preserve an unbound allocation coordinate."""
    return ResearchCase.for_model(TwoAssetPortfolio, "portfolio").observe(
        moments=Covariance(
            "moments", first=_exact(first_variance), second=_exact(second_variance), cross=_exact(covariance)
        ),
        means=ExpectedReturns("means", first=_exact(first_return), second=_exact(second_return)),
    )


def _contracts(*, unique: bool):
    requirements = [
        nonnegative(path, name=name, consumer="allocation", reason=reason, source=SOURCE)
        for path, name, reason in (
            ("moments.first", "valid_first_variance", "The first covariance diagonal is nonnegative."),
            ("moments.second", "valid_second_variance", "The second covariance diagonal is nonnegative."),
            ("moments.determinant", "valid_covariance", "The covariance matrix is positive semidefinite."),
            ("moments.contrast", "convex_budget_line", "Reduced quadratic curvature is nonnegative."),
        )
    ]
    if unique:
        requirements.append(
            positive(
                "moments.contrast",
                name="unique_curvature",
                consumer="allocation",
                reason="This shortcut requires strict reduced curvature.",
                source=SOURCE,
            )
        )
    requirements.extend(
        (
            exact_form(
                "variance",
                lambda get: (
                    get("moments.first") * get("weight") ** 2
                    + get("moments.second") * (1 - get("weight")) ** 2
                    + 2 * get("moments.cross") * get("weight") * (1 - get("weight"))
                ),
                wrt="weight",
                name="quadratic_variance_form",
                consumer="allocation",
                reason="The analytic candidate uses this declared quadratic law.",
                source=SOURCE,
            ),
            exact_form(
                "expected_return",
                lambda get: get("weight") * get("means.first") + (1 - get("weight")) * get("means.second"),
                wrt="weight",
                name="linear_return_form",
                consumer="allocation",
                reason="A target return must use this same declared mean law.",
                source=SOURCE,
            ),
            exact_form(
                "completion_residual",
                S.Zero,
                wrt="weight",
                name="quadratic_completion",
                consumer="allocation",
                reason="The global objective difference has the checked quadratic decomposition.",
                source=SOURCE,
            ),
        )
    )
    forms = (
        (
            "moments.determinant",
            lambda get: get("moments.first") * get("moments.second") - get("moments.cross") ** 2,
            None,
            "determinant_form",
        ),
        (
            "moments.contrast",
            lambda get: get("moments.first") + get("moments.second") - 2 * get("moments.cross"),
            None,
            "contrast_form",
        ),
        ("slope", lambda get: Derivative(get("variance"), get("weight"), evaluate=False), "weight", "slope_form"),
        (
            "candidate_variance",
            lambda get: Subs(get("variance"), get("weight"), get("candidate")),
            None,
            "candidate_variance_form",
        ),
        (
            "candidate_return",
            lambda get: Subs(get("expected_return"), get("weight"), get("candidate")),
            None,
            "candidate_return_form",
        ),
        (
            "candidate_slope",
            lambda get: Subs(get("slope"), get("weight"), get("candidate")),
            None,
            "candidate_slope_form",
        ),
        (
            "completion_residual",
            lambda get: (
                get("variance")
                - get("candidate_variance")
                - get("candidate_slope") * (get("weight") - get("candidate"))
                - get("moments.contrast") * (get("weight") - get("candidate")) ** 2
            ),
            "weight",
            "completion_expression_form",
        ),
    )
    for path, formula, coordinate, name in forms:
        requirements.append(
            exact_form(
                path,
                formula,
                wrt=coordinate,
                name=name,
                consumer="allocation",
                reason="The certificate consumes this live declared identity.",
                source=SOURCE,
            )
        )
    return tuple(requirements)


def _linear_restrictions(case, restrictions):
    result = []
    if restrictions.long_only:
        result.extend(
            (
                LinearRestriction("first_nonnegative", S.One, S.Zero),
                LinearRestriction("second_nonnegative", -S.One, -S.One),
            )
        )
    if restrictions.cap is not None:
        result.append(LinearRestriction("concentration_cap", -S.One, -restrictions.cap))
    if restrictions.target is not None:
        result.append(
            LinearRestriction(
                "minimum_return",
                case.symbols.means.first - case.symbols.means.second,
                restrictions.target - case.symbols.means.second,
            )
        )
    return tuple(result)


def _closed(case, value):
    value = simplify(case.expression(value).doit())
    if value.free_symbols or value.has(Float) or value.is_real is not True or value.is_finite is not True:
        raise ValueError("Bind exact finite real moments before deriving an allocation")
    return value


def _ordered_maximum(left, right):
    order = left >= right
    if order is S.true:
        return left
    if order is S.false:
        return right
    raise ValueError("The ordering of a restriction bound is unresolved")


def _ordered_minimum(left, right):
    return -_ordered_maximum(-left, -right)


def _propose(case, restrictions):
    weight_coordinate = case._reference("weight")
    require_parameter_assumptions(
        case,
        weight_coordinate,
        message="Keep weight assumptions in PortfolioRestrictions so the admissible interval is explicit",
    )
    for constraint in case.problem.constraints:
        if constraint.kind is ConstraintKind.INPUT and weight_coordinate in (
            case._law(constraint.left).free_symbols | case._law(constraint.right).free_symbols
        ):
            raise ValueError(
                "Observe portfolio parameters rather than weight-dependent outputs; use PortfolioRestrictions"
            )
    specs = _linear_restrictions(case, restrictions)
    evaluated = tuple((item, _closed(case, item.coefficient), _closed(case, item.bound)) for item in specs)
    lower, upper = S.NegativeInfinity, S.Infinity
    for item, coefficient, bound in evaluated:
        if coefficient.is_positive is True:
            lower = _ordered_maximum(lower, bound / coefficient)
        elif coefficient.is_negative is True:
            upper = _ordered_minimum(upper, bound / coefficient)
        elif coefficient.is_zero is True:
            if bound.is_positive is True:
                raise ValueError(f"Infeasible restriction {item.name}: both assets have the same expected return")
            if bound.is_nonpositive is not True:
                raise ValueError(f"The sign of restriction {item.name}'s constant bound is unresolved")
        else:
            raise ValueError(f"The sign of restriction {item.name}'s coefficient is unresolved")
    feasible_order = lower <= upper
    if feasible_order is S.false:
        raise ValueError(f"The restrictions are infeasible: lower weight {lower} exceeds upper weight {upper}")
    if feasible_order is not S.true:
        raise ValueError("The ordering of the restriction bounds is unresolved")
    a = _closed(case, "moments.contrast")
    if a == 0:
        if _closed(case, "moments.second") != _closed(case, "moments.cross"):
            raise ValueError("A zero-curvature covariance must have a constant budget-line variance")
        weight = _ordered_maximum(lower, _ordered_minimum(upper, S.Zero))
    else:
        weight = _ordered_maximum(
            lower, _ordered_minimum(upper, (_closed(case, "moments.second") - _closed(case, "moments.cross")) / a)
        )
    candidate_case = case.observe(candidate=weight)
    gradient = _closed(candidate_case, "candidate_slope")
    active = tuple(item.name for item, coefficient, bound in evaluated if simplify(coefficient * weight - bound) == 0)
    multipliers = dict.fromkeys((item.name for item in specs), S.Zero)
    if gradient != 0:
        matching = next(
            (
                item
                for item, coefficient, _ in evaluated
                if item.name in active and (gradient / coefficient).is_nonnegative is True
            ),
            None,
        )
        if matching is None:
            raise ValueError("No active constraint supports the candidate's gradient")
        multipliers[matching.name] = simplify(gradient / _closed(case, matching.coefficient))
    stationarity = simplify(
        gradient - sum((multipliers[item.name] * coefficient for item, coefficient, _ in evaluated), S.Zero)
    )
    completion = simplify(candidate_case.expression("completion_residual", wrt="weight"))
    feasible = Interval(lower, upper)
    allocation = Allocation(
        weight,
        1 - weight,
        _closed(candidate_case, "candidate_variance"),
        _closed(candidate_case, "candidate_return"),
        feasible,
        feasible if a == 0 else FiniteSet(weight),
        active,
        tuple(multipliers.items()),
        stationarity,
        completion,
        restrictions,
    )
    return candidate_case, allocation, specs


def allocate(
    case: ResearchCase, restrictions: PortfolioRestrictions | None = None, *, unique=True
) -> ResearchResult[Allocation]:
    """Derive a candidate, then independently check its exact scalar KKT proof.

    Convex KKT sufficiency is the cited trusted rule. Its scalar polynomial,
    primal/dual, stationarity, and complementarity premises are checked here.
    """
    restrictions = restrictions or PortfolioRestrictions()
    proposed = case.run(
        Method("derive allocation", _propose, _contracts(unique=unique), ("variance", "expected_return")),
        restrictions=restrictions,
    )
    candidate_case, allocation, specs = proposed.data
    multipliers = dict(allocation.multipliers)
    premises = list(_contracts(unique=unique))
    candidate = candidate_case.symbols.candidate
    gradient = candidate_case.symbols.candidate_slope
    dual_sum = sum((multipliers[item.name] * item.coefficient for item in specs), S.Zero)
    premises.append(
        predicate(
            Eq(gradient, dual_sum, evaluate=False),
            name="kkt_stationarity",
            consumer="convex_KKT",
            reason="The Lagrangian derivative vanishes.",
            source=SOURCE + "#page=263",
        )
    )
    for item in specs:
        gap = item.coefficient * candidate - item.bound
        multiplier = multipliers[item.name]
        premises.extend(
            (
                predicate(
                    Ge(gap, 0, evaluate=False),
                    name=f"primal_{item.name}",
                    consumer="convex_KKT",
                    reason="The allocation satisfies this restriction.",
                    source=SOURCE,
                ),
                predicate(
                    Ge(multiplier, 0, evaluate=False),
                    name=f"dual_{item.name}",
                    consumer="convex_KKT",
                    reason="The inequality multiplier is nonnegative.",
                    source=SOURCE,
                ),
                predicate(
                    Eq(multiplier * gap, 0, evaluate=False),
                    name=f"complementary_{item.name}",
                    consumer="convex_KKT",
                    reason="The multiplier and constraint slack are complementary.",
                    source=SOURCE,
                ),
            )
        )
    conclusion = Requirement("global_minimum", "candidate_variance", "theorem")
    certified = candidate_case.with_theorem(
        Theorem("scalar_convex_KKT", conclusion, tuple(premises), SOURCE + "#page=263", "1")
    )
    required = replace(
        conclusion,
        consumer="certified allocation",
        reason="Convex KKT certifies a global minimum on the chosen interval.",
        source=SOURCE,
    )
    return certified.run(
        Method("certified allocation", lambda current: allocation, (required,), ("variance", "expected_return"))
    )


def numerical_reference(case: ResearchCase, restrictions: PortfolioRestrictions | None = None):
    """Independent bounded scalar minimization; numerical agreement is not proof."""
    import numpy as np
    from scipy.optimize import minimize_scalar

    restrictions = restrictions or PortfolioRestrictions()
    result = allocate(case, restrictions)
    interval = result.data.feasible
    if isinstance(interval, FiniteSet):
        lower = upper = next(iter(interval))
    else:
        lower, upper = interval.start, interval.end
    if not all(endpoint.is_finite is True for endpoint in (lower, upper)):
        raise ValueError("The independent numerical reference requires a bounded feasible interval")
    w = case._reference("weight")
    expression = case.expression("variance", wrt=w)
    from sympy import lambdify

    objective = lambdify(w, expression, modules="numpy")
    if lower == upper:
        candidate = float(lower)
    else:
        answer = minimize_scalar(
            objective, bounds=(float(lower), float(upper)), method="bounded", options={"xatol": 1e-12}
        )
        if not answer.success:
            raise ValueError(answer.message)
        candidate = min((float(lower), float(upper), float(answer.x)), key=objective)
    if not np.isfinite(candidate):
        raise ValueError("The numerical reference produced a nonfinite weight")
    return {"weight": candidate, "variance": float(objective(candidate)), "kind": "numerical observation"}


def run_demo():
    """Compare active restrictions, their removal, and the loss of uniqueness."""
    case = portfolio_case()
    targeted = PortfolioRestrictions(target=Rational(7, 100), cap=Rational(4, 5))
    variants = {
        "interior": (case, PortfolioRestrictions()),
        "target": (case, targeted),
        "target_removed": (case, targeted.without("target")),
        "cap": (case, PortfolioRestrictions(cap=Rational(1, 10))),
        "no_short": (
            portfolio_case(first_variance=1, second_variance=4, covariance=Rational(3, 2)),
            PortfolioRestrictions(),
        ),
        "shorts_allowed": (
            portfolio_case(first_variance=1, second_variance=4, covariance=Rational(3, 2)),
            PortfolioRestrictions().without("long_only"),
        ),
    }
    results = {
        name: {"case": current, "result": allocate(current, restrictions)}
        for name, (current, restrictions) in variants.items()
    }
    flat = portfolio_case(first_variance=1, second_variance=1, covariance=1)
    results["flat"] = {"case": flat, "result": allocate(flat, unique=False)}
    return results


if __name__ == "__main__":
    for name, record in run_demo().items():
        data = record["result"].data
        print(f"{name}: w={data.weight}, variance={data.variance}, active={data.active}, minimizers={data.minimizers}")

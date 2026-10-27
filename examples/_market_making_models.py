"""Research contracts and general Hamiltonian inputs for Guéant's Model A."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from sympy import Expr, Piecewise, S, exp

from examples.market_making import ExponentialIntensity, MarketMakingState, QuoteHamiltonian as DeclaredQuote
from modello import InstanceDummy, component
from modello.contrib.research import (
    Requirement,
    ResearchCase,
    ResearchModel,
    Theorem,
    concave,
    curvature_bound,
    decreasing,
    exact_form,
    increasing,
    nonnegative,
    positive,
    requires,
    smooth,
    vanishes,
)

SOURCE = "https://arxiv.org/html/1605.01862"


def _neighbor_formula(neighbor: str) -> Callable[[Callable[[str], Expr]], Expr]:
    def formula(get: Callable[[str], Expr]) -> Expr:
        return (get("continuation.current") - get(f"continuation.{neighbor}")) / get("step")

    return formula


def _gain_formula(side: str) -> Callable[[Callable[[str], Expr]], Expr]:
    def formula(get: Callable[[str], Expr]) -> Expr:
        return (
            get(f"{side}.intensity.rate")
            * (1 - exp(-get("gamma") * get("step") * (get(f"{side}.intensity.distance") - get(f"{side}.cost"))))
            / get("gamma")
        )

    return formula


def intensity_contracts(prefix="") -> tuple[Requirement, ...]:
    path = f"{prefix}." if prefix else ""
    rate, distance = path + "rate", path + "distance"
    return tuple(
        replace(claim, consumer="quote_uniqueness", reason=reason, source=SOURCE)
        for claim, reason in (
            (positive(rate, wrt=distance), "Trades need a positive arrival intensity."),
            (smooth(rate, wrt=distance), "Guéant's quote theorem requires C2 intensity."),
            (decreasing(rate, wrt=distance), "Increasing distance must strictly reduce the arrival rate."),
            (vanishes(rate, wrt=distance), "The intensity must vanish at large positive quote distance."),
            (curvature_bound(rate, wrt=distance), "A uniform curvature bound establishes a unique maximizer."),
        )
    )


class StudiedExponential(ResearchModel, ExponentialIntensity):
    """Exponential intensity with live requirements, rather than copied verdicts."""

    rate = requires(
        ExponentialIntensity.scale * exp(-ExponentialIntensity.decay * ExponentialIntensity.distance),
        *intensity_contracts(),
    )


class LogisticIntensity(StudiedExponential):
    """A bounded alternative, with peak arrival rate A and decay k."""

    rate = ExponentialIntensity.scale / (1 + exp(ExponentialIntensity.decay * ExponentialIntensity.distance))


class QuoteHamiltonian(ResearchModel, DeclaredQuote):
    """Declare the gain and Hamiltonian input without assuming an optimizer."""

    intensity = component(StudiedExponential)


class CARAMarketMaker(ResearchModel, MarketMakingState):
    """The existing HJB with general Hamiltonian inputs on both trade sides."""

    bid = component(QuoteHamiltonian)
    ask = component(QuoteHamiltonian)
    horizon = InstanceDummy("T", positive=True)


def study_case(*, logistic=False, capacity=5) -> ResearchCase[CARAMarketMaker]:
    """Match both the zero-distance intensity and slope across the two families."""
    case = ResearchCase.for_model(CARAMarketMaker, "market_making")
    if logistic:
        case = case.replace("bid.intensity", LogisticIntensity).replace("ask.intensity", LogisticIntensity)
    for side in ("bid", "ask"):
        conclusion = Requirement(f"unique_{side}", f"{side}.intensity.rate", "theorem")
        case = case.with_theorem(
            Theorem(
                f"gueant_unique_{side}", conclusion, intensity_contracts(f"{side}.intensity"), SOURCE + "#S3.SS2", "1"
            )
        )
        if logistic:
            case = case.with_hint(f"{side}.intensity.rate/curvature_bound", upper=1)
    gamma, eta = S.One / 10, S.One / 20
    scale, decay = (4, 2) if logistic else (2, 1)
    # Constructor-compatible partial components retain shared parameter links.
    declaration = case.problem.model
    bid_type = declaration._modello_namespace.nested_models["bid"]._modello_model
    ask_type = declaration._modello_namespace.nested_models["ask"]._modello_model
    bid_intensity = bid_type._modello_namespace.nested_models["intensity"]._modello_model
    ask_intensity = ask_type._modello_namespace.nested_models["intensity"]._modello_model
    from examples.market_making import QuadraticInventoryRisk, QuadraticTerminalPenalty

    return case.observe(
        gamma=gamma,
        step=1,
        capacity=capacity,
        horizon=1,
        risk=QuadraticInventoryRisk("risk", sigma=1),
        terminal=QuadraticTerminalPenalty("terminal", eta=eta),
        bid=bid_type("bid", intensity=bid_intensity("bid_intensity", scale=scale, decay=decay)),
        ask=ask_type("ask", intensity=ask_intensity("ask_intensity", scale=scale, decay=decay)),
    )


def solver_requirements(case: ResearchCase) -> tuple[Requirement, ...]:
    requirements = [
        exact_form(
            "utility.value",
            lambda get: -exp(-get("gamma") * get("utility.wealth")),
            wrt="utility.wealth",
            name="cara_form",
            consumer="CARA_reduction",
            reason="The reduction uses exact exponential utility.",
            source=SOURCE,
        ),
        increasing(
            "utility.value",
            wrt="utility.wealth",
            name="utility_increasing",
            consumer="CARA_reduction",
            reason="The utility must prefer additional wealth.",
            source=SOURCE,
        ),
        concave(
            "utility.value",
            wrt="utility.wealth",
            name="utility_concave",
            consumer="CARA_reduction",
            reason="The utility must express risk aversion.",
            source=SOURCE,
        ),
    ]
    for name in ("gamma", "step", "capacity", "horizon", "risk.sigma"):
        requirements.append(
            positive(
                name, name=f"positive_{name}", consumer="ODE_solver", reason=f"{name} must be positive.", source=SOURCE
            )
        )
    requirements.append(
        nonnegative(
            "terminal.eta",
            name="terminal_cost",
            consumer="terminal_condition",
            reason="The terminal inventory penalty is nonnegative.",
            source=SOURCE,
        )
    )
    for side in ("bid", "ask"):
        neighbor = "after_bid" if side == "bid" else "after_ask"
        requirements.append(
            exact_form(
                f"continuation.{side}_cost",
                _neighbor_formula(neighbor),
                name=f"{side}_neighbor_cost",
                consumer="inventory_adapter",
                reason="The grid adapter uses this nearest-neighbor continuation difference.",
                source=SOURCE,
            )
        )
        requirements.append(
            exact_form(
                f"{side}.gain",
                _gain_formula(side),
                wrt=f"{side}.intensity.distance",
                name=f"{side}_cara_gain",
                consumer="quote_optimizer",
                reason="The optimizer solves the declared CARA gain, not only the intensity curve.",
                source=SOURCE,
            )
        )
        requirements.append(
            Requirement(
                f"unique_{side}",
                f"{side}.intensity.rate",
                "theorem",
                consumer="quote_optimizer",
                reason="The general optimizer requires a unique quote.",
                source=SOURCE,
            )
        )
    requirements.extend(
        (
            exact_form(
                "bid_term",
                lambda get: Piecewise((get("bid.value"), get("inventory") < get("capacity")), (0, True)),
                name="bid_boundary_gate",
                consumer="inventory_adapter",
                reason="The bid gate disables inventory-increasing trades at Q.",
                source=SOURCE,
            ),
            exact_form(
                "ask_term",
                lambda get: Piecewise((get("ask.value"), get("inventory") > -get("capacity")), (0, True)),
                name="ask_boundary_gate",
                consumer="inventory_adapter",
                reason="The ask gate disables inventory-decreasing trades at -Q.",
                source=SOURCE,
            ),
            exact_form(
                "residual",
                lambda get: -get("continuation.time_derivative") + get("risk.cost") - get("bid_term") - get("ask_term"),
                name="hjb_structure",
                consumer="inventory_adapter",
                reason="The time transformation and linear reference require this HJB structure.",
                source=SOURCE,
            ),
        )
    )
    return tuple(requirements)


def exponential_requirement(side: str) -> Requirement:
    return exact_form(
        f"{side}.intensity.rate",
        lambda get: (
            get(f"{side}.intensity.scale") * exp(-get(f"{side}.intensity.decay") * get(f"{side}.intensity.distance"))
        ),
        wrt=f"{side}.intensity.distance",
        name=f"exponential_{side}",
        consumer="exponential_optimizer",
        reason="The specialized formula requires exact exponential intensity.",
        source=SOURCE,
    )

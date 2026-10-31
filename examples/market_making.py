"""Explore Guéant's continuous-time CARA market maker and its component requirements.

Run ``python -m examples.market_making`` for the lightweight baseline, or add
``--study --output DIRECTORY`` for the time-dependent researcher workflow.
"""

from __future__ import annotations

from dataclasses import replace

from sympy import Piecewise, Subs, exp, log, simplify

from examples.portfolio.optimal_execution import InventoryRisk
from modello import Coordinate, InstanceDummy, Modello, component
from modello.contrib.research import (
    Requirement,
    ResearchModel,
    concave,
    convex,
    curvature_bound,
    decreasing,
    increasing,
    nonnegative,
    positive,
    requires,
    smooth,
    vanishes,
)

SOURCE = "https://arxiv.org/html/1605.01862"


def intensity_contracts(prefix="") -> tuple[Requirement, ...]:
    """Use the same intensity hypotheses for declarations and quote theorems."""
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


class CARAUtility(ResearchModel):
    """Increasing, concave utility of wealth; gamma is inverse currency."""

    wealth = Coordinate("w", real=True)
    gamma = requires(
        InstanceDummy("gamma", positive=True),
        positive("gamma", consumer="CARA_reduction", reason="The CARA reduction uses positive risk aversion."),
    )
    value = requires(
        -exp(-gamma * wealth),
        smooth("value", wrt="wealth", consumer="CARA_reduction", reason="The reduction uses price curvature."),
        increasing("value", wrt="wealth", consumer="valuation", reason="More wealth must be preferred."),
        concave("value", wrt="wealth", consumer="valuation", reason="The utility expresses risk aversion."),
    )


class ExponentialIntensity(ResearchModel):
    """Trade arrival rate A exp(-k delta), for any real quote distance."""

    distance = Coordinate("delta", real=True)
    scale = InstanceDummy("A", positive=True)
    decay = InstanceDummy("k", positive=True)
    rate = requires(scale * exp(-decay * distance), *intensity_contracts())


class QuadraticInventoryRisk(ResearchModel, InventoryRisk):
    """Reuse the execution example's gamma sigma² q² / 2 diffusion cost."""

    cost = requires(
        InventoryRisk.gamma * InventoryRisk.variance / 2,
        nonnegative("cost", wrt="inventory", reason="Inventory risk is a nonnegative running cost."),
        convex("cost", wrt="inventory", reason="The risk cost is convex in signed inventory."),
        smooth("cost", wrt="inventory", reason="The diffusion contribution is C2 in inventory."),
    )


class QuadraticTerminalPenalty(ResearchModel):
    """A smooth convex terminal cost of signed inventory."""

    inventory = Coordinate("q", real=True)
    eta = InstanceDummy("eta", nonnegative=True)
    cost = requires(
        eta * inventory**2,
        nonnegative("cost", wrt="inventory", reason="The terminal inventory penalty is nonnegative."),
        convex("cost", wrt="inventory", reason="Both long and short inventories carry a convex cost."),
        smooth("cost", wrt="inventory", reason="The terminal cost is C2 in inventory."),
    )


class ContinuationValue(Modello):
    """One state of theta(t, q) on the inventory lattice, with time continuous."""

    step = InstanceDummy("Delta", positive=True)
    current = InstanceDummy("theta", real=True)
    after_bid = InstanceDummy("theta_plus", real=True)
    after_ask = InstanceDummy("theta_minus", real=True)
    time_derivative = InstanceDummy("theta_t", real=True)
    bid_cost = (current - after_bid) / step
    ask_cost = (current - after_ask) / step


class QuoteHamiltonian(ResearchModel):
    """A CARA gain function and an independently supplied Hamiltonian value."""

    intensity = component(ExponentialIntensity)
    distance = ExponentialIntensity.distance
    gamma = InstanceDummy("gamma", positive=True)
    step = InstanceDummy("Delta", positive=True)
    cost = InstanceDummy("p", real=True)
    value = InstanceDummy("H", real=True)
    gain = intensity.rate * (1 - exp(-gamma * step * (distance - cost))) / gamma


class ExponentialQuote(QuoteHamiltonian):
    """Exact optimizer for exponential intensity, gamma > 0 and step > 0.

    Quote distances may be negative; restricting them changes the optimizer.
    """

    optimal_distance = QuoteHamiltonian.cost + log(
        1 + QuoteHamiltonian.gamma * QuoteHamiltonian.step / QuoteHamiltonian.intensity.decay
    ) / (QuoteHamiltonian.gamma * QuoteHamiltonian.step)
    value = Subs(QuoteHamiltonian.gain, QuoteHamiltonian.distance, optimal_distance)


class MarketMakingState(ResearchModel):
    """A single-asset CARA HJB state, accepting general Hamiltonian values.

    Observe residual=0 to infer theta_t from neighboring continuation values.
    The inventory belongs to {-Q, -Q+Delta, ..., Q}. This slice does not solve
    the coupled terminal-value ODE system or validate the lattice membership.
    """

    utility = component(CARAUtility)
    risk = component(QuadraticInventoryRisk)
    terminal = component(QuadraticTerminalPenalty)
    continuation = component(ContinuationValue)
    bid = component(QuoteHamiltonian)
    ask = component(QuoteHamiltonian)

    gamma = utility.gamma
    gamma = risk.gamma
    gamma = bid.gamma
    gamma = ask.gamma
    step = continuation.step
    step = bid.step
    step = ask.step
    bid_cost = continuation.bid_cost
    bid_cost = bid.cost
    ask_cost = continuation.ask_cost
    ask_cost = ask.cost

    capacity = InstanceDummy("Q", positive=True)
    inventory = risk.inventory
    bid_term = Piecewise((bid.value, inventory < capacity), (0, True))
    ask_term = Piecewise((ask.value, inventory > -capacity), (0, True))
    residual = -continuation.time_derivative + risk.cost - bid_term - ask_term
    terminal_value = -Subs(terminal.cost, QuadraticTerminalPenalty.inventory, inventory)


class MarketMaker(MarketMakingState):
    """Specialize the shared HJB to the exact exponential quote formulas."""

    bid = component(ExponentialQuote)
    ask = component(ExponentialQuote)


def main() -> None:
    """Print the definition checks and one exact interior HJB calculation."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", action="store_true", help="Run the complete time-dependent research comparison")
    parser.add_argument("--output", type=Path, default=Path("artefacts/market-making"))
    arguments = parser.parse_args()
    if arguments.study:
        from examples.market_making_study import write_study

        write_study(arguments.output)
        print(f"Research report: {arguments.output / 'report.html'}")
        return
    print(MarketMaker.definition_report())
    quote = ExponentialQuote(
        "quote", intensity=ExponentialIntensity("liquidity", scale=2, decay=1), gamma=1, step=1, cost=0
    )
    print(f"optimal distance: {quote.optimal_distance}; Hamiltonian: {simplify(quote.value)}")
    maker = MarketMaker(
        "interior",
        gamma=1,
        step=1,
        capacity=1,
        inventory=0,
        residual=0,
        risk=QuadraticInventoryRisk("risk", sigma=1),
        continuation=ContinuationValue("theta", current=0, after_bid=0, after_ask=0),
        bid=quote,
        ask=quote,
    )
    print(f"HJB time derivative at q=0: {maker.continuation.time_derivative}")


if __name__ == "__main__":
    main()

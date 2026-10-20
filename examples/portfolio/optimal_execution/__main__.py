"""Run the continuous-model execution and inverse-calibration studies."""

from sympy import Rational

from examples.portfolio.optimal_execution import ExecutionDensity, ImpactObservation
from examples.portfolio.sources import Fill, timestamp


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

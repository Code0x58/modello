"""Separate observed trading costs from inverse assumptions about preferences."""

from dataclasses import replace

import pytest
from sympy import Rational, S, Symbol

from examples.portfolio.optimal_execution import CrowdedImpact, ExecutionDensity, ImpactObservation
from examples.portfolio.sources import Fill, timestamp


def test_trade_bucket_aggregation_and_fitting_generated_impact_costs():
    at = timestamp("2026-01-07T09:00:00Z")
    first = Fill("one", "research", at, at, "msft", -4, 99.75, 1)
    second = replace(first, id="two")
    bucket = ImpactObservation.from_fills((first, second), reference_price=100, duration=1, volume=64)
    assert (bucket.quantity, bucket.cost) == (8, 2)  # Fees are deliberately excluded.
    observations = (bucket, ImpactObservation(16, 1, 64, 8), ImpactObservation(4, 1, 64, 0.5))
    model = ExecutionDensity("fit")
    fitted = model.fit_impact(observations)
    assert fitted.estimates["eta"] == 2
    assert fitted.loss == 0
    with pytest.raises(ValueError, match="identify"):
        model.fit_impact(observations, parameters=("eta", "gamma"))
    noisy = model.fit_impact((bucket, ImpactObservation(16, 1, 64, 8.1)))
    assert 2 < noisy.estimates["eta"] < 2.03
    assert noisy.loss > 0


def test_inverse_policy_identification_and_scale_ambiguity():
    actual = ExecutionDensity("actual", eta=2, sigma=3, gamma=Rational(1, 1000), volume=100)
    observed = actual.discretize(10, (1, 1, 1)).stationary().inventory
    model = ExecutionDensity("inverse", eta=2, sigma=3, volume=100)
    fitted = model.discretize(10, (1, 1, 1)).fit_policy(observed)
    assert fitted.estimates["gamma"] == Rational(1, 1000)
    assert fitted.loss == 0
    floating = model.discretize(10, (1, 1, 1)).fit_policy(tuple(float(value) for value in observed))
    assert float(floating.estimates["gamma"]) == pytest.approx(0.001)
    scaled = ExecutionDensity("scaled", eta=4, sigma=3, gamma=Rational(2, 1000), volume=100)
    assert scaled.discretize(10, (1, 1, 1)).stationary().inventory == observed
    unknown = ExecutionDensity("both", sigma=3, volume=100).discretize(10, (1, 1, 1))
    with pytest.raises(ValueError, match="identify"):
        unknown.fit_policy(observed, parameters=("eta", "gamma"))
    with pytest.raises(ValueError, match="Bind other"):
        unknown.fit_policy(observed)
    with pytest.raises(ValueError, match="fixed endpoints"):
        model.discretize(10, (1, 1, 1)).fit_policy((10, 5, 0))
    with pytest.raises(ValueError, match="decreasing"):
        model.discretize(10, (1, 1, 1)).fit_policy((10, 11, 5, 0))


def test_two_coefficient_fit_follows_a_changed_impact_definition():
    model = CrowdedImpact("calibrate")
    fitted = model.fit_impact(
        (ImpactObservation(10, 1, 100, 2.02), ImpactObservation(20, 1, 100, 8.32)),
        parameters=("eta", "beta"),
    )
    assert fitted.estimates["eta"] == 2
    assert fitted.estimates["beta"] == 2
    assert fitted.loss == 0


@pytest.mark.parametrize(
    "quantity,duration,volume,cost", [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, S.Infinity)]
)
def test_invalid_impact_observations(quantity, duration, volume, cost):
    with pytest.raises(ValueError):
        ImpactObservation(quantity, duration, volume, cost)


def test_reject_mixed_trade_sides_and_impossible_fit():
    at = timestamp("2026-01-07T09:00:00Z")
    first = Fill("one", "research", at, at, "msft", -4, 99.75, 1)
    with pytest.raises(ValueError, match="trade side"):
        ImpactObservation.from_fills((first, replace(first, quantity=4)), reference_price=100, duration=1, volume=64)
    with pytest.raises(ValueError, match="parameter domain"):
        ExecutionDensity("negative").fit_impact((ImpactObservation(10, 1, 100, -2),))
    with pytest.raises(ValueError, match="independent unknown"):
        ExecutionDensity("known", eta=2).fit_impact((ImpactObservation(10, 1, 100, 2),))
    with pytest.raises(ValueError, match="Bind other"):
        ExecutionDensity("unknown", eta=Symbol("a") * Symbol("b")).fit_impact(
            (ImpactObservation(10, 1, 100, 2),), parameters=("gamma",)
        )

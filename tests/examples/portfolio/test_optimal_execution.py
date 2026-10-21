"""Independent execution identities, discretisation, sensitivities, and calibration."""

from __future__ import annotations

from dataclasses import replace

import pytest
from sympy import Function, Rational, S, Symbol, diff, simplify, sinh, sqrt

from examples.data import NamedMap
from examples.portfolio.optimal_execution import CrowdedImpact, ExecutionDensity, ImpactObservation
from examples.portfolio.sources import Fill, timestamp
from modello import InstanceDummy


def test_variation_and_continuous_boundary_value_solution():
    model = ExecutionDensity("execution", eta=2, sigma=3, gamma=Rational(1, 1000), volume=100)
    time = Symbol("t", real=True)
    q = Function("q")(time)
    equation = model.euler_equation(time, q)
    assert simplify(equation.lhs - (9 * q / 1000 - diff(q, time, 2) / 25)) == 0
    curve = model.continuous_curve(10, 4).rhs
    assert simplify(curve.subs(time, 0)) == 10
    assert simplify(curve.subs(time, 4)) == 0
    assert simplify(equation.lhs.subs(q, curve).doit()) == 0
    kappa = sqrt(Rational(9, 40))
    analytic = 10 * sinh(kappa * (4 - time)) / sinh(4 * kappa)
    for at in (0, 1, 2, 3, 4):
        assert float(curve.subs(time, at)) == pytest.approx(float(analytic.subs(time, at)))


def test_exact_segment_integration_and_discrete_optimality():
    model = ExecutionDensity("execution", eta=2, sigma=3, gamma=Rational(1, 1000), volume=100)
    opening, closing = Symbol("a", real=True), Symbol("b", real=True)
    duration = Symbol("h", positive=True)
    segment = model.interval(opening, closing, duration)
    assert simplify(segment.impact - 2 * (opening - closing) ** 2 / (100 * duration)) == 0
    assert simplify(segment.variance - 9 * duration * (opening**2 + opening * closing + closing**2) / 3) == 0
    grid = model.discretize(10, (1, 1, 1, 1))
    optimum = grid.stationary()
    assert optimum.residuals == (0, 0, 0)
    assert grid.hessian.is_positive_definite is True
    assert all(float(quantity) > 0 for quantity in optimum.trades)
    assert sum(optimum.trades) == 10
    uniform = dict(zip(grid.unknowns, (S(15) / 2, S(5), S(5) / 2), strict=True))
    assert optimum.costs.objective < grid.objective.subs(uniform)
    assert optimum.costs.risk == optimum.costs.variance / 2000


def test_zero_risk_limit_and_volume_time_invariance():
    model = ExecutionDensity("neutral", eta=2, sigma=3, gamma=0)
    time = Symbol("t", real=True)
    assert simplify(model.continuous_curve(10, 4).rhs - (10 - S(5) * time / 2)) == 0
    grid = model.discretize(12, (1, 2, 1), volumes=(100, 100, 200))
    assert grid.stationary().trades == (S(12) / 5, S(24) / 5, S(24) / 5)
    assert grid.stationary().costs.risk == 0
    assert model.discretize(10, (1,), volumes=(100,)).stationary().inventory == (10, 0)


def test_mesh_refinement_converges_to_the_continuous_solution():
    model = ExecutionDensity("execution", eta=2, sigma=3, gamma=Rational(1, 1000), volume=100)
    kappa = sqrt(Rational(9, 40))
    midpoint = float(10 * sinh(2 * kappa) / sinh(4 * kappa))
    errors = []
    for intervals in (2, 4, 8):
        result = model.discretize(10, (Rational(4, intervals),) * intervals).stationary()
        errors.append(abs(float(result.inventory[intervals // 2]) - midpoint))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < errors[0] / 10


def test_implicit_sensitivity_and_value_envelope():
    quantity = Symbol("Q", positive=True)
    model = ExecutionDensity("preference", eta=2, sigma=3, volume=100)
    grid = model.discretize(quantity, (1, 1))
    solution = grid.stationary()
    sensitivity = solution.sensitivity(model.gamma)
    assert simplify(sensitivity[0] - diff(solution.interior[0], model.gamma)) == 0
    assert sensitivity[0].subs(model.gamma, Rational(1, 1000)) < 0
    value = solution.costs.objective
    envelope = diff(grid.objective, model.gamma).subs(solution.substitutions)
    assert simplify(diff(value, model.gamma) - envelope) == 0
    assert simplify(diff(value, model.gamma) - solution.costs.variance / 2) == 0
    assert simplify(diff(value, quantity) - 2 * value / quantity) == 0


def test_changing_the_impact_law_updates_variation_integration_and_nonlinear_solve():
    base = ExecutionDensity("base", eta=2, sigma=3, gamma=Rational(1, 1000), volume=100)
    crowded = CrowdedImpact("crowded", eta=2, beta=200, sigma=3, gamma=Rational(1, 1000), volume=100)
    time = Symbol("t", real=True)
    q = Function("q")(time)
    extra = crowded.euler_equation(time, q).lhs - base.euler_equation(time, q).lhs
    assert simplify(extra + S(3) * diff(q, time) ** 2 * diff(q, time, 2) / 1250) == 0
    grid = crowded.discretize(10, (1, 1))
    with pytest.raises(ValueError, match="Nonlinear stationarity"):
        grid.stationary()
    solution = grid.stationary(guess=(4,))
    assert abs(float(solution.residuals[0])) < 1e-12
    assert float(grid.hessian.subs(solution.substitutions)[0, 0]) > 0
    assert all(float(quantity) > 0 for quantity in solution.trades)
    assert float(solution.interior[0]) > float(base.discretize(10, (1, 1)).stationary().interior[0])


def test_nonlinear_implicit_sensitivity_uses_the_unbound_model_at_a_parameter_point():
    model = CrowdedImpact("sensitivity", eta=2, sigma=3, gamma=Rational(1, 1000), volume=100)
    grid = model.discretize(10, (1, 1))
    solution = grid.stationary(at=NamedMap({"beta": 200.0}), guess=(4,))
    derivative = float(solution.sensitivity(model.beta)[0])
    lower = grid.stationary(at=NamedMap({"beta": 199.99}), guess=(4,)).interior[0]
    upper = grid.stationary(at=NamedMap({"beta": 200.01}), guess=(4,)).interior[0]
    assert derivative > 0
    assert derivative == pytest.approx(float((upper - lower) / 0.02), rel=1e-7)
    with pytest.raises(ValueError, match="declared domains"):
        grid.stationary(at=NamedMap({"beta": -1.0}), guess=(4,))


def test_a_changed_objective_is_integrated_instead_of_reconstructed_from_cost_columns():
    class SignalAware(ExecutionDensity):
        alpha = InstanceDummy("alpha", real=True)
        objective = ExecutionDensity.impact + ExecutionDensity.risk - alpha * ExecutionDensity.inventory

    model = SignalAware("signal", eta=2, volume=100, sigma=3, gamma=Rational(1, 1000), alpha=Rational(1, 10))
    interval = model.interval(10, 0, 2)
    assert interval.objective - interval.impact - interval.risk == -1


@pytest.mark.parametrize("durations,volumes", [((), ()), ((0,), (100,)), ((1,), (0,)), ((1, 1), (100,))])
def test_invalid_execution_mesh(durations, volumes):
    with pytest.raises(ValueError):
        ExecutionDensity("invalid").discretize(10, durations, volumes=volumes)


def test_transforms_require_symbolic_state_and_explicit_volume_binding():
    with pytest.raises(ValueError, match="Keep inventory and speed"):
        ExecutionDensity("bound", inventory=10).discretize(10, (1,)).objective
    with pytest.raises(ValueError, match="Keep volume unbound"):
        ExecutionDensity("volume", volume=100).discretize(10, (1,), volumes=(200,)).objective
    with pytest.raises(ValueError, match="horizon"):
        ExecutionDensity("invalid").continuous_curve(10, 0)


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
